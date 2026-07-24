"""Single-dataset and dataset-matrix experiment orchestration."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any

import yaml

from .backends import Backend, BackendFailure, create_backend
from .configuration import dataset_config, resolve, selected_entries
from .events import ErrorLog, Events, make_events
from .inputs import iter_rows_for_source, rows_for_source, source_provenance
from .outputs import ResumeIndex, create_output_store
from .processors import GenerationRequest, ProcessedResult, Processor, RawModelResponse
from .prompting import (
    materialize_variant,
    prompt_group_id,
    read_asset,
    render,
    rendered_prompt,
    retry_values,
    system_prompt,
    variant_config_hash,
    variant_schema,
)
from .results import CONTRACT_VERSION, TOOL_VERSION, error_kind, result_from_processed, serialise
from .runtime import apply_resource_guard, gpu, sync_cuda

BackoffCallback = Callable[[BackendFailure, int, int], None]
AppendCallback = Callable[[dict[str, Any]], None]


def generation_request(
    config: Mapping[str, Any],
    variant: Mapping[str, Any],
    processor: Processor,
    schema: Mapping[str, Any] | None,
) -> GenerationRequest:
    """Compile a processor's evidence needs into a backend-neutral request."""

    options = {key: value for key, value in variant.items() if not key.startswith("_")}
    return GenerationRequest(
        variant_id=str(variant["id"]),
        requirements=processor.requirements,
        system_prompt=system_prompt(dict(config), dict(variant), dict(schema) if schema is not None else None),
        options=options,
    )


def tune(
    backend: Backend,
    request: GenerationRequest,
    prompts: Sequence[str],
    batch: Mapping[str, Any],
    events: Events,
    synchronize_cuda: bool,
) -> tuple[int, list[dict[str, Any]]]:
    """Select the highest-throughput safe configured batch size."""

    if batch.get("mode", "auto") == "fixed":
        return int(batch.get("size", batch.get("candidates", [1])[0])), []
    attempts: list[dict[str, Any]] = []
    safe: list[tuple[float, int]] = []
    candidates = sorted({int(item) for item in batch.get("candidates", [1])})
    warmup_rows = int(batch.get("warmup_rows", 64))
    warmup = list(prompts[: max(warmup_rows, *candidates)] or prompts[:1])
    for size in candidates:
        sync_cuda(synchronize_cuda)
        started = time.perf_counter()
        try:
            responses = backend.generate(warmup[:size], request)
            sync_cuda(synchronize_cuda)
            elapsed = max(time.perf_counter() - started, 1e-9)
            record = {
                "candidate": size,
                "accepted": True,
                "rows_per_second": min(size, len(warmup)) / elapsed,
                "tokens_per_second": sum(response.token_count for response in responses) / elapsed,
                "latency_seconds": elapsed,
                "gpu": gpu(),
            }
            safe.append((float(record["rows_per_second"]), size))
        except BackendFailure as exc:
            record = {"candidate": size, "accepted": False, "error": str(exc), "gpu": gpu()}
            events.emit("batch_candidate_rejected", variant=request.variant_id, **record)
        attempts.append(record)
        events.emit("batch_candidate", variant=request.variant_id, **record)
    if not safe:
        raise RuntimeError(f"No safe batch size for {request.variant_id}")
    selected = max(safe)[1]
    events.emit("batch_selected", variant=request.variant_id, batch_size=selected, candidates=attempts)
    return selected, attempts


def generate_with_backoff(
    backend: Backend,
    request: GenerationRequest,
    prompts: Sequence[str],
    batch_config: Mapping[str, Any],
    on_backoff: BackoffCallback | None = None,
) -> tuple[list[RawModelResponse], int]:
    """Generate a batch, halving it on batch-level backend failures."""

    active = list(prompts)
    minimum = int(batch_config.get("min_size", 1))
    while True:
        try:
            responses = backend.generate(active, request)
            if len(responses) != len(active):
                raise BackendFailure(f"Backend returned {len(responses)} responses for {len(active)} prompts")
            return responses, len(active)
        except BackendFailure as exc:
            if len(active) <= minimum:
                raise
            new_size = max(minimum, len(active) // 2)
            if on_backoff is not None:
                on_backoff(exc, len(active), new_size)
            active = active[:new_size]


def _counted(rows: Iterable[dict[str, Any]], counter: list[int]) -> Iterator[dict[str, Any]]:
    for row in rows:
        counter[0] += 1
        yield row


def _error_strings(processed: ProcessedResult) -> list[str]:
    return [error.as_contract_string() for error in processed.errors]


def _retry_request(request: GenerationRequest, max_tokens: int) -> GenerationRequest:
    requirements = replace(request.requirements, max_tokens=max_tokens)
    return replace(request, requirements=requirements)


def _write_processed_result(
    config: dict[str, Any],
    processed: ProcessedResult,
    *,
    run_id: str,
    variant_id: str,
    config_hash: str,
    group_id: str,
    row: dict[str, Any],
    prompt_text: str,
    response: RawModelResponse,
    attempt_count: int,
    batch_size: int,
    latency_seconds: float,
    rows_per_second: float,
    gpu_snapshot: str,
) -> dict[str, Any]:
    return result_from_processed(
        config,
        processed,
        run_id=run_id,
        variant_id=variant_id,
        config_hash=config_hash,
        group_id=group_id,
        row=row,
        prompt_text=prompt_text,
        raw=response.text,
        attempt_count=attempt_count,
        token_count=response.token_count,
        batch_size=batch_size,
        latency_seconds=latency_seconds,
        rows_per_second=rows_per_second,
        gpu_snapshot=gpu_snapshot,
    )


def run(
    config: dict[str, Any],
    backend: Backend | None = None,
    row_limit: int | None = None,
) -> dict[str, Any]:
    """Execute all variants for one normalized input source."""

    if backend is None and config["model"].get("backend") == "local_vllm" and "_resource_guard" not in config:
        apply_resource_guard(config)
    streaming = bool(config.get("streaming", {}).get("enabled", False))
    run_id = str(config["run"]["id"])
    run_dir = resolve(config, config["output"]["directory"])
    run_dir.mkdir(parents=True, exist_ok=True)
    output_store = create_output_store(config["output"], run_dir)
    events = make_events(config, run_dir, run_id)

    expected_hashes = {
        str(variant["id"]): variant_config_hash(config, materialize_variant(config, variant))
        for variant in config["variants"]
    }
    fingerprint = hashlib.sha256(json.dumps(expected_hashes, sort_keys=True).encode()).hexdigest()
    index = ResumeIndex(run_dir / ".resume.sqlite", fingerprint, store=output_store)
    if index.cleared:
        output_store.discard(str(item["id"]) for item in config["variants"])
    part_files = output_store.part_paths()
    seed_paths = part_files or ([output_store.result_path()] if output_store.result_path().exists() else [])
    seeded = index.seed_from(seed_paths, expected_hashes)

    materialised = None if streaming else rows_for_source(config, config["input"], row_limit)
    pulled = [0]
    total_input = len(materialised) if materialised is not None else 0
    input_counted = materialised is not None
    total_results = seeded
    selected: dict[str, Any] = {}
    retry_settings = config.get("validation", {}).get("retry", {})
    correction_path = retry_settings.get("correction_prompt")
    correction = read_asset(config, correction_path) if correction_path else None
    created_backend = False
    merged = 0

    try:
        for configured_variant in config["variants"]:
            variant = materialize_variant(config, configured_variant)
            processor: Processor = variant["_processor"]
            variant_id = str(variant["id"])
            config_hash = expected_hashes[variant_id]
            schema = variant_schema(config, variant)
            request = generation_request(config, variant, processor, schema)
            max_tokens = request.requirements.max_tokens
            allow_retry = bool(correction and retry_settings.get("enabled")) and not request.requirements.one_token

            if materialised is not None:
                source: Iterable[dict[str, Any]] = iter(materialised)
            else:
                source = iter_rows_for_source(config, config["input"], row_limit)
                if not input_counted:
                    source = _counted(source, pulled)
            source_iter = iter(processor.prepare_rows(source))

            batch_config = dict(config.get("batch", {}))
            declared = [int(item) for item in batch_config.get("candidates", [1])]
            maximum = int(config["model"].get("max_num_seqs", max(declared)))
            batch_config["candidates"] = [item for item in declared if item <= maximum] or [maximum]
            prefetch_count = max(
                int(batch_config.get("warmup_rows", 64)),
                *[int(item) for item in batch_config["candidates"]],
            )
            prefetched = [dict(row) for row in itertools.islice(source_iter, prefetch_count)]
            if not prefetched:
                continue

            pending_rows: int | None = None
            if materialised is not None:
                prepared_again = processor.prepare_rows(iter(materialised))
                pending_rows = sum(
                    1
                    for row in prepared_again
                    if not index.contains(
                        (variant_id, str(row[config["input"]["id_column"]]), int(row["_source_position"]))
                    )
                )
                if pending_rows == 0:
                    events.emit("variant_resumed", variant=variant_id, skipped=len(materialised))
                    continue

            if backend is None:
                backend = create_backend(config["model"], config.get("_resource_guard"))
                created_backend = True

            tune_prompts = [rendered_prompt(config, variant, row, schema) for row in prefetched]
            size, attempts = tune(
                backend,
                request,
                tune_prompts,
                batch_config,
                events,
                bool(config["model"].get("synchronize_cuda", False)),
            )
            size = min(size, maximum)
            group_id = prompt_group_id(config, variant, schema)
            selected[variant_id] = {
                "selected_batch_size": size,
                "prompt_group_id": group_id,
                "tuning": attempts,
            }
            if pending_rows is not None:
                selected[variant_id]["pending_rows"] = pending_rows
            events.emit("variant_started", variant=variant_id, prompt_group_id=group_id, batch_size=size)

            writer = output_store.part_writer(
                variant_id,
                int(config.get("streaming", {}).get("output_chunk_rows", 4096)),
            )
            errors_log = ErrorLog(run_dir / "errors" / f"variant={variant_id}.jsonl")
            deferred_retries: list[dict[str, Any]] = []
            pending_index_keys: list[tuple[str, str, int]] = []

            def record_error(
                *,
                stage: str,
                row: dict[str, Any],
                response: RawModelResponse,
                errors: list[str],
                attempt_count: int,
                current_batch_size: int,
                current_max_tokens: int,
            ) -> None:
                errors_log.write(
                    run_id=run_id,
                    dataset_id=config["run"].get("dataset_id", "default"),
                    variant_id=variant_id,
                    input_row_id=str(row[config["input"]["id_column"]]),
                    source_position=int(row["_source_position"]),
                    stage=stage,
                    error_kind=error_kind(errors, response, current_max_tokens),
                    validation_errors=errors,
                    attempt_count=attempt_count,
                    batch_size=current_batch_size,
                    max_tokens=current_max_tokens,
                    token_count=response.token_count,
                    raw_response=response.text,
                )

            def append_output(output: dict[str, Any]) -> None:
                if output["final_status"] != "failed_backend":
                    pending_index_keys.append((variant_id, str(output["input_row_id"]), int(output["source_position"])))
                if writer.append(output):
                    for key in pending_index_keys:
                        index.add(key)
                    pending_index_keys.clear()

            def on_backoff(exc: BackendFailure, old_size: int, new_size: int) -> None:
                events.emit(
                    "batch_runtime_backoff",
                    variant=variant_id,
                    old_batch_size=old_size,
                    new_batch_size=new_size,
                    error=str(exc),
                )
                errors_log.write(
                    run_id=run_id,
                    dataset_id=config["run"].get("dataset_id", "default"),
                    variant_id=variant_id,
                    stage="initial_batch_backoff",
                    error_kind="backend_batch_failure",
                    error=str(exc),
                    failed_batch_size=old_size,
                    new_batch_size=new_size,
                )

            pending_rows_iter = itertools.chain(prefetched, source_iter)
            buffer: list[dict[str, Any]] = []
            try:
                while True:
                    while len(buffer) < size:
                        try:
                            row = dict(next(pending_rows_iter))
                        except StopIteration:
                            break
                        key = (
                            variant_id,
                            str(row[config["input"]["id_column"]]),
                            int(row["_source_position"]),
                        )
                        if not index.contains(key):
                            buffer.append(row)
                    if not buffer:
                        break
                    chunk = buffer[:size]
                    prompts = [rendered_prompt(config, variant, row, schema) for row in chunk]
                    sync = bool(config["model"].get("synchronize_cuda", False))
                    sync_cuda(sync)
                    started = time.perf_counter()
                    responses, used = generate_with_backoff(backend, request, prompts, batch_config, on_backoff)
                    if used < len(chunk):
                        size = used
                        selected[variant_id]["runtime_batch_size"] = size
                        chunk = chunk[:used]
                        prompts = prompts[:used]
                    sync_cuda(sync)
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    snapshot = serialise(gpu(max_age_seconds=5.0)) or "{}"

                    for row, current_prompt, response in zip(chunk, prompts, responses):
                        attempt_count = 1
                        response_for_output = response
                        processed = processor.process(response, row)
                        errors = _error_strings(processed)
                        if response.backend_error:
                            record_error(
                                stage="backend_response",
                                row=row,
                                response=response,
                                errors=errors,
                                attempt_count=attempt_count,
                                current_batch_size=size,
                                current_max_tokens=max_tokens,
                            )

                        if errors and allow_retry and retry_settings.get("deferred", False):
                            record_error(
                                stage="initial_validation",
                                row=row,
                                response=response,
                                errors=errors,
                                attempt_count=attempt_count,
                                current_batch_size=size,
                                current_max_tokens=max_tokens,
                            )
                            deferred_retries.append(
                                {
                                    "row": row,
                                    "prompt": current_prompt,
                                    "response": response,
                                    "errors": errors,
                                    "attempt_count": attempt_count,
                                }
                            )
                            continue

                        if errors and allow_retry:
                            events.emit(
                                "retry_started",
                                variant=variant_id,
                                input_row_id=str(row[config["input"]["id_column"]]),
                            )
                            current_response = response
                            for _ in range(int(retry_settings.get("max_attempts", 0))):
                                retry_prompt = render(
                                    correction,
                                    retry_values(config, variant, row, schema, current_response.text, errors),
                                )
                                current_response = backend.generate([retry_prompt], request)[0]
                                processed = processor.process(current_response, row)
                                errors = _error_strings(processed)
                                attempt_count += 1
                                if not errors:
                                    break
                            response_for_output = current_response
                            events.emit(
                                "retry_completed",
                                variant=variant_id,
                                input_row_id=str(row[config["input"]["id_column"]]),
                                attempts=attempt_count,
                                validation_status="valid" if not errors else "invalid",
                            )

                        append_output(
                            _write_processed_result(
                                config,
                                processed,
                                run_id=run_id,
                                variant_id=variant_id,
                                config_hash=config_hash,
                                group_id=group_id,
                                row=row,
                                prompt_text=current_prompt,
                                response=response_for_output,
                                attempt_count=attempt_count,
                                batch_size=size,
                                latency_seconds=elapsed / len(chunk),
                                rows_per_second=len(chunk) / elapsed,
                                gpu_snapshot=snapshot,
                            )
                        )
                        total_results += 1
                    index.connection.commit()
                    events.emit(
                        "batch_completed",
                        variant=variant_id,
                        rows=len(chunk),
                        rows_per_second=len(chunk) / elapsed,
                    )
                    buffer = buffer[len(chunk) :]

                if not input_counted:
                    total_input = pulled[0]
                    input_counted = True

                total_results += run_deferred_retries(
                    config,
                    deferred_retries,
                    backend=backend,
                    events=events,
                    index=index,
                    errors_log=errors_log,
                    append_output=append_output,
                    request=request,
                    processor=processor,
                    schema=schema,
                    variant=variant,
                    correction=correction,
                    batch_config=batch_config,
                    size=size,
                    run_id=run_id,
                    config_hash=config_hash,
                    group_id=group_id,
                )

                if writer.flush():
                    for key in pending_index_keys:
                        index.add(key)
                    pending_index_keys.clear()
                index.connection.commit()
            finally:
                writer.close()
                errors_log.close()
            events.emit("variant_completed", variant=variant_id)
        merged = output_store.finalize(expected_hashes, index.retryable_keys)
    finally:
        if created_backend and backend is not None:
            backend.close()
        index.close()

    effective = {key: value for key, value in config.items() if not key.startswith("_")}
    (run_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(effective, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "tool_version": TOOL_VERSION,
        "run_id": run_id,
        "model_id": str(config["model"]["name"]),
        "dataset_id": config["run"].get("dataset_id", "default"),
        "input_rows": total_input,
        "result_rows": merged or total_results,
        "model": config["model"],
        "effective_config": effective,
        "variants": selected,
        "cpu_resource_guard": config.get("_resource_guard"),
        "gpu_preflight": gpu(),
        "source_provenance": source_provenance(config),
        "resume_skipped_rows": seeded,
        "event_log": str(events.path),
        "output": {
            "format": output_store.format_name,
            "result_path": str(output_store.result_path()),
            "variant_paths": {
                str(variant["id"]): str(output_store.variant_path(str(variant["id"]))) for variant in config["variants"]
            },
        },
    }
    if streaming:
        manifest["streaming"] = True
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    events.emit("run_completed", result_rows=manifest["result_rows"], gpu=gpu())
    events.close()
    return manifest


def run_deferred_retries(
    config: dict[str, Any],
    queued: list[dict[str, Any]],
    *,
    backend: Backend,
    events: Events,
    index: ResumeIndex,
    errors_log: ErrorLog,
    append_output: AppendCallback,
    request: GenerationRequest,
    processor: Processor,
    schema: dict[str, Any] | None,
    variant: dict[str, Any],
    correction: str | None,
    batch_config: dict[str, Any],
    size: int,
    run_id: str,
    config_hash: str,
    group_id: str,
) -> int:
    """Retry invalid structured responses in smaller, higher-budget batches."""

    if not queued or correction is None:
        return 0
    variant_id = str(variant["id"])
    retry_settings = config.get("validation", {}).get("retry", {})
    max_attempts = int(retry_settings.get("max_attempts", 0))
    retry_divisor = max(2, int(retry_settings.get("batch_size_divisor", 2)))
    token_multiplier = max(1, int(retry_settings.get("max_tokens_multiplier", 2)))
    token_cap = int(retry_settings.get("max_tokens_cap", 256))
    retry_batch = max(int(batch_config.get("min_size", 1)), size // retry_divisor)
    written = 0

    for retry_round in range(1, max_attempts + 1):
        if not queued:
            break
        retry_max_tokens = min(
            token_cap,
            request.requirements.max_tokens * token_multiplier**retry_round,
        )
        retry_request = _retry_request(request, retry_max_tokens)
        events.emit(
            "deferred_retry_started",
            variant=variant_id,
            retry_round=retry_round,
            rows=len(queued),
            batch_size=retry_batch,
            max_tokens=retry_max_tokens,
        )

        def on_backoff(exc: BackendFailure, old_size: int, new_size: int) -> None:
            nonlocal retry_batch
            retry_batch = new_size
            events.emit(
                "deferred_retry_backoff",
                variant=variant_id,
                retry_round=retry_round,
                new_batch_size=new_size,
                error=str(exc),
            )
            errors_log.write(
                run_id=run_id,
                dataset_id=config["run"].get("dataset_id", "default"),
                variant_id=variant_id,
                stage="deferred_retry_backoff",
                error_kind="backend_batch_failure",
                error=str(exc),
                retry_round=retry_round,
                failed_batch_size=old_size,
                new_batch_size=new_size,
            )

        next_queue: list[dict[str, Any]] = []
        offset = 0
        while offset < len(queued):
            retry_chunk = queued[offset : offset + retry_batch]
            retry_prompts = [
                render(
                    correction,
                    retry_values(
                        config,
                        variant,
                        item["row"],
                        schema,
                        item["response"].text,
                        item["errors"],
                    ),
                )
                for item in retry_chunk
            ]
            started = time.perf_counter()
            responses, used = generate_with_backoff(
                backend,
                retry_request,
                retry_prompts,
                batch_config,
                on_backoff,
            )
            retry_chunk = retry_chunk[:used]
            elapsed = max(time.perf_counter() - started, 1e-9)
            snapshot = serialise(gpu(max_age_seconds=5.0)) or "{}"

            for item, response in zip(retry_chunk, responses):
                row = item["row"]
                processed = processor.process(response, row)
                errors = _error_strings(processed)
                attempt_count = int(item["attempt_count"]) + 1
                if errors:
                    errors_log.write(
                        run_id=run_id,
                        dataset_id=config["run"].get("dataset_id", "default"),
                        variant_id=variant_id,
                        input_row_id=str(row[config["input"]["id_column"]]),
                        source_position=int(row["_source_position"]),
                        stage="deferred_validation",
                        error_kind=error_kind(errors, response, retry_max_tokens),
                        validation_errors=errors,
                        attempt_count=attempt_count,
                        batch_size=len(retry_chunk),
                        max_tokens=retry_max_tokens,
                        token_count=response.token_count,
                        raw_response=response.text,
                    )
                    if retry_round < max_attempts:
                        next_queue.append(
                            {
                                **item,
                                "response": response,
                                "errors": errors,
                                "attempt_count": attempt_count,
                            }
                        )
                        continue
                append_output(
                    _write_processed_result(
                        config,
                        processed,
                        run_id=run_id,
                        variant_id=variant_id,
                        config_hash=config_hash,
                        group_id=group_id,
                        row=row,
                        prompt_text=item["prompt"],
                        response=response,
                        attempt_count=attempt_count,
                        batch_size=len(retry_chunk),
                        latency_seconds=elapsed / len(retry_chunk),
                        rows_per_second=len(retry_chunk) / elapsed,
                        gpu_snapshot=snapshot,
                    )
                )
                written += 1
            index.connection.commit()
            offset += len(retry_chunk)
        queued = next_queue
        events.emit(
            "deferred_retry_completed",
            variant=variant_id,
            retry_round=retry_round,
            remaining=len(queued),
        )
    return written


def run_matrix(
    config: dict[str, Any],
    row_limit: int | None = None,
    selected: list[str] | None = None,
) -> dict[str, Any]:
    """Run configured datasets while sharing a single backend instance."""

    if "datasets" not in config:
        return {"datasets": [run(config, row_limit=row_limit)]}
    if config["model"].get("backend") == "local_vllm" and "_resource_guard" not in config:
        apply_resource_guard(config)
    base_output = resolve(config, config["output"]["directory"])
    entries = selected_entries(config, selected)
    shared = create_backend(config["model"], config.get("_resource_guard"))
    manifests: list[dict[str, Any]] = []
    try:
        for dataset_id, source in entries:
            lane = dataset_config(config, dataset_id, source, base_output)
            lane["logging"] = {
                **config.get("logging", {}),
                "file": str(base_output / "logs" / f"{dataset_id}.log"),
                "events": str(base_output / "logs" / f"{dataset_id}.events.jsonl"),
            }
            manifests.append(run(lane, shared, row_limit=row_limit))
    finally:
        shared.close()
    summary = {
        "run_id": config["run"]["id"],
        "model": config["model"],
        "cpu_resource_guard": config.get("_resource_guard"),
        "datasets": manifests,
        "result_rows": sum(int(item.get("result_rows", 0)) for item in manifests),
    }
    base_output.mkdir(parents=True, exist_ok=True)
    (base_output / "matrix_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


__all__ = [
    "generate_with_backoff",
    "generation_request",
    "run",
    "run_deferred_retries",
    "run_matrix",
    "tune",
]
