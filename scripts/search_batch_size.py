#!/usr/bin/env python3
"""Find a safe and high-throughput batch size by exponential/binary search.

The script runs one configured dataset lane and keeps the backend warm while it
tests each JSON variant. It doubles the batch size after successful trials. On
an out-of-memory failure it binary-searches the interval between the last safe
size and the failed size. Results are written atomically after every trial so
an interrupted search leaves a complete trial record.

Run it through ``uv run``. For example::

    uv run python scripts/search_batch_size.py \
        configs/protoethos/protoethos_full_matrix.yaml \
        --dataset mftc --rows 1024 --start-batch 8 --max-batch 1024 \
        --output results/batch_search/qwen36_27b_mftc.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llms_experiments.backends import BackendFailure, create_backend
from llms_experiments.configuration import load_config, select_dataset
from llms_experiments.inputs import rows_for_source
from llms_experiments.orchestration import generation_request
from llms_experiments.prompting import materialize_variant, rendered_prompt, variant_schema
from llms_experiments.runtime import apply_resource_guard, gpu, sync_cuda

OOM_MARKERS = (
    "out of memory",
    "cuda oom",
    "cublas_status_alloc_failed",
    "memory allocation failed",
)


def _redact(value: Any) -> Any:
    """Return JSON-safe metadata without exposing secret-like settings."""

    if isinstance(value, dict):
        return {
            str(key): "<redacted>"
            if any(marker in str(key).lower() for marker in ("token", "secret", "password", "api_key"))
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _is_oom(message: str) -> bool:
    """Identify memory exhaustion while avoiding misclassifying other failures."""

    lowered = message.lower()
    return any(marker in lowered for marker in OOM_MARKERS) or "oom" in lowered


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Persist a search snapshot without exposing a partially written JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_variants(
    config: dict[str, Any], names: list[str], rows: list[dict[str, Any]]
) -> list[tuple[str, Any, Any, list[str]]]:
    """Materialise selected variants and their prompts for the configured rows."""

    prepared: list[tuple[str, Any, Any, list[str]]] = []
    available = {str(item["id"]): item for item in config["variants"]}
    missing = set(names) - set(available)
    if missing:
        raise ValueError(f"Unknown variant id(s): {', '.join(sorted(missing))}")
    for name in names:
        variant = materialize_variant(config, available[name])
        schema = variant_schema(config, variant)
        processor = variant["_processor"]
        prepared.append(
            (
                name,
                generation_request(config, variant, processor, schema),
                schema,
                [rendered_prompt(config, variant, row, schema) for row in processor.prepare_rows(rows)],
            )
        )
    return prepared


def _trial(
    backend: Any,
    request: Any,
    prompts: list[str],
    size: int,
    repeats: int,
    synchronize_cuda: bool,
) -> dict[str, Any]:
    """Measure one batch size and return a serialisable trial record."""

    started = time.perf_counter()
    token_count = 0
    samples = 0
    try:
        for _ in range(repeats):
            sync_cuda(synchronize_cuda)
            responses = backend.generate(prompts[:size], request)
            if len(responses) != size:
                raise BackendFailure(f"Backend returned {len(responses)} responses for {size} prompts")
            samples += len(responses)
            token_count += sum(int(response.token_count) for response in responses)
            sync_cuda(synchronize_cuda)
        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            "batch_size": size,
            "status": "success",
            "oom": False,
            "repeats": repeats,
            "samples_processed": samples,
            "samples_per_second": samples / elapsed,
            "tokens_processed": token_count,
            "tokens_per_second": token_count / elapsed,
            "wall_seconds": elapsed,
            "gpu": gpu(),
        }
    except BackendFailure as exc:
        message = str(exc)
        if not _is_oom(message):
            raise
        gc.collect()
        return {
            "batch_size": size,
            "status": "oom",
            "oom": True,
            "repeats": repeats,
            "samples_processed": samples,
            "samples_per_second": 0.0,
            "tokens_processed": token_count,
            "tokens_per_second": 0.0,
            "wall_seconds": max(time.perf_counter() - started, 1e-9),
            "error": message,
            "gpu": gpu(),
        }


def _search_variant(
    backend: Any,
    request: Any,
    prompts: list[str],
    *,
    start: int,
    maximum: int,
    repeats: int,
    synchronize_cuda: bool,
    on_trial: Any,
) -> dict[str, Any]:
    """Double then binary-search the largest batch that does not OOM."""

    trials: list[dict[str, Any]] = []
    by_size: dict[int, dict[str, Any]] = {}

    def run(size: int) -> dict[str, Any]:
        if size in by_size:
            return by_size[size]
        result = _trial(backend, request, prompts, size, repeats, synchronize_cuda)
        by_size[size] = result
        trials.append(result)
        on_trial(result)
        return result

    safe = 0
    failed: int | None = None
    candidate = start
    while candidate <= maximum:
        result = run(candidate)
        if result["status"] == "success":
            safe = candidate
            candidate *= 2
            continue
        failed = candidate
        break

    while failed is not None and failed - safe > 1:
        midpoint = safe + (failed - safe) // 2
        result = run(midpoint)
        if result["status"] == "success":
            safe = midpoint
        else:
            failed = midpoint

    successful = [item for item in trials if item["status"] == "success"]
    best_throughput = max(
        successful,
        key=lambda item: (float(item["samples_per_second"]), int(item["batch_size"])),
        default=None,
    )
    return {
        "trials": trials,
        "highest_safe_batch_size": safe,
        "first_failed_batch_size": failed,
        "boundary_found": failed is not None,
        "throughput_best_batch_size": best_throughput["batch_size"] if best_throughput else None,
        "throughput_best_samples_per_second": best_throughput["samples_per_second"] if best_throughput else None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML experiment configuration")
    parser.add_argument("--dataset", required=True, help="One dataset id from the matrix configuration")
    parser.add_argument(
        "--variants",
        default="single_label_json,multi_label_json",
        help="Comma-separated variant ids (default: both JSON-only strategies)",
    )
    parser.add_argument("--rows", type=int, default=1024, help="Rows available for each trial")
    parser.add_argument("--start-batch", type=int, default=8, help="First batch size to test")
    parser.add_argument("--max-batch", type=int, default=4096, help="Maximum batch size to test")
    parser.add_argument("--repeats", type=int, default=1, help="Measurements per candidate batch")
    parser.add_argument("--output", type=Path, required=True, help="Atomic JSON trial report path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Configuration override KEY=VALUE")
    return parser


def main() -> int:
    """Load the experiment, run the search for both variants, and write its report."""

    args = _parser().parse_args()
    if args.rows < 1 or args.start_batch < 1 or args.max_batch < args.start_batch or args.repeats < 1:
        raise SystemExit("rows, batches, and repeats must be positive; max-batch must be >= start-batch")

    base_config = load_config(args.config, args.overrides)
    config = select_dataset(base_config, args.dataset)
    rows = rows_for_source(config, config["input"], args.rows)
    maximum = min(args.max_batch, len(rows))
    if maximum < args.start_batch:
        raise SystemExit(f"Only {len(rows)} rows are available, less than start-batch={args.start_batch}")
    variant_names = [item.strip() for item in args.variants.split(",") if item.strip()]
    if not variant_names:
        raise SystemExit("At least one variant is required")

    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": hashlib.sha256(
            json.dumps(_redact(base_config), sort_keys=True, default=str).encode()
        ).hexdigest(),
        "dataset": args.dataset,
        "variants": variant_names,
        "rows_available": len(rows),
        "search": {
            "start_batch": args.start_batch,
            "max_batch": maximum,
            "repeats": args.repeats,
            "algorithm": "doubling_then_binary_search_on_oom",
        },
        "model": _redact(config["model"]),
        "results": {},
    }
    _write_atomic(args.output, report)

    prepared = _prepare_variants(config, variant_names, rows)
    if config["model"].get("backend") == "local_vllm" and "_resource_guard" not in config:
        apply_resource_guard(config)
    backend = create_backend(config["model"], config.get("_resource_guard"))
    try:
        for name, request, _schema, prompts in prepared:
            report["results"][name] = {}

            def save_trial(result: dict[str, Any], variant_name: str = name) -> None:
                report["results"][variant_name]["latest_trial"] = result
                report["results"][variant_name]["trial_count"] = (
                    len(report["results"][variant_name].get("trials", [])) + 1
                )
                report["results"][variant_name].setdefault("trials", []).append(result)
                _write_atomic(args.output, report)

            result = _search_variant(
                backend,
                request,
                prompts,
                start=args.start_batch,
                maximum=maximum,
                repeats=args.repeats,
                synchronize_cuda=bool(config["model"].get("synchronize_cuda", False)),
                on_trial=save_trial,
            )
            report["results"][name] = result
            _write_atomic(args.output, report)
    finally:
        backend.close()

    report["finished_at"] = datetime.now(UTC).isoformat()
    _write_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
