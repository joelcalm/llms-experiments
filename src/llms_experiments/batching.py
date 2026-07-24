"""External ``vllm run-batch`` preparation and response ingestion."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from .backends.base import extract_token_positions
from .configuration import dataset_config, resolve, selected_entries
from .events import make_events
from .inputs import iter_rows_for_source, rows_for_source, source_provenance
from .outputs import create_output_store
from .processors import Processor, RawModelResponse
from .prompting import (
    conversation,
    materialize_variant,
    prompt_group_id,
    render,
    rendered_prompt,
    request_for_row,
    retry_values,
    system_prompt,
    variant_config_hash,
    variant_schema,
)
from .results import CONTRACT_VERSION, TOOL_VERSION, result_row, serialise


def prepare(config: dict[str, Any]) -> Path:
    """Write one OpenAI-compatible JSONL request for every prepared row."""

    run_dir = resolve(config, config["output"]["directory"])
    path = run_dir / "requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for configured in config["variants"]:
            variant = materialize_variant(config, configured)
            processor: Processor = variant["_processor"]
            source = iter_rows_for_source(config, config["input"])
            for row in processor.prepare_rows(source):
                request = request_for_row(config, variant, dict(row))
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")
    return path


def prepare_matrix(config: dict[str, Any], selected: list[str] | None = None) -> list[Path]:
    """Prepare an independent request file for each selected dataset."""

    if "datasets" not in config:
        return [prepare(config)]
    base_output = resolve(config, config["output"]["directory"])
    return [
        prepare(dataset_config(config, dataset_id, source, base_output))
        for dataset_id, source in selected_entries(config, selected)
    ]


def _batch_response(item: dict[str, Any]) -> RawModelResponse:
    response = item.get("response") or {}
    if item.get("error") or response.get("status_code", 200) != 200:
        error = str(item.get("error") or response.get("status_code", "batch_response_error"))
        return RawModelResponse("", 0, backend_error=error)
    body = response.get("body") or {}
    choice = (body.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    if content is None:
        return RawModelResponse("", 0, backend_error="missing_chat_completion_content")
    usage = body.get("usage") or {}
    positions = extract_token_positions((choice.get("logprobs") or {}).get("content"))
    return RawModelResponse(
        str(content),
        int(usage.get("completion_tokens", 0) or 0),
        positions,
    )


def _response_key(custom_id: str) -> tuple[tuple[str, str, int], int] | None:
    if not custom_id.startswith("retry:"):
        return None
    body, attempt = custom_id.rsplit(":", 1)
    variant_id, remainder = body[len("retry:") :].split(":", 1)
    row_id, position = remainder.rsplit(":", 1)
    return (variant_id, row_id, int(position)), int(attempt)


def _completion_body(
    config: dict[str, Any],
    variant: dict[str, Any],
    processor: Processor,
    schema: dict[str, Any] | None,
    user_prompt: str,
) -> dict[str, Any]:
    requirements = processor.requirements
    body: dict[str, Any] = {
        "model": config["model"]["name"],
        "messages": conversation(system_prompt(config, variant, schema), user_prompt),
        "temperature": 0,
        "max_completion_tokens": requirements.max_tokens,
    }
    if requirements.capture_logprobs:
        body.update({"logprobs": True, "top_logprobs": requirements.top_logprobs})
    if requirements.structured_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": variant["id"],
                "schema": dict(requirements.structured_schema),
                "strict": True,
            },
        }
    return body


def parse_batch(config: dict[str, Any], response_path: str | Path) -> dict[str, Any]:
    """Merge external batch responses into the common result contract."""

    source = Path(response_path)
    if source.is_dir():
        source = source / "responses.jsonl"
    responses: dict[str, dict[str, Any]] = {}
    retries: dict[tuple[str, str, int], list[tuple[int, dict[str, Any]]]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = str(item["custom_id"])
        if custom_id in responses:
            raise ValueError(f"Duplicate custom_id in batch response: {custom_id}")
        responses[custom_id] = item
        retry_key = _response_key(custom_id)
        if retry_key is not None:
            key, attempt = retry_key
            retries.setdefault(key, []).append((attempt, item))

    run_id = str(config["run"]["id"])
    run_dir = resolve(config, config["output"]["directory"])
    output_store = create_output_store(config["output"], run_dir)
    events = make_events(config, run_dir, run_id)
    rows = rows_for_source(config, config["input"])
    expected_hashes = {
        str(item["id"]): variant_config_hash(config, materialize_variant(config, item)) for item in config["variants"]
    }
    saved = [
        row
        for row in output_store.read_final()
        if row.get("config_hash") == expected_hashes.get(str(row["variant_id"]))
    ]
    saved_by_key = {
        (str(row["variant_id"]), str(row["input_row_id"]), int(row.get("source_position", -1))): row for row in saved
    }
    complete = set(saved_by_key)
    retry_settings = config.get("validation", {}).get("retry", {})
    correction_path = retry_settings.get("correction_prompt")
    correction = Path(resolve(config, correction_path)).read_text(encoding="utf-8") if correction_path else None
    retry_pending = bool(
        correction and retry_settings.get("enabled") and int(retry_settings.get("max_attempts", 0)) >= 1
    )
    retry_requests: list[dict[str, Any]] = []

    def add_retry_request(
        variant: dict[str, Any],
        processor: Processor,
        schema: dict[str, Any] | None,
        row: dict[str, Any],
        response: RawModelResponse,
        errors: list[str],
        attempt: int,
    ) -> None:
        if (
            not errors
            or not correction
            or not retry_settings.get("enabled")
            or processor.requirements.one_token
            or attempt > int(retry_settings.get("max_attempts", 0))
        ):
            return
        retry_prompt = render(
            correction,
            retry_values(config, variant, row, schema, response.text, errors),
        )
        retry_requests.append(
            {
                "custom_id": (
                    f"retry:{variant['id']}:{row[config['input']['id_column']]}:{row['_source_position']}:{attempt}"
                ),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": _completion_body(config, variant, processor, schema, retry_prompt),
            }
        )

    for configured in config["variants"]:
        variant = materialize_variant(config, configured)
        processor: Processor = variant["_processor"]
        variant_id = str(variant["id"])
        config_hash = expected_hashes[variant_id]
        schema = variant_schema(config, variant)
        group_id = prompt_group_id(config, variant, schema)
        for prepared in processor.prepare_rows(rows):
            row = dict(prepared)
            row_id = str(row[config["input"]["id_column"]])
            key = (variant_id, row_id, int(row["_source_position"]))
            retry_items = retries.get(key, [])
            if key in complete and not retry_items:
                continue
            current_prompt = rendered_prompt(config, variant, row, schema)
            custom_id = f"{variant_id}:{row_id}:{row['_source_position']}"
            response = (
                _batch_response(responses[custom_id])
                if custom_id in responses
                else RawModelResponse("", 0, backend_error="missing_batch_response")
            )
            processed = processor.process(response, row)
            errors = [error.as_contract_string() for error in processed.errors]
            attempt_count = 1

            if retry_items:
                attempt_count, latest = max(retry_items, key=lambda pair: pair[0])
                response = _batch_response(latest)
                processed = processor.process(response, row)
                errors = [error.as_contract_string() for error in processed.errors]
                target = saved_by_key.get(key)
                if target is not None:
                    target.update(
                        {
                            "attempt_count": attempt_count,
                            "raw_response": response.text,
                            "parsed_output": serialise(processed.value),
                            "validation_status": "valid" if not errors else "invalid",
                            "validation_errors": errors,
                            "final_status": "completed" if not errors else "failed_validation",
                            "candidate_scores": (
                                dict(processed.candidate_scores) if processed.candidate_scores is not None else None
                            ),
                        }
                    )
                    add_retry_request(
                        variant,
                        processor,
                        schema,
                        row,
                        response,
                        errors,
                        attempt_count + 1,
                    )
                    continue

            add_retry_request(
                variant,
                processor,
                schema,
                row,
                response,
                errors,
                attempt_count + 1,
            )
            output = result_row(
                config,
                run_id=run_id,
                variant_id=variant_id,
                config_hash=config_hash,
                group_id=group_id,
                row=row,
                prompt_text=current_prompt,
                raw=response.text if not response.backend_error else None,
                parsed=processed.value,
                errors=errors,
                attempt_count=attempt_count,
                token_count=response.token_count,
                candidate_logprobs=(
                    dict(processed.candidate_scores) if processed.candidate_scores is not None else None
                ),
                final_status=(
                    None
                    if not errors
                    else "retry_pending"
                    if retry_pending and not processor.requirements.one_token
                    else None
                ),
            )
            saved.append(output)
            saved_by_key[key] = output

    output_store.write_snapshot(saved)
    retry_path: Path | None = None
    if retry_requests:
        retry_path = run_dir / "retry_requests.jsonl"
        retry_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in retry_requests),
            encoding="utf-8",
        )
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "tool_version": TOOL_VERSION,
        "run_id": run_id,
        "model_id": str(config["model"]["name"]),
        "dataset_id": config["run"].get("dataset_id", "default"),
        "input_rows": len(rows),
        "result_rows": len(saved),
        "model": config["model"],
        "effective_config": {key: value for key, value in config.items() if not key.startswith("_")},
        "variants": {variant["id"]: {"external_batch": True} for variant in config["variants"]},
        "batch_response_path": str(source),
        "source_provenance": source_provenance(config),
        "resume_skipped_rows": len(complete),
        "event_log": str(events.path),
        "retry_request_path": str(retry_path) if retry_path else None,
        "retry_requests": len(retry_requests),
        "output": {
            "format": output_store.format_name,
            "result_path": str(output_store.result_path()),
            "variant_paths": {
                str(variant["id"]): str(output_store.variant_path(str(variant["id"]))) for variant in config["variants"]
            },
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    events.emit("batch_parse_completed", result_rows=len(saved), response_path=str(source))
    events.close()
    return manifest


def batch_command_args(
    config: dict[str, Any],
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> list[str]:
    run_dir = resolve(config, config["output"]["directory"])
    model = config["model"]
    command = [
        "vllm",
        "run-batch",
        "-i",
        str(input_path or run_dir / "requests.jsonl"),
        "-o",
        str(output_path or run_dir / "responses.jsonl"),
        "--model",
        str(model["name"]),
        "--gpu-memory-utilization",
        str(model.get("gpu_memory_utilization", 0.9)),
        "--max-model-len",
        str(model.get("max_model_len", 2048)),
        "--max-num-seqs",
        str(model.get("max_num_seqs", 128)),
    ]
    if model.get("enable_prefix_caching", True):
        command.append("--enable-prefix-caching")
    if model.get("language_model_only", False):
        command.append("--language-model-only")
    for option, flag in (
        ("tokenizer_mode", "--tokenizer-mode"),
        ("config_format", "--config-format"),
        ("load_format", "--load-format"),
    ):
        if option in model:
            command.extend([flag, str(model[option])])
    return command


def batch_command(config: dict[str, Any]) -> str:
    return shlex.join(batch_command_args(config))


__all__ = [
    "batch_command",
    "batch_command_args",
    "parse_batch",
    "prepare",
    "prepare_matrix",
]
