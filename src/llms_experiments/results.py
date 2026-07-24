"""Logical result contract, response validation compatibility, and row construction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .outputs import RESULT_SCHEMA
from .processors import ProcessedResult, RawModelResponse

CONTRACT_VERSION = "2.0"
TOOL_VERSION = "0.2.0"
BACKEND_ERROR_PREFIX = "backend_error:"


def serialise(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def check_schema(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    expected = schema.get("type")
    valid_type = (
        expected is None
        or (expected == "object" and isinstance(value, dict))
        or (expected == "array" and isinstance(value, list))
        or (expected == "string" and isinstance(value, str))
        or (expected == "number" and isinstance(value, int | float) and not isinstance(value, bool))
        or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (expected == "boolean" and isinstance(value, bool))
        or (expected == "null" and value is None)
    )
    if not valid_type:
        errors.append(f"{path}: expected {expected}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not in enum")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: missing required property")
        properties = schema.get("properties", {})
        for key, child in value.items():
            if key in properties:
                check_schema(child, properties[key], f"{path}.{key}", errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: additional property is not allowed")
    if isinstance(value, list) and "items" in schema:
        for index, child in enumerate(value):
            check_schema(child, schema["items"], f"{path}[{index}]", errors)
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: value is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: value is above maximum")


def validate_response(raw: str, schema: dict[str, Any] | None) -> tuple[Any | None, list[str]]:
    if schema is None:
        return raw, []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"json_parse_error: {exc}"]
    errors: list[str] = []
    check_schema(value, schema, "$", errors)
    return value, errors


def failure_status(errors: list[str]) -> str:
    return "failed_backend" if any(item.startswith(BACKEND_ERROR_PREFIX) for item in errors) else "failed_validation"


def error_kind(errors: list[str], response: RawModelResponse | Any, max_tokens: int) -> str:
    if response.token_count >= int(max_tokens or 0):
        return "output_length_limit"
    if any(error.startswith("json_parse_error:") for error in errors):
        return "format"
    return "schema"


def semantic_result_type(variant: dict[str, Any]) -> str:
    processor = variant.get("processor")
    if isinstance(processor, dict) and processor.get("result"):
        return str(processor["result"])
    if variant.get("_result_type"):
        return str(variant["_result_type"])
    return "raw"


def result_row(
    config: dict[str, Any],
    *,
    run_id: str,
    variant_id: str,
    config_hash: str,
    group_id: str | None,
    row: dict[str, Any],
    prompt_text: str,
    raw: str | None,
    parsed: Any | None,
    errors: list[str],
    attempt_count: int,
    token_count: int | None = None,
    batch_size: int | None = None,
    latency_seconds: float | None = None,
    rows_per_second: float | None = None,
    gpu_snapshot: str | None = None,
    candidate_logprobs: dict[str, float] | None = None,
    final_status: str | None = None,
) -> dict[str, Any]:
    source = config["input"]
    configured = next(item for item in config["variants"] if item["id"] == variant_id)
    return {
        "contract_version": CONTRACT_VERSION,
        "tool_version": TOOL_VERSION,
        "run_id": run_id,
        "model_id": str(config["model"]["name"]),
        "dataset_id": config["run"].get("dataset_id", "default"),
        "variant_id": variant_id,
        "result_type": semantic_result_type(configured),
        "input_row_id": str(row[source["id_column"]]),
        "source_position": row["_source_position"],
        "input_text": str(row[source["text_column"]]) if config["output"].get("include_text") else None,
        "gold_labels": list(row.get("_gold_labels") or []),
        "prompt_hash": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "config_hash": config_hash,
        "prompt_group_id": group_id,
        "attempt_count": attempt_count,
        "raw_response": raw if config["output"].get("include_raw_response", True) else None,
        "parsed_output": serialise(parsed),
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": list(errors),
        "final_status": final_status or ("completed" if not errors else failure_status(errors)),
        "batch_size": batch_size,
        "latency_seconds": latency_seconds,
        "rows_per_second": rows_per_second,
        "token_count": token_count,
        "gpu_snapshot": gpu_snapshot,
        "candidate_scores": candidate_logprobs,
        "target_label": row.get("_target_label"),
    }


def result_from_processed(
    config: dict[str, Any],
    processed: ProcessedResult,
    **metadata: Any,
) -> dict[str, Any]:
    return result_row(
        config,
        parsed=processed.value,
        errors=[error.as_contract_string() for error in processed.errors],
        candidate_logprobs=dict(processed.candidate_scores) if processed.candidate_scores is not None else None,
        **metadata,
    )


__all__ = [
    "BACKEND_ERROR_PREFIX",
    "CONTRACT_VERSION",
    "RESULT_SCHEMA",
    "TOOL_VERSION",
    "check_schema",
    "error_kind",
    "failure_status",
    "result_from_processed",
    "result_row",
    "semantic_result_type",
    "serialise",
    "validate_response",
]
