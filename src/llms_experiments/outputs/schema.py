"""Format-independent result fields and the Parquet contract schema."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

RESULT_SCHEMA = pa.schema(
    [
        ("contract_version", pa.string()),
        ("tool_version", pa.string()),
        ("run_id", pa.string()),
        ("model_id", pa.string()),
        ("dataset_id", pa.string()),
        ("variant_id", pa.string()),
        ("result_type", pa.string()),
        ("input_row_id", pa.string()),
        ("source_position", pa.int64()),
        ("input_text", pa.string()),
        ("gold_labels", pa.list_(pa.string())),
        ("prompt_hash", pa.string()),
        ("config_hash", pa.string()),
        ("prompt_group_id", pa.string()),
        ("attempt_count", pa.int64()),
        ("raw_response", pa.string()),
        ("parsed_output", pa.string()),
        ("validation_status", pa.string()),
        ("validation_errors", pa.list_(pa.string())),
        ("final_status", pa.string()),
        ("batch_size", pa.int64()),
        ("latency_seconds", pa.float64()),
        ("rows_per_second", pa.float64()),
        ("token_count", pa.int64()),
        ("gpu_snapshot", pa.string()),
        ("candidate_scores", pa.map_(pa.string(), pa.float64())),
        ("target_label", pa.string()),
    ]
)
RESULT_COLUMNS = tuple(RESULT_SCHEMA.names)


def normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a complete logical row with stable map/list representations."""

    normalized = {name: row.get(name) for name in RESULT_COLUMNS}
    for name in ("gold_labels", "validation_errors"):
        value = normalized[name]
        normalized[name] = list(value or [])
    scores = normalized["candidate_scores"]
    if isinstance(scores, list):
        normalized["candidate_scores"] = {str(key): float(value) for key, value in scores}
    elif scores is not None:
        normalized["candidate_scores"] = {str(key): float(value) for key, value in dict(scores).items()}
    return normalized
