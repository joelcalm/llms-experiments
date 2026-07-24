"""Equivalent durability and logical-row behavior across all output stores."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from llms_experiments.outputs import RESULT_COLUMNS, ResumeIndex, create_output_store


def _row(position: int, *, status: str = "completed", raw: str | None = "") -> dict:
    row = dict.fromkeys(RESULT_COLUMNS)
    row.update(
        {
            "contract_version": "2.0",
            "tool_version": "0.2.0",
            "run_id": "run",
            "model_id": "fake",
            "dataset_id": "data",
            "variant_id": "variant",
            "result_type": "categorical_logprobs",
            "input_row_id": f"row-{position}",
            "source_position": position,
            "gold_labels": ["care"],
            "config_hash": "expected",
            "attempt_count": 1,
            "raw_response": raw,
            "parsed_output": '{"candidates":{"A":0.0}}',
            "validation_status": "valid",
            "validation_errors": [],
            "final_status": status,
            "candidate_scores": {"A": 0.0, "B": -float("inf")},
        }
    )
    return row


@pytest.mark.parametrize("output_format", ["parquet", "csv", "jsonl"])
def test_store_roundtrip_atomic_resume_and_finalize(tmp_path: Path, output_format: str) -> None:
    store = create_output_store({"format": output_format, "directory": str(tmp_path)}, tmp_path)
    writer = store.part_writer("variant", target_rows=2)
    writer.append(_row(0, raw=""))

    before = ResumeIndex(tmp_path / "before.sqlite", "v2", store)
    assert before.seed_from(store.part_paths()) == 0
    before.close()

    assert writer.append(_row(1, raw=None))
    after = ResumeIndex(tmp_path / "after.sqlite", "v2", store)
    assert after.seed_from(store.part_paths(), {"variant": "expected"}) == 2
    after.close()

    assert store.finalize({"variant": "expected"}) == 2
    rows = store.read_final()
    assert rows[0]["raw_response"] == ""
    assert rows[1]["raw_response"] is None
    assert rows[0]["gold_labels"] == ["care"]
    assert rows[0]["candidate_scores"]["A"] == 0.0
    assert math.isinf(rows[0]["candidate_scores"]["B"])
    assert store.variant_path("variant").exists()


@pytest.mark.parametrize("output_format", ["parquet", "csv", "jsonl"])
def test_store_keeps_latest_retried_attempt(tmp_path: Path, output_format: str) -> None:
    store = create_output_store({"format": output_format, "directory": str(tmp_path)}, tmp_path)
    first = store.part_writer("variant", target_rows=1)
    first.append(_row(0, status="failed_backend", raw="first"))
    second = store.part_writer("variant", target_rows=1)
    replacement = _row(0, raw="second")
    replacement["attempt_count"] = 2
    second.append(replacement)
    key = ("variant", "row-0", 0)

    assert store.finalize({"variant": "expected"}, {key}) == 1
    rows = store.read_final()
    assert len(rows) == 1
    assert rows[0]["raw_response"] == "second"
    assert rows[0]["attempt_count"] == 2
