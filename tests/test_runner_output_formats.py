"""The complete runner has identical semantics for every output store."""

from __future__ import annotations

from pathlib import Path

import pytest

from llms_experiments.orchestration import run
from llms_experiments.outputs import create_output_store


@pytest.mark.parametrize("output_format", ["parquet", "csv", "jsonl"])
def test_runner_roundtrip_and_resume_across_output_formats(tmp_path: Path, output_format: str) -> None:
    source = tmp_path / "input.csv"
    source.write_text("id,text\n1,first\n2,second\n", encoding="utf-8")
    (tmp_path / "prompt.md").write_text("Classify {{text}}", encoding="utf-8")
    run_dir = tmp_path / output_format
    config = {
        "_root": str(tmp_path),
        "_override_keys": [],
        "run": {"id": f"format-{output_format}", "dataset_id": "test"},
        "input": {
            "path": str(source),
            "format": "csv",
            "id_column": "id",
            "text_column": "text",
        },
        "model": {"name": "fake", "backend": "fake"},
        "variants": [
            {
                "id": "raw",
                "prompts": ["prompt.md"],
                "processor": {"result": "raw", "stages": [{"type": "identity"}]},
            },
            {
                "id": "scores",
                "prompts": ["prompt.md"],
                "processor": {
                    "result": "categorical_logprobs",
                    "stages": [{"type": "candidate_logprobs", "candidates": ["A", "B"]}],
                },
            },
        ],
        "output": {
            "directory": str(run_dir),
            "format": output_format,
            "include_raw_response": True,
            "include_text": True,
        },
        "batch": {"mode": "fixed", "size": 2, "candidates": [2], "min_size": 1},
        "logging": {
            "file": str(tmp_path / f"{output_format}.log"),
            "events": str(tmp_path / f"{output_format}.events.jsonl"),
        },
    }

    first = run(config)
    second = run(config)
    rows = create_output_store(config["output"], run_dir).read_final()

    assert first["result_rows"] == 4
    assert second["result_rows"] == 4
    assert second["resume_skipped_rows"] == 4
    assert {(row["variant_id"], row["input_row_id"]) for row in rows} == {
        ("raw", "1"),
        ("raw", "2"),
        ("scores", "1"),
        ("scores", "2"),
    }
    score_rows = [row for row in rows if row["variant_id"] == "scores"]
    assert all(row["candidate_scores"] == {"A": 0.0, "B": -1.0} for row in score_rows)
