"""Tests for compact parsed-result Parquet exports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_parsed_results import OUTPUT_SCHEMA, export_source


def _write_lane(root: Path, lane: str, rows: list[dict[str, object]]) -> None:
    lane_dir = root / f"dataset={lane}"
    lane_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), lane_dir / "single_label_json.parquet")
    pq.write_table(pa.Table.from_pylist(rows), lane_dir / "multi_label_json.parquet")
    (lane_dir / "manifest.json").write_text("{}\n", encoding="utf-8")


def test_export_projects_valid_json_to_zero_one_columns(tmp_path: Path) -> None:
    root = tmp_path / "source"
    rows = [
        {
            "input_row_id": "r0",
            "source_position": 0,
            "parsed_output": json.dumps({"label": "authority"}),
            "validation_status": "valid",
            "final_status": "completed",
        },
        {
            "input_row_id": "r1",
            "source_position": 1,
            "parsed_output": json.dumps({"label": "care"}),
            "validation_status": "invalid",
            "final_status": "completed",
        },
    ]
    _write_lane(root, "mftc", rows)
    output = tmp_path / "single_label.parquet"
    summary = export_source(root, output, "example/model", "single_label_json", require_eleven_lanes=False)

    table = pq.read_table(output)
    assert table.schema == OUTPUT_SCHEMA
    assert table.num_rows == 1
    record = table.to_pylist()[0]
    assert record["mft__authority"] == 1
    assert record["mft__care"] == 0
    assert record["shvt__achievement"] == 0
    assert summary.included_rows == 1
    assert summary.exclusion_reasons == {"validation_status_not_valid": 1}


def test_multi_label_empty_set_is_all_zero(tmp_path: Path) -> None:
    root = tmp_path / "source"
    lane = root / "dataset=valueeval"
    lane.mkdir(parents=True)
    rows = [
        {
            "input_row_id": "r0",
            "source_position": 0,
            "parsed_output": json.dumps({"labels": []}),
            "validation_status": "valid",
            "final_status": "completed",
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows), lane / "multi_label_json.parquet")
    pq.write_table(pa.Table.from_pylist(rows), lane / "single_label_json.parquet")
    (lane / "manifest.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "multi_label.parquet"

    export_source(root, output, "example/model", "multi_label_json", require_eleven_lanes=False)
    record = pq.read_table(output).to_pylist()[0]
    assert sum(record[name] for name in record if name.startswith("shvt__")) == 0
