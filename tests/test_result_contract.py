"""Version 2 Parquet, manifest, and durability contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from conftest import REPO_ROOT

from llms_experiments import _core as core


def _run(tmp_path: Path) -> tuple[Path, dict]:
    config = core.load_config(
        REPO_ROOT / "experiments" / "matrix_smoke.yaml",
        [f"output.directory={tmp_path / 'run'}", "model.backend=fake", "streaming.output_chunk_rows=2"],
        check_files=True,
    )
    lane = core.select_dataset(config, "nested_json")
    lane["variants"] = config["variants"]
    manifest = core.run(lane, core.FakeBackend(), row_limit=2)
    return Path(lane["output"]["directory"]), manifest


def test_v2_rows_use_native_arrow_values_and_required_semantics(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path)
    table = pq.read_table(run_dir / "results.parquet")
    rows = table.to_pylist()

    assert table.schema.field("gold_labels").type == pa.list_(pa.string())
    assert table.schema.field("validation_errors").type == pa.list_(pa.string())
    assert table.schema.field("candidate_scores").type == pa.map_(pa.string(), pa.float64())
    assert {row["contract_version"] for row in rows} == {"2.0"}
    assert {row["tool_version"] for row in rows} == {"0.2.0"}
    assert {row["model_id"] for row in rows} == {"fake"}
    assert {row["result_type"] for row in rows} == {
        "single_label",
        "multi_label",
        "ordinal_score",
        "categorical_logprobs",
        "fixed_binary_probe",
        "label_yes_no_logprobs",
    }
    assert all(isinstance(row["gold_labels"], list) for row in rows)
    assert all(isinstance(row["validation_errors"], list) for row in rows)
    assert json.loads(rows[0]["parsed_output"]) is not None

    assert manifest["contract_version"] == "2.0"
    assert manifest["tool_version"] == "0.2.0"
    assert manifest["model_id"] == "fake"
    assert manifest["effective_config"]["run"]["id"] == "matrix_smoke__nested_json"


def test_only_atomically_published_parts_become_resumable(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    writer = core.PartWriter(run_dir, "single_label_json", target_rows=2)
    writer.append(_row(0))

    before = core.ResumeIndex(run_dir / "before.sqlite", "v2")
    assert before.seed_from(list((run_dir / "parts").glob("variant=*/part-*.parquet"))) == 0
    before.close()

    assert writer.append(_row(1)) is True
    after = core.ResumeIndex(run_dir / "after.sqlite", "v2")
    assert after.seed_from(list((run_dir / "parts").glob("variant=*/part-*.parquet"))) == 2
    assert after.contains(("single_label_json", "row-0", 0))
    assert after.contains(("single_label_json", "row-1", 1))
    after.close()


def _row(position: int) -> dict:
    row = dict.fromkeys(core.RESULT_SCHEMA.names)
    row.update(
        {
            "contract_version": "2.0",
            "tool_version": "0.2.0",
            "run_id": "run",
            "model_id": "fake",
            "dataset_id": "data",
            "variant_id": "single_label_json",
            "result_type": "single_label",
            "input_row_id": f"row-{position}",
            "source_position": position,
            "gold_labels": [],
            "validation_errors": [],
            "final_status": "completed",
        }
    )
    return row
