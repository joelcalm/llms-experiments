"""End-to-end coverage for the soft multi-label yes/no variant.

The soft multi-label probe fans each item out into one yes/no question per
label. These tests use the fake backend (no GPU) to lock the expansion contract:
exactly n_items * n_labels requests, deterministic per-label identities, correct
resume, and the prefix-cache-friendly prompt ordering.
"""

from __future__ import annotations

from pathlib import Path

import experiment_cli as cli
import pyarrow.parquet as pq
from conftest import REPO_ROOT, run_cli

CONFIG = "experiments/soft_multi_label_smoke.yaml"
VARIANT = "soft_multi_label_yes_no_logits"
LABELS = ["alpha", "beta"]  # dataset_labels in the smoke config


def _rows(out: Path) -> list[dict]:
    return pq.read_table(out / "results.parquet").to_pylist()


def test_expansion_emits_one_request_per_item_and_label(tmp_path: Path) -> None:
    out = tmp_path / "run"
    run_cli("run", "--config", CONFIG, "--backend", "fake", "--rows", "4", "--output", str(out))
    rows = _rows(out)
    assert len(rows) == 4 * len(LABELS)

    by_item: dict[str, list[dict]] = {}
    for row in rows:
        assert row["variant_id"] == VARIANT
        by_item.setdefault(row["input_row_id"], []).append(row)

    assert len(by_item) == 4, "every source item must appear exactly once"
    for probes in by_item.values():
        # Exactly the label set, no duplicates, each carrying its target_label.
        assert sorted(p["target_label"] for p in probes) == sorted(LABELS)
        positions = {p["target_label"]: p["source_position"] for p in probes}
        base = min(positions.values()) // len(LABELS)
        # Positions are the deterministic base*width + label_index offsets.
        assert positions == {label: base * len(LABELS) + index for index, label in enumerate(LABELS)}


def test_soft_multi_label_run_is_resumable(tmp_path: Path) -> None:
    import json

    out = tmp_path / "run"
    first = json.loads(run_cli("run", "--config", CONFIG, "--backend", "fake", "--rows", "4", "--output", str(out)))
    second = json.loads(run_cli("run", "--config", CONFIG, "--backend", "fake", "--rows", "4", "--output", str(out)))
    assert first["result_rows"] == 4 * len(LABELS)
    assert second["resume_skipped_rows"] == first["result_rows"]
    assert second["result_rows"] == first["result_rows"]


def test_target_question_follows_the_shared_prefix() -> None:
    """The label-specific question must come after the input text.

    That ordering is what lets vLLM's prefix cache reuse system+context+input
    across every label of an item; if it regressed to the front, each label would
    invalidate the cached prefix.
    """
    config = cli.load_config(str(Path(REPO_ROOT) / CONFIG), check_files=False)
    variant = next(v for v in config["variants"] if v["id"] == VARIANT)
    row = {"id": "x", "text": "SENTINEL_TEXT", "_source_position": 0, "_target_label": "alpha"}
    rendered = cli.rendered_prompt(config, variant, row)
    assert "SENTINEL_TEXT" in rendered
    assert "Does this text express alpha?" in rendered
    assert rendered.index("SENTINEL_TEXT") < rendered.index("Does this text express alpha?")
