"""Regression tests for the standalone, generic experiment CLI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

CLI_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(CLI_ROOT))
from config import load_config, validate_config  # noqa: E402
from engine import BackendFailure, FakeBackend, Response  # noqa: E402
from prompts import render  # noqa: E402
from tuning import tune_batch  # noqa: E402
import cli  # noqa: E402

CONFIG = CLI_ROOT / "configs" / "local_all_modes_smoke.yaml"


def _fake_config(tmp_path: Path) -> dict:
    config = load_config(CONFIG)
    config["model"] = {"backend": "fake", "name": "fake"}
    config["batch"] = {"mode": "auto", "candidates": [1, 2], "warmup_rows": 2, "min_size": 1, "on_failure": "halve"}
    config["output"]["directory"] = str(tmp_path / "outputs")
    config["logging"] = {"file": str(tmp_path / "run.log"), "events": str(tmp_path / "events.jsonl")}
    payloads = {"single_label_json": {"label": "alpha"}, "multi_label_json": {"labels": ["alpha", "beta"]}, "ordinal_score_json": {"score": 3}}
    for variant in config["variants"]:
        if variant["id"] in payloads:
            variant["fake_response"] = payloads[variant["id"]]
    return config


def test_yaml_validation_and_five_generic_modes(tmp_path: Path) -> None:
    config = _fake_config(tmp_path)
    manifest = cli.run(config, backend=FakeBackend())
    results = pq.read_table(Path(config["output"]["directory"]) / "results.parquet").to_pylist()
    assert manifest["result_rows"] == 128 * 5
    assert {row["variant_id"] for row in results} == {variant["id"] for variant in config["variants"]}
    assert all("batch_size" in row and "gpu_snapshot" in row for row in results)
    assert (Path(config["output"]["directory"]) / "manifest.json").exists()
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8").count('"event"') > 5


def test_resume_does_not_duplicate_rows(tmp_path: Path) -> None:
    config = _fake_config(tmp_path)
    cli.run(config, backend=FakeBackend())
    resumed = cli.run(config, backend=FakeBackend())
    results = pq.read_table(Path(config["output"]["directory"]) / "results.parquet").to_pylist()
    assert len(results) == 128 * 5
    assert resumed["resume_skipped_rows"] == 128 * 5


def test_prompt_render_and_schema_contract() -> None:
    assert render("row={{ row_id }} text={{text}} unknown={{unknown}}", {"row_id": "7", "text": "hello"}) == "row=7 text=hello unknown={{unknown}}"
    invalid = {"run": {"id": "x"}, "input": {}, "model": {}, "variants": [], "output": {}}
    with pytest.raises(ValueError, match="input.path"):
        validate_config(invalid)


class _RetryBackend:
    def __init__(self) -> None:
        self.responses = ["not json", '{"label":"alpha"}']

    def generate(self, prompts: list[str], variant: dict) -> list[Response]:
        return [Response(self.responses.pop(0), 1) for _ in prompts]


def test_bounded_retry_records_second_attempt(tmp_path: Path) -> None:
    config = _fake_config(tmp_path)
    config["variants"] = [config["variants"][0]]
    config["batch"] = {"mode": "fixed", "size": 128, "candidates": [128]}
    config["input"]["path"] = str(tmp_path / "one_row.jsonl")
    Path(config["input"]["path"]).write_text('{"id":"one","text":"text"}\n', encoding="utf-8")
    config["input"]["format"] = "jsonl"
    cli.run(config, backend=_RetryBackend())
    row = pq.read_table(Path(config["output"]["directory"]) / "results.parquet").to_pylist()[0]
    assert row["attempt_count"] == 2
    assert row["validation_status"] == "valid"
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert '"event": "retry_started"' in events
    assert '"event": "retry_completed"' in events


class _OOMBackend:
    def generate(self, prompts: list[str], variant: dict) -> list[Response]:
        if len(prompts) > 2:
            raise BackendFailure("CUDA out of memory")
        return [Response("{}", 1) for _ in prompts]


def test_tuning_uses_oom_backoff_and_event_log() -> None:
    events: list[dict] = []
    size, attempts = tune_batch(_OOMBackend(), {"id": "generic", "request_mode": "generate"}, ["x"] * 8,
                                 {"mode": "auto", "candidates": [1, 4], "warmup_rows": 8, "on_failure": "halve", "min_size": 1},
                                 lambda event, **payload: events.append({"event": event, **payload}))
    assert size in {1, 2}
    assert any(not attempt["accepted"] for attempt in attempts)
    assert any(event["event"] == "batch_candidate_rejected" for event in events)


def test_prepare_writes_generic_jsonl_requests(tmp_path: Path) -> None:
    config = _fake_config(tmp_path)
    path = cli.prepare(config)
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert {"custom_id", "variant_id", "input_row_id", "prompt"} <= set(first)
