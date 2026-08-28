"""Continuous-dispatch persistence must survive an orderly interruption."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from llms_experiments.backends import AsyncStreamingBackend
from llms_experiments.orchestration import RunInterrupted, run
from llms_experiments.outputs import create_output_store
from llms_experiments.processors import GenerationRequest, RawModelResponse


class RecordingAsyncBackend(AsyncStreamingBackend):
    """CPU-only stand-in that consumes only the requested bounded queue."""

    def __init__(self, interrupt_after: int | None = None, fail_after: int | None = None) -> None:
        self.interrupt_after = interrupt_after
        self.fail_after = fail_after
        self.generated = 0
        self.maximum_pending = 0
        self.loop_ids: set[int] = set()

    def generate(
        self,
        prompts: Sequence[str],
        request: GenerationRequest | Mapping[str, Any],
    ) -> list[RawModelResponse]:
        raise AssertionError("main async dispatch must not call batch generate")

    async def generate_stream(
        self,
        prompts: Iterable[tuple[str, str]],
        request: GenerationRequest | Mapping[str, Any],
        *,
        max_in_flight: int,
    ) -> AsyncIterator[tuple[str, RawModelResponse]]:
        pending: list[tuple[str, str]] = []
        self.loop_ids.add(id(asyncio.get_running_loop()))
        iterator = iter(prompts)
        while True:
            while len(pending) < max_in_flight:
                try:
                    pending.append(next(iterator))
                except StopIteration:
                    break
            self.maximum_pending = max(self.maximum_pending, len(pending))
            if not pending:
                return
            request_id, _ = pending.pop()
            self.generated += 1
            if self.interrupt_after is not None and self.generated > self.interrupt_after:
                raise RunInterrupted("test interruption")
            if self.fail_after is not None and self.generated > self.fail_after:
                raise RuntimeError("test backend failure")
            yield request_id, RawModelResponse('{"label":"alpha"}', 1)


def _config(tmp_path: Path) -> dict[str, Any]:
    source = tmp_path / "input.csv"
    source.write_text("id,text\n" + "".join(f"{index},row{index}\n" for index in range(5)), encoding="utf-8")
    (tmp_path / "prompt.md").write_text("Classify {{text}}", encoding="utf-8")
    return {
        "_root": str(tmp_path),
        "_override_keys": [],
        "run": {"id": "async-checkpoint", "dataset_id": "test", "dataset_labels": ["alpha"]},
        "input": {"path": str(source), "format": "csv", "id_column": "id", "text_column": "text"},
        "model": {"name": "fake", "backend": "local_vllm_async", "max_num_seqs": 2},
        "variants": [
            {
                "id": "single_label_json",
                "prompts": ["prompt.md"],
                "processor": {
                    "result": "single_label",
                    "stages": [
                        {"type": "json_decode"},
                        {"type": "json_schema", "schema": "schema.json", "enum_from": "dataset_labels"},
                    ],
                },
            }
        ],
        "output": {"directory": str(tmp_path / "out"), "format": "parquet"},
        "batch": {"mode": "fixed", "size": 1, "candidates": [1], "warmup_rows": 1},
        "streaming": {
            "enabled": True,
            "dispatch": "async_vllm",
            "max_in_flight": 2,
            "output_chunk_rows": 2,
        },
        "logging": {"file": str(tmp_path / "run.log"), "events": str(tmp_path / "events.jsonl")},
    }


def test_async_dispatch_is_bounded_and_publishes_checkpointed_parts(tmp_path: Path) -> None:
    (tmp_path / "schema.json").write_text(
        json.dumps({"type": "object", "properties": {"label": {"enum": ["alpha"]}}, "required": ["label"]}),
        encoding="utf-8",
    )
    config = _config(tmp_path)
    backend = RecordingAsyncBackend(interrupt_after=3)

    with pytest.raises(RunInterrupted):
        run(config, backend)

    run_dir = Path(config["output"]["directory"])
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["state"] == "interrupted"
    assert checkpoint["durable_result_rows"] == 3
    assert checkpoint["durable_part_count"] == 2
    assert backend.maximum_pending <= 2

    resumed = RecordingAsyncBackend()
    manifest = run(config, resumed)
    rows = create_output_store(config["output"], run_dir).read_final()

    assert resumed.generated == 2
    assert manifest["result_rows"] == 5
    assert len(rows) == 5
    finished = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert finished["state"] == "completed"
    assert finished["durable_result_rows"] == 5


def test_async_dispatch_reuses_one_event_loop_for_multiple_variants(tmp_path: Path) -> None:
    """A shared AsyncLLM must not cross a closed loop between strategy lanes."""
    (tmp_path / "schema.json").write_text(
        json.dumps({"type": "object", "properties": {"label": {"enum": ["alpha"]}}, "required": ["label"]}),
        encoding="utf-8",
    )
    config = _config(tmp_path)
    second = deepcopy(config["variants"][0])
    second["id"] = "multi_label_json"
    config["variants"].append(second)
    backend = RecordingAsyncBackend()

    manifest = run(config, backend)

    assert manifest["result_rows"] == 10
    assert len(backend.loop_ids) == 1


def test_async_failure_marks_the_last_durable_boundary(tmp_path: Path) -> None:
    """A backend failure must not leave the recovery checkpoint as running."""
    (tmp_path / "schema.json").write_text(
        json.dumps({"type": "object", "properties": {"label": {"enum": ["alpha"]}}, "required": ["label"]}),
        encoding="utf-8",
    )
    config = _config(tmp_path)

    with pytest.raises(RuntimeError, match="test backend failure"):
        run(config, RecordingAsyncBackend(fail_after=3))

    checkpoint = json.loads((Path(config["output"]["directory"]) / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["state"] == "failed"
    assert checkpoint["durable_result_rows"] == 3
