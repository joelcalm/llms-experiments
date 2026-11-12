"""A chunk may exceed engine concurrency so the engine queue stays fed."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from llms_experiments.backends import AsyncStreamingBackend, Backend, BackendFailure
from llms_experiments.events import Events
from llms_experiments.orchestration import run, tune, tune_async
from llms_experiments.processors import GenerationRequest, RawModelResponse, ResponseRequirements


class RecordingBackend(Backend):
    """Records the size of every chunk it is handed."""

    def __init__(self) -> None:
        self.chunk_sizes: list[int] = []

    def generate(
        self,
        prompts: Sequence[str],
        request: GenerationRequest | Mapping[str, Any],
    ) -> list[RawModelResponse]:
        self.chunk_sizes.append(len(prompts))
        return [RawModelResponse("ok", 1) for _ in prompts]


class CapacityBackend(RecordingBackend):
    """Simulate a backend whose first unsafe request is an OOM."""

    def generate(
        self,
        prompts: Sequence[str],
        request: GenerationRequest | Mapping[str, Any],
    ) -> list[RawModelResponse]:
        if len(prompts) > 128:
            raise BackendFailure("out of memory")
        return super().generate(prompts, request)


class ThresholdBackend(RecordingBackend):
    """Simulate a backend with a non-power-of-two safe capacity."""

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity

    def generate(
        self,
        prompts: Sequence[str],
        request: GenerationRequest | Mapping[str, Any],
    ) -> list[RawModelResponse]:
        if len(prompts) > self.capacity:
            raise BackendFailure("out of memory")
        return super().generate(prompts, request)


class AsyncThresholdBackend(AsyncStreamingBackend):
    """Simulate an async engine whose safe concurrency is non-power-of-two."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.restart_count = 0

    def generate(
        self,
        prompts: Sequence[str],
        request: GenerationRequest | Mapping[str, Any],
    ) -> list[RawModelResponse]:
        raise AssertionError("async tuning must use generate_stream")

    async def generate_stream(
        self,
        prompts: Iterable[tuple[str, str]],
        request: GenerationRequest | Mapping[str, Any],
        *,
        max_in_flight: int,
    ) -> AsyncIterator[tuple[str, RawModelResponse]]:
        if max_in_flight > self.capacity:
            raise BackendFailure("out of memory")
        for request_id, _ in prompts:
            yield request_id, RawModelResponse("ok", 1)

    def restart_engine(self) -> None:
        self.restart_count += 1


def _config(tmp_path: Path, batch: dict[str, Any]) -> dict[str, Any]:
    source = tmp_path / "input.csv"
    source.write_text("id,text\n" + "".join(f"{index},row{index}\n" for index in range(8)), encoding="utf-8")
    (tmp_path / "prompt.md").write_text("Classify {{text}}", encoding="utf-8")
    return {
        "_root": str(tmp_path),
        "_override_keys": [],
        "run": {"id": "queue-depth", "dataset_id": "test"},
        "input": {"path": str(source), "format": "csv", "id_column": "id", "text_column": "text"},
        "model": {"name": "fake", "backend": "fake", "max_num_seqs": 2},
        "variants": [
            {
                "id": "raw",
                "prompts": ["prompt.md"],
                "processor": {"result": "raw", "stages": [{"type": "identity"}]},
            }
        ],
        "output": {"directory": str(tmp_path / "out"), "format": "jsonl"},
        "batch": batch,
        "logging": {
            "file": str(tmp_path / "run.log"),
            "events": str(tmp_path / "events.jsonl"),
        },
    }


@pytest.mark.parametrize(
    ("queue_depth", "expected_chunk"),
    [(None, 2), (1, 2), (4, 8)],
)
def test_queue_depth_scales_the_chunk_above_max_num_seqs(
    tmp_path: Path, queue_depth: int | None, expected_chunk: int
) -> None:
    batch: dict[str, Any] = {"mode": "fixed", "size": 8, "candidates": [8], "min_size": 1}
    if queue_depth is not None:
        batch["queue_depth"] = queue_depth
    backend = RecordingBackend()

    run(_config(tmp_path, batch), backend=backend)

    assert backend.chunk_sizes
    assert max(backend.chunk_sizes) == expected_chunk


def test_dynamic_limit_doubles_until_first_capacity_failure(tmp_path: Path) -> None:
    backend = CapacityBackend()
    events = Events(tmp_path / "run.log", tmp_path / "events.jsonl")
    request = GenerationRequest("raw", ResponseRequirements(max_tokens=1))

    selected, attempts = tune(
        backend,
        request,
        ["prompt"] * 4096,
        {"mode": "auto", "dynamic_limit": True, "start_size": 64, "warmup_rows": 4096},
        events,
        False,
    )
    events.close()

    assert [int(item["candidate"]) for item in attempts] == [64, 128, 256]
    assert attempts[-1]["accepted"] is False
    assert selected in {64, 128}
    assert backend.chunk_sizes == [64, 128]


def test_dynamic_limit_can_refine_to_a_non_power_of_two_capacity(tmp_path: Path) -> None:
    backend = ThresholdBackend(capacity=200)
    events = Events(tmp_path / "run.log", tmp_path / "events.jsonl")
    request = GenerationRequest("raw", ResponseRequirements(max_tokens=1))

    selected, attempts = tune(
        backend,
        request,
        ["prompt"] * 512,
        {
            "mode": "auto",
            "dynamic_limit": True,
            "refine_after_failure": True,
            "start_size": 64,
            "warmup_rows": 512,
        },
        events,
        False,
    )
    events.close()

    accepted = [int(item["candidate"]) for item in attempts if item["accepted"]]
    assert max(accepted) == 200
    assert selected in accepted
    assert 192 in accepted


def test_dynamic_limit_is_not_clamped_to_engine_queue_depth(tmp_path: Path) -> None:
    backend = RecordingBackend()

    run(
        _config(
            tmp_path,
            {
                "mode": "auto",
                "dynamic_limit": True,
                "start_size": 2,
                "candidates": [2],
                "warmup_rows": 8,
                "min_size": 1,
                "queue_depth": 1,
            },
        ),
        backend=backend,
    )

    assert max(backend.chunk_sizes) == 8


def test_dynamic_limit_extends_probe_window_until_input_exhausted(tmp_path: Path) -> None:
    backend = RecordingBackend()

    run(
        _config(
            tmp_path,
            {
                "mode": "auto",
                "dynamic_limit": True,
                "start_size": 2,
                "candidates": [2],
                "warmup_rows": 2,
                "min_size": 1,
                "queue_depth": 1,
            },
        ),
        backend=backend,
    )

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    extensions = [event for event in events if event["event"] == "batch_window_extended"]
    accepted = [event["candidate"] for event in events if event["event"] == "batch_candidate" and event["accepted"]]

    assert [(event["old_rows"], event["new_rows"]) for event in extensions] == [(2, 4), (4, 8)]
    assert max(accepted) == 8


def test_async_dynamic_limit_bisects_after_restart(tmp_path: Path) -> None:
    backend = AsyncThresholdBackend(capacity=200)
    events = Events(tmp_path / "run.log", tmp_path / "events.jsonl")
    request = GenerationRequest("raw", ResponseRequirements(max_tokens=1))
    loop = asyncio.new_event_loop()
    try:
        selected, attempts = tune_async(
            backend,
            request,
            ["prompt"] * 512,
            {
                "mode": "auto",
                "dynamic_limit": True,
                "refine_after_failure": True,
                "start_size": 64,
            },
            events,
            loop,
            512,
        )
    finally:
        loop.close()
        events.close()

    accepted = [item["candidate"] for item in attempts if item["accepted"]]
    assert max(accepted) == 200
    assert selected in accepted
    assert attempts[0]["candidate"] == 64
    assert attempts[2]["candidate"] == 256
    assert attempts[2]["accepted"] is False
    assert any(item["candidate"] == 200 and item["accepted"] for item in attempts)
    assert backend.restart_count >= 1
