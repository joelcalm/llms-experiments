"""Backend failures must survive into the results as failures.

This pins the bug that motivated merging the two execution engines: the
non-streaming engine never inspected `Response.backend_error`, so a failed
nvidia_api call was recorded as a completed row with empty candidates. The
streaming engine checked it; the two had silently drifted apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import experiment_cli as cli
import pytest
from conftest import REPO_ROOT

SMOKE = REPO_ROOT / "experiments" / "local_all_modes_smoke.yaml"


class FailingBackend:
    """Mimics NvidiaAPIBackend giving up after its HTTP retries.

    `raw` is empty and `backend_error` carries the whole story, matching the
    real backend: it no longer fabricates a `{"nvidia_api_error": ...}` body
    that a later retry could mistake for an off-schema model answer.
    """

    def generate(self, prompts: list[str], variant: dict) -> list[cli.Response]:
        return [cli.Response("", 0, None, "http_500: upstream exploded") for _ in prompts]

    def close(self) -> None:
        return None


class InvalidBackend:
    """A healthy backend serving a model that answers off-schema."""

    def generate(self, prompts: list[str], variant: dict) -> list[cli.Response]:
        return [cli.Response(json.dumps({"label": "not-a-permitted-label"}), 5, None, None) for _ in prompts]

    def close(self) -> None:
        return None


@pytest.fixture
def smoke_config(tmp_path: Path):
    def _load(**overrides: object) -> dict:
        config = cli.load_config(
            SMOKE,
            [
                f"output.directory={tmp_path / 'out'}",
                "validation.retry.enabled=false",
                *[f"{k}={v}" for k, v in overrides.items()],
            ],
            check_files=True,
        )
        config["logging"] = {"file": str(tmp_path / "r.log"), "events": str(tmp_path / "e.jsonl")}
        return config

    return _load


@pytest.mark.parametrize("streaming", [False, True])
def test_backend_error_is_never_reported_as_completed(smoke_config, streaming: bool) -> None:
    config = smoke_config()
    config["streaming"] = {"enabled": streaming}
    config["variants"] = [v for v in config["variants"] if v["id"] == "single_label_code_logits"]

    manifest = cli.run(config, FailingBackend(), row_limit=4)

    rows = _read_rows(Path(manifest["run_id"] and config["output"]["directory"]))
    assert rows, "the failed rows must still be recorded, not dropped"
    assert {row["final_status"] for row in rows} == {"failed_backend"}
    assert all("backend_error" in (row["validation_errors"] or "") for row in rows)


@pytest.mark.parametrize("streaming", [False, True])
def test_outage_during_inline_retry_stays_failed_backend(smoke_config, streaming: bool) -> None:
    """An outage that persists through the retry loop must not become validation.

    The inline retry loop used to read only `response.raw` and re-validate it,
    ignoring `backend_error`. When a retry hit a fresh outage the empty/foreign
    body validated as off-schema, so the row was written `failed_validation` and
    then never re-attempted on resume -- the outage silently ate those rows.
    Here the backend fails on every attempt, including the retries.
    """
    config = smoke_config(**{"validation.retry.enabled": "true"})
    config["streaming"] = {"enabled": streaming}
    config["variants"] = [v for v in config["variants"] if v["id"] == "single_label_json"]

    cli.run(config, FailingBackend(), row_limit=4)

    rows = _read_rows(Path(config["output"]["directory"]))
    assert {row["final_status"] for row in rows} == {"failed_backend"}, "a retried outage is still an outage"
    assert all("backend_error" in (row["validation_errors"] or "") for row in rows)


@pytest.mark.parametrize("streaming", [False, True])
def test_resume_retries_rows_a_broken_backend_lost(smoke_config, streaming: bool) -> None:
    """A transient outage must not permanently poison the rows it hit.

    Resume is keyed on row identity, so it used to skip anything already on
    disk regardless of status: after an API outage, re-running left the broken
    rows broken forever. Only `failed_backend` is retried -- see
    test_validation_failures_are_not_retried_on_resume for the other half.
    """
    config = smoke_config()
    config["streaming"] = {"enabled": streaming}
    config["variants"] = [v for v in config["variants"] if v["id"] == "single_label_code_logits"]

    cli.run(config, FailingBackend(), row_limit=4)
    broken = _read_rows(Path(config["output"]["directory"]))
    assert {row["final_status"] for row in broken} == {"failed_backend"}

    manifest = cli.run(config, cli.FakeBackend(), row_limit=4)

    rows = _read_rows(Path(config["output"]["directory"]))
    assert {row["final_status"] for row in rows} == {"completed"}, "the outage rows must be re-attempted"
    assert len(rows) == 4, "the superseded attempt must not survive the merge"
    assert manifest["resume_skipped_rows"] == 0, "a failed_backend row is not a skippable row"
    keys = [(row["variant_id"], row["input_row_id"], row["source_position"]) for row in rows]
    assert len(set(keys)) == len(keys), "retrying must not duplicate result keys"


@pytest.mark.parametrize("streaming", [False, True])
def test_validation_failures_are_not_retried_on_resume(smoke_config, streaming: bool) -> None:
    """Deterministic failures stay done, or every resume re-burns them.

    A model that answers off-schema will answer off-schema again, so retrying
    `failed_validation` on resume would pay full GPU cost each time to arrive
    at the identical row. Only infrastructure failures are worth re-attempting.
    """
    config = smoke_config()
    config["streaming"] = {"enabled": streaming}
    config["variants"] = [v for v in config["variants"] if v["id"] == "single_label_json"]

    cli.run(config, InvalidBackend(), row_limit=4)
    first = _read_rows(Path(config["output"]["directory"]))
    assert {row["final_status"] for row in first} == {"failed_validation"}

    manifest = cli.run(config, cli.FakeBackend(), row_limit=4)

    rows = _read_rows(Path(config["output"]["directory"]))
    assert {row["final_status"] for row in rows} == {"failed_validation"}, "a model failure is not retried"
    assert manifest["resume_skipped_rows"] == 4
    assert len(rows) == 4


def _read_rows(run_dir: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(run_dir / "results.parquet").to_pylist()
