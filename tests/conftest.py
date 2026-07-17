"""Shared helpers for the experiment-cli test suite.

The runner is a single script rather than an installed package, so tests import
it by path.  This mirrors how the HTCondor workers invoke it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI = REPO_ROOT / "experiment-cli" / "experiment_cli.py"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

sys.path.insert(0, str(CLI.parent))

# Fields whose values legitimately change between identical runs, so they can
# never be part of a golden comparison.
#
# batch_size is in here for a non-obvious reason: `tune` picks the batch size by
# measuring throughput, so it varies with machine load rather than with config.
VOLATILE_ROW_FIELDS = {
    "latency_seconds",
    "rows_per_second",
    "gpu_snapshot",
    "batch_size",
}
VOLATILE_MANIFEST_KEYS = {
    "cpu_resource_guard",
    "gpu_preflight",
    "event_log",
    "source_provenance",
    "batch_response_path",
    "retry_request_path",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--golden-update",
        action="store_true",
        default=False,
        help="Rewrite the golden snapshots from the current code instead of comparing.",
    )


@pytest.fixture
def golden_update(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--golden-update"))


def run_cli(*args: str) -> str:
    """Invoke the CLI exactly the way the cluster workers do: by file path."""
    completed = subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"CLI failed ({completed.returncode}): {' '.join(args)}\n"
            f"--- stdout ---\n{completed.stdout[-3000:]}\n"
            f"--- stderr ---\n{completed.stderr[-3000:]}"
        )
    return completed.stdout


def normalise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop volatile fields and sort into a stable order for comparison."""
    cleaned = [{k: v for k, v in row.items() if k not in VOLATILE_ROW_FIELDS} for row in rows]
    return sorted(cleaned, key=lambda row: (str(row.get("variant_id")), int(row.get("source_position") or 0)))


def normalise_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Keep the stable, behaviour-defining parts of a manifest.

    `variants` keeps prompt_group_id but not the tuning measurements: the group
    id is a hash of the rendered prompt, which is exactly what a refactor must
    not change.
    """
    cleaned = {k: v for k, v in manifest.items() if k not in VOLATILE_MANIFEST_KEYS}
    variants = cleaned.get("variants")
    if isinstance(variants, dict):
        cleaned["variants"] = {
            name: {k: v for k, v in value.items() if k in {"prompt_group_id", "pending_rows", "external_batch"}}
            if isinstance(value, dict)
            else value
            for name, value in variants.items()
        }
    if isinstance(cleaned.get("datasets"), list):
        cleaned["datasets"] = [normalise_manifest(item) for item in cleaned["datasets"]]
    return cleaned


def read_results(run_dir: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return normalise_rows(pq.read_table(run_dir / "results.parquet").to_pylist())


def read_manifest(run_dir: Path, name: str = "manifest.json") -> dict[str, Any]:
    return normalise_manifest(json.loads((run_dir / name).read_text(encoding="utf-8")))


def assert_golden(name: str, payload: Any, update: bool) -> None:
    """Compare `payload` against the stored snapshot, or rewrite it."""
    path = GOLDEN_DIR / f"{name}.json"
    serialised = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if update:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialised, encoding="utf-8")
        return
    if not path.exists():
        raise AssertionError(f"Missing golden snapshot {path}. Regenerate with: pytest --golden-update")
    expected = path.read_text(encoding="utf-8")
    if expected != serialised:
        raise AssertionError(
            f"Golden mismatch for {name}.\n"
            f"Behaviour changed against the pre-refactor baseline.\n"
            f"If the change is intended, rerun with --golden-update and review the diff."
        )
