"""Resumable Parquet output with a stable, flat contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def load_existing(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "results.parquet"
    return pq.read_table(path).to_pylist() if path.exists() else []


def write_results(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, run_dir / "results.parquet", compression="zstd")
    for variant in sorted({str(row["variant_id"]) for row in rows}):
        part = run_dir / "parts" / f"variant={variant}"
        part.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([row for row in rows if row["variant_id"] == variant]), part / "part-00000.parquet", compression="zstd")


def serialise(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)

