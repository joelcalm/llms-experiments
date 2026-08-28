#!/usr/bin/env python3
"""Validate an OSF-ready compact parsed-result release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from export_parsed_results import OUTPUT_SCHEMA, VALUE_COLUMNS


def _check_constant(array: pa.Array, expected: str, field: str) -> None:
    if not bool(pc.all(pc.equal(array, pa.scalar(expected))).as_py()):
        raise ValueError(f"{field} is not constant at {expected!r}")


def validate_release(release_root: Path) -> dict[str, Any]:
    """Validate every release Parquet and return a compact quality report."""

    release_root = release_root.resolve()
    manifest = json.loads((release_root / "manifest.json").read_text(encoding="utf-8"))
    files: list[dict[str, Any]] = []
    total_rows = 0
    total_ones = 0
    for expected in manifest["files"]:
        path = release_root / expected["relative_path"]
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing release Parquet: {path}")
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != OUTPUT_SCHEMA:
            raise ValueError(f"Schema mismatch: {path}")
        if parquet.metadata.num_rows != expected["output_rows"]:
            raise ValueError(f"Row-count mismatch: {path}")
        rows = 0
        ones = 0
        for batch in parquet.iter_batches(columns=list(OUTPUT_SCHEMA.names), batch_size=65536):
            rows += batch.num_rows
            _check_constant(batch.column("model_id"), expected["model_id"], "model_id")
            _check_constant(batch.column("strategy"), expected["strategy"], "strategy")
            for name in VALUE_COLUMNS:
                column = batch.column(name)
                if column.null_count:
                    raise ValueError(f"Null indicator values in {path}:{name}")
                minimum = pc.min(column).as_py()
                maximum = pc.max(column).as_py()
                if minimum < 0 or maximum > 1:
                    raise ValueError(f"Non-binary indicator values in {path}:{name}")
                ones += int(pc.sum(column).as_py() or 0)
            if expected["strategy"] == "single_label_json":
                total = batch.column(VALUE_COLUMNS[0])
                for name in VALUE_COLUMNS[1:]:
                    total = pc.add(total, batch.column(name))
                if not bool(pc.all(pc.equal(total, pa.scalar(1))).as_py()):
                    raise ValueError(f"Single-label row invariant failed in {path}")
        if rows != expected["output_rows"]:
            raise ValueError(f"Streamed row-count mismatch: {path}")
        files.append(
            {
                "relative_path": expected["relative_path"],
                "rows": rows,
                "value_ones": ones,
                "zero_fraction": 1.0 - (ones / (rows * len(VALUE_COLUMNS))) if rows else 1.0,
            }
        )
        total_rows += rows
        total_ones += ones
    return {
        "release_version": manifest["release_version"],
        "files": files,
        "file_count": len(files),
        "total_rows": total_rows,
        "total_value_cells": total_rows * len(VALUE_COLUMNS),
        "total_value_ones": total_ones,
        "overall_zero_fraction": 1.0 - (total_ones / (total_rows * len(VALUE_COLUMNS))) if total_rows else 1.0,
        "status": "valid",
    }


def main() -> int:
    """Run release validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_root", type=Path)
    args = parser.parse_args()
    report = validate_release(args.release_root)
    output = args.release_root / "validation_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
