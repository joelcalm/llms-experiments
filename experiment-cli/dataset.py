"""Small, dependency-light input loader."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path, data_format: str, id_column: str, text_column: str) -> list[dict[str, Any]]:
    if data_format == "csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    elif data_format == "jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif data_format == "parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:  # protected by config validation
        raise ValueError(f"Unsupported input format: {data_format}")
    for position, row in enumerate(rows):
        if id_column not in row or text_column not in row:
            raise ValueError(f"Input row {position} lacks `{id_column}` or `{text_column}`")
        row["_source_position"] = position
    return rows

