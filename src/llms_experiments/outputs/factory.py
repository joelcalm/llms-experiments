"""Explicit output-store factory and compatibility functions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .base import AtomicPartWriter, OutputStore
from .implementations.csv import CsvOutputStore
from .implementations.jsonl import JsonLinesOutputStore
from .implementations.parquet import ParquetOutputStore

OUTPUT_STORE_TYPES: dict[str, type[OutputStore]] = {
    "parquet": ParquetOutputStore,
    "csv": CsvOutputStore,
    "jsonl": JsonLinesOutputStore,
}


def create_output_store(output: Mapping[str, Any], run_dir: str | Path | None = None) -> OutputStore:
    """Build the concrete OutputStore declared by output configuration."""
    name = str(output.get("format", "parquet"))
    try:
        store_type = OUTPUT_STORE_TYPES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(OUTPUT_STORE_TYPES))
        raise ValueError(f"unsupported output format {name!r}; expected one of: {supported}") from exc
    directory = Path(run_dir) if run_dir is not None else Path(str(output["directory"]))
    return store_type(directory)


def create_output_store_for_path(path: Path) -> OutputStore:
    """Infer and build concrete OutputStore from a file path extension."""
    suffix = path.suffix.lower().lstrip(".")
    try:
        store_type = OUTPUT_STORE_TYPES[suffix]
    except KeyError as exc:
        raise ValueError(f"cannot infer output store from {path}") from exc
    parts = path.parts
    run_dir = path.parent.parent.parent if "parts" in parts else path.parent
    return store_type(run_dir)


class PartWriter(AtomicPartWriter):
    """Backward-compatible Parquet part writer."""

    def __init__(self, run_dir: Path, variant_id: str, target_rows: int = 4096) -> None:
        """Initialize part writer using default Parquet output store."""
        super().__init__(ParquetOutputStore(run_dir), variant_id, target_rows)


def merge_parts(
    run_dir: Path,
    expected_hashes: Mapping[str, str] | None = None,
    retried_keys: set[tuple[str, str, int]] | None = None,
) -> int:
    """Finalize and merge partition parts into canonical result files."""
    return ParquetOutputStore(run_dir).finalize(expected_hashes, retried_keys)
