"""Explicit input-reader factory and compatibility helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .base import InputReader
from .implementations.csv_reader import CsvInputReader
from .implementations.jsonl_reader import JsonLinesInputReader
from .implementations.nested_json_reader import NestedJsonInputReader
from .implementations.paired_tsv_reader import PairedTsvInputReader
from .implementations.parquet_reader import ParquetInputReader
from .implementations.tsv_reader import TsvInputReader

INPUT_READER_TYPES: dict[str, type[InputReader]] = {
    "csv": CsvInputReader,
    "tsv": TsvInputReader,
    "jsonl": JsonLinesInputReader,
    "parquet": ParquetInputReader,
    "nested_json": NestedJsonInputReader,
    "paired_tsv": PairedTsvInputReader,
}


def create_input_reader(source: Mapping[str, Any], root: str | Path) -> InputReader:
    """Build the concrete reader declared by an input source configuration."""
    name = str(source.get("format", ""))
    try:
        reader_type = INPUT_READER_TYPES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(INPUT_READER_TYPES))
        raise ValueError(f"unsupported input format {name!r}; expected one of: {supported}") from exc
    return reader_type(source, root)


def rows_for_source(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read all normalized rows for one configured source."""
    return create_input_reader(source, str(config["_root"])).read_rows(limit)


def iter_rows_for_source(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream normalized rows for one configured source."""
    yield from create_input_reader(source, str(config["_root"])).iter_rows(limit)


def source_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize the provenance for the configured input source."""
    return create_input_reader(config["input"], str(config["_root"])).provenance()


def read_rows(
    path: Path,
    data_format: str,
    id_column: str,
    text_column: str,
    source: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Read rows from a file path using explicit reader arguments."""
    declared = dict(
        source
        or {
            "path": str(path),
            "format": data_format,
            "id_column": id_column,
            "text_column": text_column,
        }
    )
    declared["path"] = str(path)
    declared.setdefault("format", data_format)
    declared.setdefault("id_column", id_column)
    declared.setdefault("text_column", text_column)
    return create_input_reader(declared, path.parent).read_rows()
