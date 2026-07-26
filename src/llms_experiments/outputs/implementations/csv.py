"""Lossless schema-aware CSV durable output store."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..base import OutputStore, ResultFileWriter
from ..codec import decode_value, encode_value
from ..schema import RESULT_COLUMNS, normalize_result_row


class _CsvWriter(ResultFileWriter):
    """Writer helper for formatting and writing CSV result rows."""

    def __init__(self, path: Path) -> None:
        """Open output CSV file and write schema header."""
        self.handle = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=RESULT_COLUMNS)
        self.writer.writeheader()

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Encode and write a batch of result rows to CSV format."""
        for row in rows:
            normalized = normalize_result_row(dict(row))
            self.writer.writerow(
                {
                    name: json.dumps(encode_value(normalized[name]), ensure_ascii=False, separators=(",", ":"))
                    for name in RESULT_COLUMNS
                }
            )

    def close(self) -> None:
        """Close output CSV file handle."""
        self.handle.close()


class CsvOutputStore(OutputStore):
    """Schema-aware CSV implementation of OutputStore."""

    format_name = "csv"
    extension = "csv"

    def open_writer(self, path: Path) -> ResultFileWriter:
        """Open a new CSV result writer instance for the specified file path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        return _CsvWriter(path)

    def iter_file(self, path: Path):
        """Iterate over and decode normalized result rows from CSV file."""
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                yield normalize_result_row({name: decode_value(json.loads(row[name])) for name in RESULT_COLUMNS})
