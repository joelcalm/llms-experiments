"""Shared base for delimited text readers."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from typing import Any

from .base import InputReader


class DelimitedInputReader(InputReader):
    """Read rows from delimiter-separated text files such as CSV or TSV."""

    delimiter = ","

    def iter_rows(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        """Stream rows from the file, skipping rows that do not match filters."""
        effective_limit = self.effective_limit(limit)
        delimiter = str(self.source.get("delimiter", self.delimiter))
        emitted = 0
        position = 0
        where = dict(self.source.get("where", {}))
        with self.path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle, delimiter=delimiter):
                if where and any(str(raw.get(key)) != str(value) for key, value in where.items()):
                    continue
                yield self.normalize(raw, position)
                emitted += 1
                position += 1
                if effective_limit is not None and emitted >= effective_limit:
                    break