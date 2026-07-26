"""Parquet input reader implementation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq

from ..base import InputReader


class ParquetInputReader(InputReader):
    """Read rows from a Parquet file source."""

    format_name = "parquet"

    def iter_rows(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        """Stream rows batch by batch so large files do not need full loading."""
        effective_limit = self.effective_limit(limit)
        where = dict(self.source.get("where", {}))
        emitted = 0
        position = 0
        parquet = pq.ParquetFile(self.path)
        for batch in parquet.iter_batches():
            for row in batch.to_pylist():
                if where and any(str(row.get(key)) != str(value) for key, value in where.items()):
                    continue
                yield self.normalize(row, position)
                emitted += 1
                position += 1
                if effective_limit is not None and emitted >= effective_limit:
                    return
