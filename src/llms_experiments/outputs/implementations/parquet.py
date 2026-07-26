"""Apache Parquet durable output store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..base import OutputStore, ResultFileWriter
from ..schema import RESULT_SCHEMA, normalize_result_row


class _ParquetWriter(ResultFileWriter):
    """Writer helper for formatting and writing Parquet result tables."""

    def __init__(self, path: Path) -> None:
        """Open Apache Parquet table writer with ZSTD compression."""
        self.writer = pq.ParquetWriter(path, RESULT_SCHEMA, compression="zstd")

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Encode and write a batch of result rows to Apache Parquet format."""
        if rows:
            self.writer.write_table(
                pa.Table.from_pylist([normalize_result_row(dict(row)) for row in rows], RESULT_SCHEMA)
            )

    def close(self) -> None:
        """Close underlying Parquet writer."""
        self.writer.close()


class ParquetOutputStore(OutputStore):
    """Apache Parquet implementation of OutputStore."""

    format_name = "parquet"
    extension = "parquet"

    def open_writer(self, path: Path) -> ResultFileWriter:
        """Open a new Parquet result writer instance for the specified file path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        return _ParquetWriter(path)

    def iter_file(self, path: Path):
        """Iterate over and yield normalized result rows from Parquet file."""
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches():
            for row in batch.to_pylist():
                yield normalize_result_row(row)
