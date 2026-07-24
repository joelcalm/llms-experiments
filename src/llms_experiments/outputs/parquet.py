"""Apache Parquet durable output store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .base import OutputStore, ResultFileWriter
from .schema import RESULT_SCHEMA, normalize_result_row


class _ParquetWriter(ResultFileWriter):
    def __init__(self, path: Path) -> None:
        self.writer = pq.ParquetWriter(path, RESULT_SCHEMA, compression="zstd")

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if rows:
            self.writer.write_table(
                pa.Table.from_pylist([normalize_result_row(dict(row)) for row in rows], RESULT_SCHEMA)
            )

    def close(self) -> None:
        self.writer.close()


class ParquetOutputStore(OutputStore):
    format_name = "parquet"
    extension = "parquet"

    def open_writer(self, path: Path) -> ResultFileWriter:
        path.parent.mkdir(parents=True, exist_ok=True)
        return _ParquetWriter(path)

    def iter_file(self, path: Path):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches():
            for row in batch.to_pylist():
                yield normalize_result_row(row)
