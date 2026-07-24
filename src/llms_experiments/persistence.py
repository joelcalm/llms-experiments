"""Compatibility facade for interchangeable durable output stores."""

from .outputs import (
    RESULT_SCHEMA,
    AtomicPartWriter,
    CsvOutputStore,
    JsonLinesOutputStore,
    OutputStore,
    ParquetOutputStore,
    PartWriter,
    ResumeIndex,
    create_output_store,
    merge_parts,
)

__all__ = [
    "RESULT_SCHEMA",
    "AtomicPartWriter",
    "CsvOutputStore",
    "JsonLinesOutputStore",
    "OutputStore",
    "ParquetOutputStore",
    "PartWriter",
    "ResumeIndex",
    "create_output_store",
    "merge_parts",
]
