"""Compatibility re-exports for input reader implementations."""

from __future__ import annotations

from .delimited import DelimitedInputReader
from .implementations.csv_reader import CsvInputReader
from .implementations.jsonl_reader import JsonLinesInputReader
from .implementations.nested_json_reader import NestedJsonInputReader
from .implementations.paired_tsv_reader import PairedTsvInputReader
from .implementations.parquet_reader import ParquetInputReader
from .implementations.tsv_reader import TsvInputReader

__all__ = [
    "CsvInputReader",
    "DelimitedInputReader",
    "JsonLinesInputReader",
    "NestedJsonInputReader",
    "PairedTsvInputReader",
    "ParquetInputReader",
    "TsvInputReader",
]
