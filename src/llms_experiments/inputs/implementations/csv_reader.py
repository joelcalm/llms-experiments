"""CSV input reader implementation."""

from __future__ import annotations

from ..delimited import DelimitedInputReader


class CsvInputReader(DelimitedInputReader):
    """Read comma-separated input rows from a CSV source file."""

    format_name = "csv"
    delimiter = ","
