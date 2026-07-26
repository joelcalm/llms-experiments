"""TSV input reader implementation."""

from __future__ import annotations

from ..delimited import DelimitedInputReader


class TsvInputReader(DelimitedInputReader):
    """Read tab-separated input rows from a TSV source file."""

    format_name = "tsv"
    delimiter = "\t"
