"""Bounded input readers for supported dataset formats."""

from ._core import iter_rows_for_source, read_rows

__all__ = ["iter_rows_for_source", "read_rows"]
