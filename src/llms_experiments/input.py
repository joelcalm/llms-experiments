"""Compatibility facade for pluggable input readers."""

from .inputs import InputReader, create_input_reader, iter_rows_for_source, read_rows, rows_for_source

__all__ = ["InputReader", "create_input_reader", "iter_rows_for_source", "read_rows", "rows_for_source"]
