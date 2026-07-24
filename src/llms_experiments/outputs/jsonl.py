"""Lossless JSON Lines durable output store."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .base import OutputStore, ResultFileWriter
from .codec import decode_value, encode_value
from .schema import normalize_result_row


class _JsonLinesWriter(ResultFileWriter):
    def __init__(self, path: Path) -> None:
        self.handle = path.open("w", encoding="utf-8")

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            encoded = encode_value(normalize_result_row(dict(row)))
            self.handle.write(json.dumps(encoded, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")

    def close(self) -> None:
        self.handle.close()


class JsonLinesOutputStore(OutputStore):
    format_name = "jsonl"
    extension = "jsonl"

    def open_writer(self, path: Path) -> ResultFileWriter:
        path.parent.mkdir(parents=True, exist_ok=True)
        return _JsonLinesWriter(path)

    def iter_file(self, path: Path):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield normalize_result_row(decode_value(json.loads(line)))
