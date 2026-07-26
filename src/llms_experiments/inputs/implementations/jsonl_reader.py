"""JSON Lines input reader implementation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ..base import InputReader


class JsonLinesInputReader(InputReader):
    """Read line-delimited JSON objects from a JSONL source file."""

    format_name = "jsonl"

    def iter_rows(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        """Stream parsed JSON objects, respecting filters and row limits."""
        effective_limit = self.effective_limit(limit)
        where = dict(self.source.get("where", {}))
        emitted = 0
        position = 0
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if where and any(str(row.get(key)) != str(value) for key, value in where.items()):
                    continue
                yield self.normalize(row, position)
                emitted += 1
                position += 1
                if effective_limit is not None and emitted >= effective_limit:
                    break
