"""Nested JSON input reader implementation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ..base import InputReader
from ..utils import split_labels


class NestedJsonInputReader(InputReader):
    """Read records nested inside parent JSON objects and flatten labels."""

    format_name = "nested_json"

    def iter_rows(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        """Walk nested records, collect annotations, and emit normalized rows."""
        effective_limit = self.effective_limit(limit)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        records_key = str(self.source.get("records_key", "Tweets"))
        labels_key = str(self.source.get("labels_column", "annotations"))
        label_value_key = str(self.source.get("label_value_key", "annotation"))
        where = dict(self.source.get("where", {}))
        emitted = 0
        position = 0
        for parent in payload:
            for record in parent.get(records_key, []):
                if where and any(str(record.get(key)) != str(value) for key, value in where.items()):
                    continue
                labels: list[str] = []
                for annotation in record.get(labels_key, []):
                    value = annotation.get(label_value_key) if isinstance(annotation, dict) else annotation
                    labels.extend(split_labels(value))
                yield self.normalize({**record, "_gold_labels": sorted(set(labels))}, position)
                emitted += 1
                position += 1
                if effective_limit is not None and emitted >= effective_limit:
                    return
