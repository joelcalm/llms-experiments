"""Input-reader abstraction and shared row normalization."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .utils import normalize_gold_labels, split_labels


class InputReader(ABC):
    """Abstract source of normalized, position-stable input rows."""

    format_name: str

    def __init__(self, source: Mapping[str, Any], root: str | Path) -> None:
        """Store the declared source configuration and validate it up front."""
        self.source = dict(source)
        self.root = Path(root)
        self.validate()

    @property
    def path(self) -> Path:
        """Resolve the configured input path against the reader root."""
        path = Path(str(self.source["path"]))
        return path if path.is_absolute() else self.root / path

    def resolve(self, value: str | Path) -> Path:
        """Resolve a secondary configured path against the reader root."""
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def validate(self) -> None:
        """Reject incomplete reader declarations before any file access occurs."""
        for key in ("path", "format", "id_column", "text_column"):
            if not self.source.get(key):
                raise ValueError(f"input.{key} is required")

    @abstractmethod
    def iter_rows(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        """Yield normalized rows without changing their deterministic identities."""

    def read_rows(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Materialize all normalized rows into a list."""
        return list(self.iter_rows(limit))

    def normalize(self, row: Mapping[str, Any], position: int) -> dict[str, Any]:
        """Attach stable bookkeeping fields and normalize any gold labels."""
        normalized = dict(row)
        id_column = str(self.source["id_column"])
        text_column = str(self.source["text_column"])
        if id_column not in normalized or text_column not in normalized:
            raise ValueError(f"Input row {position} lacks `{id_column}` or `{text_column}`")
        if "_gold_labels" not in normalized and self.source.get("labels_column"):
            normalized["_gold_labels"] = normalize_gold_labels(
                split_labels(normalized.get(str(self.source["labels_column"]))),
                self.source.get("labels"),
            )
        normalized["_source_position"] = position
        return normalized

    def effective_limit(self, limit: int | None) -> int | None:
        """Use the explicit limit when present, otherwise the configured default."""
        value = limit if limit is not None else self.source.get("limit")
        if value is None:
            return None
        parsed = int(value)
        if parsed < 1:
            raise ValueError("row limit must be positive")
        return parsed

    def provenance_paths(self) -> list[Path]:
        """Return the files that define this reader's provenance."""
        return [self.path]

    def provenance(self) -> dict[str, Any]:
        """Summarize the input files into a stable provenance fingerprint."""
        records = []
        for path in self.provenance_paths():
            stat = path.stat()
            records.append({"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        metadata_hash = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        return {"format": self.format_name, "files": records, "metadata_hash": metadata_hash}
