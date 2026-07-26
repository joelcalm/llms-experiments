"""Shared helpers for input readers."""

from __future__ import annotations

import json
import re
from typing import Any


def split_labels(value: Any) -> list[str]:
    """Turn label values into a flat list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return [str(item) for item in decoded if str(item)]
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value)]


def _slugify(label: str) -> str:
    """Normalize a label into a comparison-friendly key."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def normalize_gold_labels(raw: list[str], canonical: list[str] | None) -> list[str]:
    """Map raw labels back to canonical names when a canonical list is available."""
    if not canonical:
        return raw
    canonical_set = set(canonical)
    lookup = {_slugify(label): label for label in canonical}
    return [label if label in canonical_set else lookup.get(_slugify(label), label) for label in raw]
