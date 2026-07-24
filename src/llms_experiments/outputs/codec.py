"""Lossless JSON-compatible codec shared by text output stores."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_FLOAT_TAG = "__llms_experiments_float__"


def encode_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {_FLOAT_TAG: "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
    if isinstance(value, Mapping):
        return {str(key): encode_value(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [encode_value(child) for child in value]
    return value


def decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {_FLOAT_TAG}:
            declared = value[_FLOAT_TAG]
            return float("nan") if declared == "nan" else float("inf") if declared == "inf" else -float("inf")
        return {key: decode_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [decode_value(child) for child in value]
    return value
