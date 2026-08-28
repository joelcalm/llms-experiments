"""Atomic, monitor-friendly checkpoints for streamed inference runs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def publish_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a compact checkpoint after a durable output part.

    A checkpoint is only written after the corresponding immutable output part
    has been renamed into place and its resume keys committed.  Readers can
    therefore use it as a progress hint without treating it as a substitute
    for the Parquet parts or the final manifest.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
