"""Structured run events and append-only diagnostic records."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .configuration import resolve


class Events:
    """Write human-readable logs and machine-readable JSONL events."""

    def __init__(self, log_path: Path, event_path: Path, level: str = "INFO") -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = event_path
        self.logger = logging.getLogger(f"llms-experiments.{event_path}")
        self.logger.handlers.clear()
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def emit(self, event: str, **payload: Any) -> None:
        record = {"timestamp": time.time(), "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.logger.info("%s %s", event, json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


class ErrorLog:
    """Hold open one append-only JSONL diagnostic stream."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def write(self, **record: Any) -> None:
        payload = {"timestamp": time.time(), **record}
        self.handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> ErrorLog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_events(config: dict[str, Any], run_dir: Path, run_id: str) -> Events:
    """Create lane-safe paths for a run's logs and structured events."""

    logging_config = config.get("logging", {})
    if any(key in config.get("_override_keys", []) for key in ("run.id", "output.directory")):
        logging_config = {
            **logging_config,
            "file": str(run_dir / "logs" / f"{run_id}.log"),
            "events": str(run_dir / "logs" / f"{run_id}.events.jsonl"),
        }
    return Events(
        resolve(config, logging_config.get("file", f"logs/{run_id}.log")),
        resolve(config, logging_config.get("events", f"logs/{run_id}.events.jsonl")),
        str(logging_config.get("level", "INFO")),
    )


__all__ = ["ErrorLog", "Events", "make_events"]
