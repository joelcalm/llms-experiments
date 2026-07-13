"""Human-readable logging, JSON events, and optional GPU snapshots."""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any


class EventLogger:
    def __init__(self, log_path: Path, event_path: Path, level: str = "INFO") -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        self.events = event_path
        self.logger = logging.getLogger(f"experiment-cli.{event_path}")
        self.logger.handlers.clear()
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)

    def emit(self, event: str, **payload: Any) -> None:
        record = {"timestamp": time.time(), "event": event, **payload}
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.logger.info("%s %s", event, json.dumps(payload, sort_keys=True))


def gpu_snapshot() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        line = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=5).strip().splitlines()[0]
        used, total, utilisation = [int(value.strip()) for value in line.split(",")]
        snapshot: dict[str, Any] = {"available": True, "memory_used_mib": used, "memory_total_mib": total, "utilization_percent": utilisation}
        try:
            import torch
            if torch.cuda.is_available():
                snapshot.update({
                    "cuda_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
                    "cuda_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2),
                    "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                    "cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
                })
        except Exception:  # torch is optional for config validation and JSONL parsing
            pass
        return snapshot
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError, ValueError) as exc:
        return {"available": False, "error": str(exc)}
