"""Per-variant warm-up tuning with OOM/context backoff."""
from __future__ import annotations

import time
from typing import Any, Callable

from engine import BackendFailure, synchronize_cuda
from logging_utils import gpu_snapshot


def tune_batch(backend: Any, variant: dict[str, Any], prompts: list[str], batch: dict[str, Any], emit: Callable[..., None], *, synchronize: bool = False) -> tuple[int, list[dict[str, Any]]]:
    if batch.get("mode", "auto") == "fixed":
        return int(batch.get("size", batch.get("candidates", [1])[0])), []
    candidates = sorted(set(int(value) for value in batch.get("candidates", [1])))
    warmup = prompts[:int(batch.get("warmup_rows", 64))] or prompts[:1]
    attempts: list[dict[str, Any]] = []
    safe: list[tuple[float, int]] = []
    for candidate in candidates:
        trial = warmup[:candidate]
        synchronize_cuda(synchronize)
        started = time.perf_counter()
        try:
            responses = backend.generate(trial, variant)
            synchronize_cuda(synchronize)
            elapsed = max(time.perf_counter() - started, 1e-9)
            result = {"candidate": candidate, "accepted": True, "rows_per_second": len(trial) / elapsed,
                      "tokens_per_second": sum(item.token_count for item in responses) / elapsed, "latency_seconds": elapsed,
                      "gpu": gpu_snapshot()}
            safe.append((result["rows_per_second"], candidate))
        except BackendFailure as exc:
            result = {"candidate": candidate, "accepted": False, "error": str(exc), "gpu": gpu_snapshot()}
            emit("batch_candidate_rejected", variant=variant["id"], **result)
            if batch.get("on_failure") == "halve" and candidate > int(batch.get("min_size", 1)):
                fallback = max(int(batch.get("min_size", 1)), candidate // 2)
                if fallback not in candidates:
                    candidates.append(fallback)
                    candidates.sort()
        attempts.append(result)
        emit("batch_candidate", variant=variant["id"], **result)
    if not safe:
        raise RuntimeError(f"No safe batch size for variant {variant['id']}")
    selected = max(safe)[1]
    emit("batch_selected", variant=variant["id"], batch_size=selected, candidates=attempts)
    return selected, attempts
