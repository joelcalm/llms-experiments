"""CPU, GPU, and optional vLLM runtime resource management."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping
from functools import cache
from typing import Any

_GPU_CACHE: dict[str, Any] = {"taken_at": 0.0, "snapshot": None}


def _gpu_query() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            name, total, used, utilization, temperature = [item.strip() for item in line.split(",", 4)]
            rows.append(
                {
                    "name": name,
                    "memory_total_mb": int(total),
                    "memory_used_mb": int(used),
                    "utilization_percent": int(utilization),
                    "temperature_c": int(temperature),
                }
            )
        return {"available": bool(rows), "gpus": rows}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"available": False, "gpus": []}


def gpu(max_age_seconds: float = 0.0) -> dict[str, Any]:
    """Return a bounded-age NVIDIA telemetry snapshot."""

    now = time.monotonic()
    cached = _GPU_CACHE["snapshot"]
    if cached is not None and max_age_seconds > 0 and now - _GPU_CACHE["taken_at"] <= max_age_seconds:
        return dict(cached)
    snapshot = _gpu_query()
    _GPU_CACHE.update({"taken_at": now, "snapshot": snapshot})
    return dict(snapshot)


def gpu_preflight() -> dict[str, Any]:
    snapshot = gpu()
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        pass
    return {**snapshot, "cuda_available": cuda_available}


def sync_cuda(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def available_cpu_ids() -> list[int]:
    try:
        # os.sched_getaffinity is POSIX-only; absent on Windows and in typeshed for this platform.
        return sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return list(range(os.cpu_count() or 1))


def cpu_resource_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    cpu = dict(config.get("resources", {}).get("cpu", {}))
    available = available_cpu_ids()
    requested = cpu.get("cores", "auto")
    reserve = int(cpu.get("reserve_cores", 2))
    if requested == "all":
        count = len(available)
    elif requested == "auto":
        count = max(1, len(available) - reserve)
    else:
        count = min(len(available), int(requested))
    selected = available[:count]
    return {
        "available_cpu_ids": available,
        "selected_cpu_ids": selected,
        "thread_pool_size": int(cpu.get("thread_pool_size", 1)),
        "affinity": bool(cpu.get("affinity", True)),
    }


def apply_resource_guard(config: dict[str, Any]) -> dict[str, Any]:
    plan = cpu_resource_plan(config)
    if plan["affinity"] and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, plan["selected_cpu_ids"])
        except OSError:
            plan["affinity_applied"] = False
        else:
            plan["affinity_applied"] = True
    else:
        plan["affinity_applied"] = False
    threads = str(plan["thread_pool_size"])
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = threads
    config["_resource_guard"] = plan
    return plan


def configure_torch_cpu_threads(resource_guard: Mapping[str, Any] | None) -> None:
    if not resource_guard:
        return
    try:
        import torch

        torch.set_num_threads(int(resource_guard.get("thread_pool_size", 1)))
    except ImportError:
        return


@cache
def _gpu_compute_capability() -> tuple[int, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return int(major), int(minor)
    except (ImportError, RuntimeError):
        return None


def configure_vllm_environment(model: Mapping[str, Any]) -> dict[str, str]:
    """Apply explicitly configured VLLM variables and the SM12 sampler guard."""

    configured = {str(key): str(value) for key, value in dict(model.get("vllm_environment", {})).items()}
    if (
        (_gpu_compute_capability() or (0, 0))[0] >= 12
        and "VLLM_USE_FLASHINFER_SAMPLER" not in configured
        and "VLLM_USE_FLASHINFER_SAMPLER" not in os.environ
    ):
        configured["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ.update(configured)
    return configured


def serialise_snapshot(snapshot: Mapping[str, Any]) -> str:
    return json.dumps(dict(snapshot), ensure_ascii=False, sort_keys=True)
