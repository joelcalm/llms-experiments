"""Developer throughput benchmarks built on the public modular contracts."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .backends import create_backend
from .backends.utils import extract_token_positions
from .batching import batch_command_args
from .configuration import resolve
from .inputs import rows_for_source
from .orchestration import generation_request
from .processors import RawModelResponse
from .prompting import (
    build_requests,
    materialize_variant,
    rendered_prompt,
    variant_config_hash,
    variant_schema,
)
from .runtime import configure_vllm_environment, gpu, sync_cuda


def benchmark_rows(config: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    if "datasets" in config:
        raise ValueError("benchmark accepts one input; benchmark each dataset lane separately")
    rows = rows_for_source(config, config["input"], limit)
    requested = int(limit if limit is not None else config.get("benchmark", {}).get("rows", len(rows)))
    if requested < 1:
        raise ValueError("benchmark.rows must be positive")
    return rows[:requested]


def benchmark_python(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    settings = config.get("benchmark", {})
    batch_size = int(settings.get("batch_size", config["model"].get("max_num_seqs", 1)))
    warmup = int(settings.get("warmup_requests", 0))
    repeats = int(settings.get("repeats", 1))
    if batch_size < 1 or repeats < 1 or warmup < 0:
        raise ValueError("benchmark.batch_size, repeats, and warmup_requests are invalid")
    load_started = time.perf_counter()
    backend = create_backend(config["model"], config.get("_resource_guard"))
    load_seconds = time.perf_counter() - load_started
    entries = []
    for configured in config["variants"]:
        variant = materialize_variant(config, configured)
        processor = variant["_processor"]
        schema = variant_schema(config, variant)
        request = generation_request(config, variant, processor, schema)
        prompts = [rendered_prompt(config, variant, dict(row), schema) for row in processor.prepare_rows(rows)]
        entries.append((request, prompts))
    try:
        if warmup:
            for request, prompts in entries:
                backend.generate(prompts[:warmup], request)
        measurements: list[dict[str, Any]] = []
        for repeat in range(repeats):
            before = gpu()
            started = time.perf_counter()
            token_count = 0
            completed = 0
            for request, prompts in entries:
                for offset in range(0, len(prompts), batch_size):
                    responses = backend.generate(prompts[offset : offset + batch_size], request)
                    completed += len(responses)
                    token_count += sum(response.token_count for response in responses)
            sync_cuda(bool(config["model"].get("synchronize_cuda", False)))
            elapsed = max(time.perf_counter() - started, 1e-9)
            measurements.append(
                {
                    "repeat": repeat + 1,
                    "requests": completed,
                    "tokens": token_count,
                    "wall_seconds": elapsed,
                    "requests_per_second": completed / elapsed,
                    "tokens_per_second": token_count / elapsed,
                    "gpu_before": before,
                    "gpu_after": gpu(),
                }
            )
    finally:
        backend.close()
    return {
        "measurements": measurements,
        "model_load_seconds": load_seconds,
        "includes_model_startup": False,
        "batch_size": batch_size,
        "warmup_requests_per_variant": warmup,
    }


def _response_from_api(completion: Any) -> RawModelResponse:
    choice = completion.choices[0]
    usage = getattr(completion, "usage", None)
    return RawModelResponse(
        text=str(getattr(choice.message, "content", None) or ""),
        token_count=int(getattr(usage, "completion_tokens", 0) or 0),
        token_positions=extract_token_positions(getattr(getattr(choice, "logprobs", None), "content", None)),
    )


def benchmark_api(config: dict[str, Any], requests: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The API benchmark requires the openai package") from exc
    settings = config.get("benchmark", {})
    api_settings = settings.get("api", {})
    client = OpenAI(
        api_key=str(api_settings.get("api_key", "EMPTY")),
        base_url=str(api_settings.get("base_url", "http://127.0.0.1:8000/v1")),
        timeout=float(api_settings.get("timeout_seconds", 300)),
    )
    concurrency = int(api_settings.get("concurrency", 1))
    warmup = int(settings.get("warmup_requests", 0))
    repeats = int(settings.get("repeats", 1))
    if concurrency < 1 or repeats < 1 or warmup < 0:
        raise ValueError("benchmark.api.concurrency, repeats, and warmup_requests are invalid")

    def call(request: dict[str, Any]) -> tuple[int, str | None]:
        try:
            completion = client.chat.completions.create(**request["body"])
            return _response_from_api(completion).token_count, None
        except Exception as exc:
            return 0, str(exc)

    for request in requests[:warmup]:
        call(request)
    measurements = []
    for repeat in range(repeats):
        before = gpu()
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            call_results = list(executor.map(call, requests))
        token_count = sum(tokens for tokens, _ in call_results)
        errors = sum(error is not None for _, error in call_results)
        elapsed = max(time.perf_counter() - started, 1e-9)
        completed = len(requests) - errors
        measurements.append(
            {
                "repeat": repeat + 1,
                "requests": len(requests),
                "completed": completed,
                "errors": errors,
                "tokens": token_count,
                "wall_seconds": elapsed,
                "requests_per_second": completed / elapsed,
                "tokens_per_second": token_count / elapsed,
                "gpu_before": before,
                "gpu_after": gpu(),
            }
        )
    return {
        "measurements": measurements,
        "base_url": str(api_settings.get("base_url", "http://127.0.0.1:8000/v1")),
        "includes_model_startup": False,
        "concurrency": concurrency,
        "warmup_requests": warmup,
    }


def benchmark_run_batch(
    config: dict[str, Any],
    requests: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    if config["model"].get("backend") != "local_vllm":
        raise RuntimeError("The run-batch benchmark requires model.backend=local_vllm")
    settings = config.get("benchmark", {})
    repeats = int(settings.get("repeats", 1))
    timeout = float(settings.get("run_batch_timeout_seconds", 86400))
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    request_path = benchmark_dir / "requests.jsonl"
    request_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests),
        encoding="utf-8",
    )
    measurements = []
    for repeat in range(repeats):
        response_path = benchmark_dir / f"responses-{repeat + 1:02d}.jsonl"
        command = batch_command_args(config, request_path, response_path)
        before = gpu()
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                cwd=config["_root"],
                env={**os.environ, **configure_vllm_environment(config["model"])},
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            response_items = [
                json.loads(line) for line in response_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            errors = sum(bool(item.get("error")) for item in response_items)
            completed = len(response_items) - errors
            token_count = sum(
                int(((item.get("response") or {}).get("body", {}).get("usage") or {}).get("completion_tokens", 0) or 0)
                for item in response_items
                if not item.get("error")
            )
            command_error = None
        except Exception as exc:
            process = None
            completed = 0
            token_count = 0
            errors = 1
            command_error = str(exc)
        elapsed = max(time.perf_counter() - started, 1e-9)
        measurements.append(
            {
                "repeat": repeat + 1,
                "requests": len(requests),
                "completed": completed,
                "errors": errors,
                "tokens": token_count,
                "wall_seconds": elapsed,
                "requests_per_second": completed / elapsed,
                "tokens_per_second": token_count / elapsed,
                "gpu_before": before,
                "gpu_after": gpu(),
                "command": shlex.join(command),
                "error": command_error,
                "stdout_tail": process.stdout[-1000:] if process else None,
                "stderr_tail": process.stderr[-1000:] if process else None,
            }
        )
    return {
        "measurements": measurements,
        "includes_model_startup": True,
        "request_file": str(request_path),
    }


def summarise_benchmark(data: dict[str, Any]) -> dict[str, Any]:
    measurements = data.get("measurements", [])
    if not measurements:
        return {}
    return {
        "mean_wall_seconds": sum(item["wall_seconds"] for item in measurements) / len(measurements),
        "mean_requests_per_second": sum(item["requests_per_second"] for item in measurements) / len(measurements),
        "mean_tokens_per_second": sum(item["tokens_per_second"] for item in measurements) / len(measurements),
        "total_errors": sum(int(item.get("errors", 0)) for item in measurements),
    }


def benchmark(
    config: dict[str, Any],
    approaches: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    allowed = {"api", "run-batch", "python"}
    settings = config.get("benchmark", {})
    selected = [str(item) for item in (approaches or settings.get("approaches", sorted(allowed)))]
    if not selected or any(item not in allowed for item in selected):
        raise ValueError("benchmark approaches must be selected from api, run-batch, and python")
    rows = benchmark_rows(config, limit)
    requests = build_requests(config, rows)
    output_dir = resolve(config, config["output"]["directory"])
    path = resolve(config, settings.get("output", output_dir / "benchmark.json"))
    result: dict[str, Any] = {
        "run_id": config["run"]["id"],
        "model": config["model"],
        "rows": len(rows),
        "variants": len(config["variants"]),
        "requests": len(requests),
        "workload_hash": hashlib.sha256(
            json.dumps(
                [variant_config_hash(config, materialize_variant(config, variant)) for variant in config["variants"]],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "approaches": {},
    }
    for approach in selected:
        if approach == "python":
            measurement = benchmark_python(config, rows)
        elif approach == "api":
            measurement = benchmark_api(config, requests)
        else:
            measurement = benchmark_run_batch(config, requests, output_dir)
        measurement["summary"] = summarise_benchmark(measurement)
        result["approaches"][approach] = measurement
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


__all__ = [
    "benchmark",
    "benchmark_api",
    "benchmark_python",
    "benchmark_rows",
    "benchmark_run_batch",
    "summarise_benchmark",
]
