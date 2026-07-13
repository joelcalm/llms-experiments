#!/usr/bin/env python3
"""Generic, YAML-configured local LLM experiment runner.

All application-specific choices live in a YAML file and Markdown/JSON assets.
This file owns the complete execution path so the CLI is easy to copy and audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import subprocess
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

MODES = {"generate", "candidate_logprobs"}
TOKEN = re.compile(r"{{\s*(text|row_id|output_schema|raw_response|validation_errors)\s*}}")


def resolve(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_root"]) / path


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    config = deepcopy(config)
    config["_root"] = str(path.parent.parent)
    validate_config(config, check_files=True)
    return config


def validate_config(config: dict[str, Any], *, check_files: bool = False) -> None:
    for key in ("run", "input", "model", "variants", "output"):
        if key not in config:
            raise ValueError(f"Missing required top-level key `{key}`")
    if not config["run"].get("id"):
        raise ValueError("run.id is required")
    source = config["input"]
    for key in ("path", "format", "id_column", "text_column"):
        if not source.get(key):
            raise ValueError(f"input.{key} is required")
    if source["format"] not in {"csv", "jsonl", "parquet"}:
        raise ValueError("input.format must be csv, jsonl, or parquet")
    if config["model"].get("backend") not in {"local_vllm", "fake"}:
        raise ValueError("model.backend must be local_vllm or fake")
    seen: set[str] = set()
    for variant in config["variants"]:
        identifier = variant.get("id")
        if not identifier or identifier in seen:
            raise ValueError("Every variant needs a unique id")
        seen.add(identifier)
        if variant.get("request_mode") not in MODES:
            raise ValueError(f"{identifier}: unsupported request_mode")
        if not variant.get("prompts"):
            raise ValueError(f"{identifier}: prompts must not be empty")
        if variant["request_mode"] == "candidate_logprobs" and not variant.get("candidates"):
            raise ValueError(f"{identifier}: candidate_logprobs requires candidates")
    sizes = config.get("batch", {}).get("candidates", [1])
    if not sizes or any(not isinstance(size, int) or size < 1 for size in sizes):
        raise ValueError("batch.candidates must contain positive integers")
    if check_files:
        paths = [config["input"]["path"]]
        for variant in config["variants"]:
            paths.extend(variant["prompts"])
            if schema := variant.get("validation", {}).get("schema"):
                paths.append(schema)
        retry = config.get("validation", {}).get("retry", {}).get("correction_prompt")
        if retry:
            paths.append(retry)
        for item in paths:
            if not resolve(config, item).is_file():
                raise ValueError(f"Configured file does not exist: {item}")


def read_rows(path: Path, data_format: str, id_column: str, text_column: str) -> list[dict[str, Any]]:
    if data_format == "parquet":
        rows = pq.read_table(path).to_pylist()
    elif data_format == "jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    for position, row in enumerate(rows):
        if id_column not in row or text_column not in row:
            raise ValueError(f"Input row {position} lacks `{id_column}` or `{text_column}`")
        row["_source_position"] = position
    return rows


def render(template: str, values: dict[str, Any]) -> str:
    return TOKEN.sub(lambda match: str(values.get(match.group(1), "")), template)


def prompt(config: dict[str, Any], paths: list[str], values: dict[str, Any]) -> str:
    return "\n\n".join(render(resolve(config, path).read_text(encoding="utf-8").strip(), values) for path in paths)


def check_schema(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    expected = schema.get("type")
    matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if expected and not matches.get(expected, True):
        errors.append(f"{path}: expected {expected}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not an allowed candidate")
    if isinstance(value, str) and schema.get("pattern") and not re.search(schema["pattern"], value):
        errors.append(f"{path}: does not match pattern")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key `{key}`")
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}: unexpected key `{key}`" for key in value if key not in properties)
        for key, child in properties.items():
            if key in value:
                check_schema(value[key], child, f"{path}.{key}", errors)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            check_schema(item, schema["items"], f"{path}[{index}]", errors)


def validate_response(raw: str, schema: dict[str, Any] | None) -> tuple[Any | None, list[str]]:
    if schema is None:
        return raw, []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"json_parse_error: {exc.msg}"]
    errors: list[str] = []
    check_schema(parsed, schema, "$", errors)
    return parsed, errors


class Events:
    def __init__(self, log_path: Path, event_path: Path, level: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = event_path
        self.logger = logging.getLogger(f"experiment-cli.{event_path}")
        self.logger.handlers.clear()
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)

    def emit(self, event: str, **payload: Any) -> None:
        record = {"timestamp": time.time(), "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.logger.info("%s %s", event, json.dumps(payload, sort_keys=True))


def gpu() -> dict[str, Any]:
    try:
        line = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=5,
            )
            .strip()
            .splitlines()[0]
        )
        used, total, utilisation = [int(value.strip()) for value in line.split(",")]
        snapshot: dict[str, Any] = {
            "available": True,
            "memory_used_mib": used,
            "memory_total_mib": total,
            "utilization_percent": utilisation,
        }
        try:
            import torch

            if torch.cuda.is_available():
                snapshot.update(
                    {
                        "cuda_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
                        "cuda_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2),
                        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                        "cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
                    }
                )
        except Exception:
            pass
        return snapshot
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError, ValueError) as exc:
        return {"available": False, "error": str(exc)}


def sync_cuda(enabled: bool) -> None:
    if enabled:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


class BackendFailure(RuntimeError):
    pass


@dataclass
class Response:
    raw: str
    token_count: int
    candidate_logprobs: dict[str, float] | None = None


class FakeBackend:
    def generate(self, prompts: list[str], variant: dict[str, Any]) -> list[Response]:
        if variant["request_mode"] == "candidate_logprobs":
            scores = {candidate: -float(index) for index, candidate in enumerate(variant["candidates"])}
            return [Response(json.dumps({"candidates": scores}), 1, scores) for _ in prompts]
        return [Response(json.dumps(variant.get("fake_response", {"label": "alpha"})), 1) for _ in prompts]

    def close(self) -> None:
        return None


class VLLMBackend:
    def __init__(self, model: dict[str, Any]) -> None:
        if not gpu().get("available"):
            raise RuntimeError("GPU preflight failed: nvidia-smi cannot communicate with an NVIDIA driver.")
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("local_vllm requires vLLM; install the uv project dependencies first.") from exc
        self.params = SamplingParams
        self.llm = LLM(
            model=model["name"],
            gpu_memory_utilization=model.get("gpu_memory_utilization", 0.9),
            max_model_len=model.get("max_model_len", 2048),
            max_num_seqs=model.get("max_num_seqs", 128),
            enable_prefix_caching=model.get("enable_prefix_caching", True),
        )

    def generate(self, prompts: list[str], variant: dict[str, Any]) -> list[Response]:
        if variant["request_mode"] == "candidate_logprobs":
            prompts = [item + "\n\nAnswer with exactly one candidate token:" for item in prompts]
            params = self.params(temperature=0, max_tokens=1, logprobs=max(20, len(variant["candidates"]) + 5))
        else:
            kwargs: dict[str, Any] = {"temperature": 0, "max_tokens": variant.get("max_tokens", 128)}
            if schema := variant.get("_schema"):
                from vllm.sampling_params import StructuredOutputsParams

                kwargs["structured_outputs"] = StructuredOutputsParams(
                    json=schema, disable_any_whitespace=True, disable_additional_properties=True
                )
            params = self.params(**kwargs)
        try:
            outputs = self.llm.generate(prompts, params, use_tqdm=False)
        except Exception as exc:
            if any(word in str(exc).lower() for word in ("out of memory", "oom", "context length", "max model len")):
                raise BackendFailure(str(exc)) from exc
            raise
        result: list[Response] = []
        for output in outputs:
            generated = output.outputs[0]
            if variant["request_mode"] == "candidate_logprobs":
                observed = {
                    str(getattr(logprob, "decoded_token", token)).strip(): float(getattr(logprob, "logprob", logprob))
                    for token, logprob in ((generated.logprobs or [{}])[0] or {}).items()
                }
                scores = {
                    candidate: observed.get(str(candidate).strip(), -float("inf"))
                    for candidate in variant["candidates"]
                }
                result.append(Response(json.dumps({"candidates": scores}), len(generated.token_ids), scores))
            else:
                result.append(Response(generated.text, len(generated.token_ids)))
        return result

    def close(self) -> None:
        del self.llm


def tune(
    backend: Any, variant: dict[str, Any], prompts: list[str], batch: dict[str, Any], events: Events, sync: bool
) -> tuple[int, list[dict[str, Any]]]:
    if batch.get("mode", "auto") == "fixed":
        return int(batch.get("size", batch.get("candidates", [1])[0])), []
    attempts: list[dict[str, Any]] = []
    safe: list[tuple[float, int]] = []
    candidates = sorted(set(int(item) for item in batch.get("candidates", [1])))
    warmup = prompts[: int(batch.get("warmup_rows", 64))] or prompts[:1]
    for size in candidates:
        sync_cuda(sync)
        started = time.perf_counter()
        try:
            responses = backend.generate(warmup[:size], variant)
            sync_cuda(sync)
            elapsed = max(time.perf_counter() - started, 1e-9)
            record = {
                "candidate": size,
                "accepted": True,
                "rows_per_second": min(size, len(warmup)) / elapsed,
                "tokens_per_second": sum(response.token_count for response in responses) / elapsed,
                "latency_seconds": elapsed,
                "gpu": gpu(),
            }
            safe.append((record["rows_per_second"], size))
        except BackendFailure as exc:
            record = {"candidate": size, "accepted": False, "error": str(exc), "gpu": gpu()}
            events.emit("batch_candidate_rejected", variant=variant["id"], **record)
        attempts.append(record)
        events.emit("batch_candidate", variant=variant["id"], **record)
    if not safe:
        raise RuntimeError(f"No safe batch size for {variant['id']}")
    selected = max(safe)[1]
    events.emit("batch_selected", variant=variant["id"], batch_size=selected, candidates=attempts)
    return selected, attempts


def write_results(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), run_dir / "results.parquet", compression="zstd")
    for identifier in sorted({str(row["variant_id"]) for row in rows}):
        pq.write_table(
            pa.Table.from_pylist([row for row in rows if row["variant_id"] == identifier]),
            run_dir / f"{identifier}.parquet",
            compression="zstd",
        )


def serialise(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def run(config: dict[str, Any], backend: Any | None = None) -> dict[str, Any]:
    run_id = config["run"]["id"]
    run_dir = resolve(config, config["output"]["directory"])
    logging_config = config.get("logging", {})
    events = Events(
        resolve(config, logging_config.get("file", f"logs/{run_id}.log")),
        resolve(config, logging_config.get("events", f"logs/{run_id}.events.jsonl")),
        logging_config.get("level", "INFO"),
    )
    rows = read_rows(
        resolve(config, config["input"]["path"]),
        config["input"]["format"],
        config["input"]["id_column"],
        config["input"]["text_column"],
    )
    result_path = run_dir / "results.parquet"
    existing = pq.read_table(result_path).to_pylist() if result_path.exists() else []
    complete = {(str(row["variant_id"]), str(row["input_row_id"])) for row in existing}
    previous = (
        json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if (run_dir / "manifest.json").exists()
        else {}
    )
    selected: dict[str, Any] = dict(previous.get("variants", {}))
    created = False
    correction_path = config.get("validation", {}).get("retry", {}).get("correction_prompt")
    correction = resolve(config, correction_path).read_text(encoding="utf-8") if correction_path else None
    try:
        for variant in config["variants"]:
            pending = [row for row in rows if (variant["id"], str(row[config["input"]["id_column"]])) not in complete]
            if not pending:
                events.emit("variant_resumed", variant=variant["id"], skipped=len(rows))
                continue
            if backend is None:
                backend = FakeBackend() if config["model"]["backend"] == "fake" else VLLMBackend(config["model"])
                created = True
            schema_path = variant.get("validation", {}).get("schema")
            schema = json.loads(resolve(config, schema_path).read_text(encoding="utf-8")) if schema_path else None
            request_variant = {**variant, "_schema": schema}
            values = lambda row: {
                "text": row[config["input"]["text_column"]],
                "row_id": row[config["input"]["id_column"]],
                "output_schema": json.dumps(schema or {}, sort_keys=True),
            }
            prompts = [prompt(config, variant["prompts"], values(row)) for row in pending]
            batch = dict(config.get("batch", {}))
            maximum = int(config["model"].get("max_num_seqs", max(batch.get("candidates", [1]))))
            batch["candidates"] = [item for item in batch.get("candidates", [1]) if int(item) <= maximum] or [maximum]
            size, attempts = tune(
                backend, request_variant, prompts, batch, events, bool(config["model"].get("synchronize_cuda", False))
            )
            selected[variant["id"]] = {"selected_batch_size": size, "tuning": attempts, "pending_rows": len(pending)}
            events.emit("variant_started", variant=variant["id"], rows=len(pending), batch_size=size)
            for offset in range(0, len(pending), size):
                chunk, chunk_prompts = pending[offset : offset + size], prompts[offset : offset + size]
                sync_cuda(bool(config["model"].get("synchronize_cuda", False)))
                started = time.perf_counter()
                responses = backend.generate(chunk_prompts, request_variant)
                sync_cuda(bool(config["model"].get("synchronize_cuda", False)))
                elapsed = max(time.perf_counter() - started, 1e-9)
                for row, current_prompt, response in zip(chunk, chunk_prompts, responses):
                    parsed, errors = validate_response(response.raw, schema)
                    raw, count = response.raw, 1
                    if request_variant["request_mode"] == "candidate_logprobs":
                        parsed = {"candidates": response.candidate_logprobs or {}}
                    if errors and correction and config.get("validation", {}).get("retry", {}).get("enabled"):
                        events.emit(
                            "retry_started", variant=variant["id"], input_row_id=str(row[config["input"]["id_column"]])
                        )
                        for _ in range(int(config["validation"]["retry"].get("max_attempts", 0))):
                            retry_prompt = render(
                                correction, {**values(row), "raw_response": raw, "validation_errors": "; ".join(errors)}
                            )
                            raw = backend.generate([retry_prompt], request_variant)[0].raw
                            parsed, errors = validate_response(raw, schema)
                            count += 1
                            if not errors:
                                break
                        events.emit(
                            "retry_completed",
                            variant=variant["id"],
                            input_row_id=str(row[config["input"]["id_column"]]),
                            attempts=count,
                            validation_status="valid" if not errors else "invalid",
                        )
                    existing.append(
                        {
                            "run_id": run_id,
                            "variant_id": variant["id"],
                            "input_row_id": str(row[config["input"]["id_column"]]),
                            "source_position": row["_source_position"],
                            "input_text": str(row[config["input"]["text_column"]])
                            if config["output"].get("include_text")
                            else None,
                            "prompt_hash": hashlib.sha256(current_prompt.encode()).hexdigest(),
                            "config_hash": hashlib.sha256(json.dumps(variant, sort_keys=True).encode()).hexdigest(),
                            "attempt_count": count,
                            "raw_response": raw if config["output"].get("include_raw_response", True) else None,
                            "parsed_output": serialise(parsed),
                            "validation_status": "valid" if not errors else "invalid",
                            "validation_errors": serialise(errors),
                            "final_status": "completed" if not errors else "failed_validation",
                            "batch_size": size,
                            "latency_seconds": elapsed / len(chunk),
                            "rows_per_second": len(chunk) / elapsed,
                            "token_count": response.token_count,
                            "gpu_snapshot": serialise(gpu()),
                            "candidate_logprobs": serialise(response.candidate_logprobs),
                        }
                    )
                events.emit(
                    "batch_completed",
                    variant=variant["id"],
                    rows=len(chunk),
                    latency_seconds=elapsed,
                    rows_per_second=len(chunk) / elapsed,
                    gpu=gpu(),
                )
            write_results(run_dir, existing)
            events.emit("variant_completed", variant=variant["id"], rows=len(pending))
    finally:
        if created and backend is not None:
            backend.close()
    effective = {key: value for key, value in config.items() if not key.startswith("_")}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "effective_config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "input_rows": len(rows),
        "result_rows": len(existing),
        "model": config["model"],
        "variants": selected,
        "gpu_preflight": gpu(),
        "resume_skipped_rows": len(complete),
        "event_log": str(events.path),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    events.emit("run_completed", result_rows=len(existing), gpu=gpu())
    return manifest


def prepare(config: dict[str, Any]) -> Path:
    run_dir = resolve(config, config["output"]["directory"])
    rows = read_rows(
        resolve(config, config["input"]["path"]),
        config["input"]["format"],
        config["input"]["id_column"],
        config["input"]["text_column"],
    )
    path = run_dir / "requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for variant in config["variants"]:
            schema_path = variant.get("validation", {}).get("schema")
            schema = json.loads(resolve(config, schema_path).read_text(encoding="utf-8")) if schema_path else None
            for row in rows:
                values = {
                    "text": row[config["input"]["text_column"]],
                    "row_id": row[config["input"]["id_column"]],
                    "output_schema": json.dumps(schema or {}),
                }
                body: dict[str, Any] = {
                    "model": config["model"]["name"],
                    "messages": [{"role": "user", "content": prompt(config, variant["prompts"], values)}],
                    "temperature": 0,
                }
                if variant["request_mode"] == "candidate_logprobs":
                    body.update(
                        {
                            "max_completion_tokens": 1,
                            "logprobs": True,
                            "top_logprobs": max(20, len(variant["candidates"]) + 5),
                        }
                    )
                else:
                    body["max_completion_tokens"] = variant.get("max_tokens", 128)
                    if schema:
                        body["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {"name": variant["id"], "schema": schema, "strict": True},
                        }
                handle.write(
                    json.dumps(
                        {
                            "custom_id": f"{variant['id']}:{row[config['input']['id_column']]}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": body,
                        }
                    )
                    + "\n"
                )
    return path


def _batch_text(item: dict[str, Any]) -> tuple[str | None, str | None, dict[str, float] | None]:
    response = item.get("response") or {}
    if item.get("error") or response.get("status_code", 200) != 200:
        return None, str(item.get("error") or response.get("status_code", "batch_response_error")), None
    choice = ((response.get("body") or {}).get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    if content is None:
        return None, "missing_chat_completion_content", None
    observed: dict[str, float] = {}
    for token in (choice.get("logprobs") or {}).get("content") or []:
        for candidate in token.get("top_logprobs") or []:
            observed[str(candidate.get("token", "")).strip()] = float(candidate.get("logprob", -float("inf")))
    return str(content), None, observed or None


def parse_batch(config: dict[str, Any], response_path: str | Path) -> dict[str, Any]:
    source = Path(response_path)
    if source.is_dir():
        source = source / "responses.jsonl"
    responses = {
        str(item["custom_id"]): item
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in [json.loads(line)]
    }
    run_id = config["run"]["id"]
    run_dir = resolve(config, config["output"]["directory"])
    logging_config = config.get("logging", {})
    events = Events(
        resolve(config, logging_config.get("file", f"logs/{run_id}.log")),
        resolve(config, logging_config.get("events", f"logs/{run_id}.events.jsonl")),
        logging_config.get("level", "INFO"),
    )
    rows = read_rows(
        resolve(config, config["input"]["path"]),
        config["input"]["format"],
        config["input"]["id_column"],
        config["input"]["text_column"],
    )
    result_path = run_dir / "results.parquet"
    saved = pq.read_table(result_path).to_pylist() if result_path.exists() else []
    complete = {(str(row["variant_id"]), str(row["input_row_id"])) for row in saved}
    for variant in config["variants"]:
        schema_path = variant.get("validation", {}).get("schema")
        schema = json.loads(resolve(config, schema_path).read_text(encoding="utf-8")) if schema_path else None
        for row in rows:
            row_id = str(row[config["input"]["id_column"]])
            if (variant["id"], row_id) in complete:
                continue
            values = {
                "text": row[config["input"]["text_column"]],
                "row_id": row_id,
                "output_schema": json.dumps(schema or {}, sort_keys=True),
            }
            current_prompt = prompt(config, variant["prompts"], values)
            raw, batch_error, observed = _batch_text(responses.get(f"{variant['id']}:{row_id}", {}))
            parsed, errors = (
                validate_response(raw or "", schema)
                if raw is not None
                else (None, [batch_error or "missing_batch_response"])
            )
            scores: dict[str, float] | None = None
            if variant["request_mode"] == "candidate_logprobs" and observed is not None:
                scores = {
                    candidate: observed.get(str(candidate).strip(), -float("inf"))
                    for candidate in variant["candidates"]
                }
                parsed, errors = {"candidates": scores}, []
            saved.append(
                {
                    "run_id": run_id,
                    "variant_id": variant["id"],
                    "input_row_id": row_id,
                    "source_position": row["_source_position"],
                    "input_text": str(row[config["input"]["text_column"]])
                    if config["output"].get("include_text")
                    else None,
                    "prompt_hash": hashlib.sha256(current_prompt.encode()).hexdigest(),
                    "config_hash": hashlib.sha256(json.dumps(variant, sort_keys=True).encode()).hexdigest(),
                    "attempt_count": 1,
                    "raw_response": raw if config["output"].get("include_raw_response", True) else None,
                    "parsed_output": serialise(parsed),
                    "validation_status": "valid" if not errors else "invalid",
                    "validation_errors": serialise(errors),
                    "final_status": "completed" if not errors else "failed_validation",
                    "batch_size": None,
                    "latency_seconds": None,
                    "rows_per_second": None,
                    "token_count": None,
                    "gpu_snapshot": None,
                    "candidate_logprobs": serialise(scores),
                }
            )
    write_results(run_dir, saved)
    manifest = {
        "run_id": run_id,
        "input_rows": len(rows),
        "result_rows": len(saved),
        "model": config["model"],
        "variants": {variant["id"]: {"external_batch": True} for variant in config["variants"]},
        "batch_response_path": str(source),
        "resume_skipped_rows": len(complete),
        "event_log": str(events.path),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    events.emit("batch_parse_completed", result_rows=len(saved), response_path=str(source))
    return manifest


def batch_command(config: dict[str, Any]) -> str:
    run_dir = resolve(config, config["output"]["directory"])
    model = config["model"]
    command = [
        "uv",
        "run",
        "vllm",
        "run-batch",
        "-i",
        str(run_dir / "requests.jsonl"),
        "-o",
        str(run_dir / "responses.jsonl"),
        "--model",
        model["name"],
        "--gpu-memory-utilization",
        str(model.get("gpu_memory_utilization", 0.9)),
        "--max-model-len",
        str(model.get("max_model_len", 2048)),
        "--max-num-seqs",
        str(model.get("max_num_seqs", 128)),
    ]
    if model.get("enable_prefix_caching", True):
        command.append("--enable-prefix-caching")
    return " ".join(command)


def self_test(config: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        test = deepcopy(config)
        test["model"] = {"backend": "fake", "name": "fake", "max_num_seqs": 8}
        test["batch"] = {"mode": "auto", "candidates": [1, 2, 4, 8], "warmup_rows": 4}
        test["output"]["directory"] = str(Path(directory) / "output")
        test["logging"] = {"file": str(Path(directory) / "run.log"), "events": str(Path(directory) / "events.jsonl")}
        payloads = {
            "single_label_json": {"label": "alpha"},
            "multi_label_json": {"labels": ["alpha"]},
            "ordinal_score_json": {"score": 3},
        }
        for variant in test["variants"]:
            if variant["id"] in payloads:
                variant["fake_response"] = payloads[variant["id"]]
        first = run(test, FakeBackend())
        second = run(test, FakeBackend())
        if first["result_rows"] != len(test["variants"]) * 128 or second["resume_skipped_rows"] != first["result_rows"]:
            raise RuntimeError("self-test failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "prepare", "batch-command", "self-test"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
    parse = commands.add_parser("parse")
    parse.add_argument("--config", required=True)
    parse.add_argument("--responses", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "validate":
        print(f"Valid configuration: {config['run']['id']}")
    elif args.command == "prepare":
        print(prepare(config))
    elif args.command == "batch-command":
        print(batch_command(config))
    elif args.command == "parse":
        print(json.dumps(parse_batch(config, args.responses), indent=2))
    elif args.command == "self-test":
        self_test(config)
        print("Self-test passed")
    else:
        print(json.dumps(run(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
