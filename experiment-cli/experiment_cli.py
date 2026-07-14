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
import os
import re
import shlex
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

MODES = {"generate", "candidate_logprobs"}
TOKEN = re.compile(r"{{\s*(text|row_id|dataset_id|output_schema|raw_response|validation_errors|candidates)\s*}}")


def resolve(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_root"]) / path


def _set_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current: Any = config
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            if part not in current or current[part] is None:
                current[part] = {}
            current = current[part]
    last = parts[-1]
    if isinstance(current, list):
        current[int(last)] = value
    else:
        current[last] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    config = deepcopy(config)
    config["_root"] = str(path.parent.parent if path.parent.name in {"config", "experiments"} else path.parent)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must use KEY=VALUE syntax: {item}")
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError(f"Override key is empty: {item}")
        _set_path(config, key, yaml.safe_load(raw_value))
    validate_config(config, check_files=True)
    return config


def validate_config(config: dict[str, Any], *, check_files: bool = False) -> None:
    for key in ("run", "model", "variants", "output"):
        if key not in config:
            raise ValueError(f"Missing required top-level key `{key}`")
    if not config["run"].get("id"):
        raise ValueError("run.id is required")
    if "datasets" in config:
        if not isinstance(config["datasets"], list) or not config["datasets"]:
            raise ValueError("datasets must be a non-empty list")
        sources = []
        dataset_ids: set[str] = set()
        for dataset in config["datasets"]:
            identifier = str(dataset.get("id", ""))
            if not identifier or identifier in dataset_ids:
                raise ValueError("Every dataset needs a unique id")
            dataset_ids.add(identifier)
            source = dataset.get("input", dataset)
            _validate_source(source, f"datasets[{identifier}].input")
            sources.append(source)
    else:
        if "input" not in config:
            raise ValueError("Missing required top-level key `input` (or `datasets`)")
        _validate_source(config["input"], "input")
        sources = [config["input"]]
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
    benchmark_config = config.get("benchmark", {})
    approaches = benchmark_config.get("approaches", ["api", "run-batch", "python"])
    if not approaches or any(item not in {"api", "run-batch", "python"} for item in approaches):
        raise ValueError("benchmark.approaches must contain api, run-batch, or python")
    if int(benchmark_config.get("rows", 1)) < 1:
        raise ValueError("benchmark.rows must be positive")
    if check_files:
        paths = []
        for source in sources:
            paths.append(source["path"])
            if source.get("format") == "paired_tsv":
                paths.append(source["labels_path"])
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


def _validate_source(source: dict[str, Any], name: str) -> None:
    for key in ("path", "format", "id_column", "text_column"):
        if not source.get(key):
            raise ValueError(f"{name}.{key} is required")
    if source["format"] not in {"csv", "tsv", "jsonl", "parquet", "nested_json", "paired_tsv"}:
        raise ValueError(f"{name}.format is unsupported")
    if source["format"] == "paired_tsv" and not source.get("labels_path"):
        raise ValueError(f"{name}.labels_path is required for paired_tsv")


def _split_labels(value: Any) -> list[str]:
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


def read_rows(
    path: Path,
    data_format: str,
    id_column: str,
    text_column: str,
    source: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    source = source or {
        "path": str(path),
        "format": data_format,
        "id_column": id_column,
        "text_column": text_column,
    }
    if data_format == "parquet":
        rows = pq.read_table(path).to_pylist()
    elif data_format == "jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif data_format in {"csv", "tsv"}:
        delimiter = source.get("delimiter", "\t" if data_format == "tsv" else ",")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = []
            where = source.get("where", {})
            limit = int(source["limit"]) if source.get("limit") else None
            for row in csv.DictReader(handle, delimiter=delimiter):
                if where and any(str(row.get(key)) != str(value) for key, value in where.items()):
                    continue
                rows.append(dict(row))
                if limit and len(rows) >= limit:
                    break
    elif data_format == "nested_json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        records_key = source.get("records_key", "Tweets")
        labels_key = source.get("labels_column", "annotations")
        label_value_key = source.get("label_value_key", "annotation")
        for parent in payload:
            for record in parent.get(records_key, []):
                annotations = record.get(labels_key, [])
                labels = []
                for annotation in annotations:
                    value = annotation.get(label_value_key) if isinstance(annotation, dict) else annotation
                    labels.extend(_split_labels(value))
                rows.append({**record, "_gold_labels": sorted(set(labels))})
    elif data_format == "paired_tsv":
        with path.open(encoding="utf-8", newline="") as handle:
            arguments = {row[id_column]: dict(row) for row in csv.DictReader(handle, delimiter="\t")}
        label_path = Path(source["labels_path"])
        with label_path.open(encoding="utf-8", newline="") as handle:
            labels = {row[id_column]: dict(row) for row in csv.DictReader(handle, delimiter="\t")}
        columns = source.get("label_columns")
        rows = []
        for row_id, argument in arguments.items():
            if row_id not in labels:
                continue
            label_row = labels[row_id]
            selected = columns or [key for key in label_row if key != id_column]
            rows.append(
                {
                    id_column: row_id,
                    text_column: argument.get(text_column, ""),
                    "_gold_labels": [
                        key for key in selected if str(label_row.get(key, "0")).strip() in {"1", "1.0", "true", "True"}
                    ],
                }
            )
    else:
        raise ValueError(f"Unsupported input format: {data_format}")
    where = source.get("where", {})
    if where and data_format not in {"csv", "tsv"}:
        rows = [row for row in rows if all(str(row.get(key)) == str(value) for key, value in where.items())]
    for position, row in enumerate(rows):
        if id_column not in row or text_column not in row:
            raise ValueError(f"Input row {position} lacks `{id_column}` or `{text_column}`")
        if "_gold_labels" not in row and source.get("labels_column"):
            row["_gold_labels"] = _split_labels(row.get(source["labels_column"]))
        row["_source_position"] = position
    return rows


def dataset_entries(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if "datasets" not in config:
        return [(str(config.get("run", {}).get("dataset_id", "default")), config["input"])]
    return [(str(item["id"]), item.get("input", item)) for item in config["datasets"]]


def select_dataset(config: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    for identifier, source in dataset_entries(config):
        if identifier == dataset_id:
            selected = deepcopy(config)
            selected.pop("datasets", None)
            selected["input"] = source
            selected["run"] = {**config["run"], "id": f"{config['run']['id']}__{identifier}", "dataset_id": identifier}
            base_output = resolve(config, config["output"]["directory"])
            selected["output"] = {**config["output"], "directory": str(base_output / f"dataset={identifier}")}
            return selected
    raise ValueError(f"Unknown dataset id: {dataset_id}")


def rows_for_source(config: dict[str, Any], source: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_rows(
        resolve(config, source["path"]),
        source["format"],
        source["id_column"],
        source["text_column"],
        source,
    )
    configured_limit = source.get("limit")
    effective_limit = limit if limit is not None else configured_limit
    return rows[: int(effective_limit)] if effective_limit else rows


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
        # vLLM V1 starts an EngineCore process.  CUDA must never be inherited
        # through fork after the telemetry preflight has touched torch.
        import multiprocessing as mp

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("local_vllm requires vLLM; install the uv project dependencies first.") from exc
        self.params = SamplingParams
        llm_kwargs: dict[str, Any] = {
            "model": model["name"],
            "gpu_memory_utilization": model.get("gpu_memory_utilization", 0.9),
            "max_model_len": model.get("max_model_len", 2048),
            "max_num_seqs": model.get("max_num_seqs", 128),
            "enable_prefix_caching": model.get("enable_prefix_caching", True),
        }
        if "language_model_only" in model:
            llm_kwargs["language_model_only"] = bool(model["language_model_only"])
        if "limit_mm_per_prompt" in model:
            llm_kwargs["limit_mm_per_prompt"] = model["limit_mm_per_prompt"]
        self.llm = LLM(
            **llm_kwargs,
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


def run(config: dict[str, Any], backend: Any | None = None, row_limit: int | None = None) -> dict[str, Any]:
    run_id = config["run"]["id"]
    run_dir = resolve(config, config["output"]["directory"])
    logging_config = config.get("logging", {})
    events = Events(
        resolve(config, logging_config.get("file", f"logs/{run_id}.log")),
        resolve(config, logging_config.get("events", f"logs/{run_id}.events.jsonl")),
        logging_config.get("level", "INFO"),
    )
    rows = rows_for_source(config, config["input"], row_limit)
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
                "dataset_id": config["run"].get("dataset_id", "default"),
                "candidates": ", ".join(str(item) for item in variant.get("candidates", [])),
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
                            "dataset_id": config["run"].get("dataset_id", "default"),
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
        "dataset_id": config["run"].get("dataset_id", "default"),
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


def run_matrix(
    config: dict[str, Any], row_limit: int | None = None, selected: list[str] | None = None
) -> dict[str, Any]:
    """Run every configured dataset with one shared backend/model process."""
    if "datasets" not in config:
        return {"datasets": [run(config, row_limit=row_limit)]}
    base_output = resolve(config, config["output"]["directory"])
    entries = dataset_entries(config)
    if selected:
        wanted = set(selected)
        entries = [(identifier, source) for identifier, source in entries if identifier in wanted]
        missing = wanted - {identifier for identifier, _ in entries}
        if missing:
            raise ValueError(f"Unknown dataset id(s): {', '.join(sorted(missing))}")
    shared = FakeBackend() if config["model"].get("backend") == "fake" else VLLMBackend(config["model"])
    manifests: list[dict[str, Any]] = []
    try:
        for dataset_id, source in entries:
            dataset_config = deepcopy(config)
            dataset_config.pop("datasets", None)
            dataset_config["input"] = source
            dataset_config["run"] = {
                **config["run"],
                "id": f"{config['run']['id']}__{dataset_id}",
                "dataset_id": dataset_id,
            }
            dataset_config["output"] = {
                **config["output"],
                "directory": str(base_output / f"dataset={dataset_id}"),
            }
            logging_config = dict(config.get("logging", {}))
            if logging_config.get("file"):
                logging_config["file"] = str(resolve(config, logging_config["file"]).with_name(f"{dataset_id}.log"))
            if logging_config.get("events"):
                logging_config["events"] = str(
                    resolve(config, logging_config["events"]).with_name(f"{dataset_id}.events.jsonl")
                )
            dataset_config["logging"] = logging_config
            manifests.append(run(dataset_config, shared, row_limit=row_limit))
    finally:
        shared.close()
    summary = {
        "run_id": config["run"]["id"],
        "model": config["model"],
        "datasets": manifests,
        "result_rows": sum(int(item.get("result_rows", 0)) for item in manifests),
    }
    base_output.mkdir(parents=True, exist_ok=True)
    (base_output / "matrix_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def config_overrides(args: argparse.Namespace) -> list[str]:
    overrides = list(args.overrides or [])
    shortcuts = {
        "run_id": "run.id",
        "model": "model.name",
        "backend": "model.backend",
        "output": "output.directory",
    }
    for argument, key in shortcuts.items():
        value = getattr(args, argument, None)
        if value is not None:
            overrides.append(f"{key}={value}")
    return overrides


def prepare(config: dict[str, Any]) -> Path:
    run_dir = resolve(config, config["output"]["directory"])
    rows = rows_for_source(config, config["input"])
    path = run_dir / "requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for request in build_requests(config, rows):
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
    return path


def prepare_matrix(config: dict[str, Any], selected: list[str] | None = None) -> list[Path]:
    """Prepare one independent vLLM JSONL request file per selected dataset."""
    if "datasets" not in config:
        return [prepare(config)]
    entries = dataset_entries(config)
    if selected:
        wanted = set(selected)
        entries = [(identifier, source) for identifier, source in entries if identifier in wanted]
        missing = wanted - {identifier for identifier, _ in entries}
        if missing:
            raise ValueError(f"Unknown dataset id(s): {', '.join(sorted(missing))}")
    base_output = resolve(config, config["output"]["directory"])
    paths: list[Path] = []
    for dataset_id, source in entries:
        dataset_config = deepcopy(config)
        dataset_config.pop("datasets", None)
        dataset_config["input"] = source
        dataset_config["run"] = {
            **config["run"],
            "id": f"{config['run']['id']}__{dataset_id}",
            "dataset_id": dataset_id,
        }
        dataset_config["output"] = {**config["output"], "directory": str(base_output / f"dataset={dataset_id}")}
        paths.append(prepare(dataset_config))
    return paths


def variant_schema(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any] | None:
    schema_path = variant.get("validation", {}).get("schema")
    return json.loads(resolve(config, schema_path).read_text(encoding="utf-8")) if schema_path else None


def request_for_row(
    config: dict[str, Any], variant: dict[str, Any], row: dict[str, Any], schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    if schema is None and variant.get("validation", {}).get("schema"):
        schema = variant_schema(config, variant)
    values = {
        "text": row[config["input"]["text_column"]],
        "row_id": row[config["input"]["id_column"]],
        "dataset_id": config["run"].get("dataset_id", "default"),
        "candidates": ", ".join(str(item) for item in variant.get("candidates", [])),
        "output_schema": json.dumps(schema or {}, sort_keys=True),
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
    return {
        "custom_id": f"{variant['id']}:{row[config['input']['id_column']]}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def build_requests(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for variant in config["variants"]:
        schema = variant_schema(config, variant)
        requests.extend(request_for_row(config, variant, row, schema) for row in rows)
    return requests


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


def batch_command_args(
    config: dict[str, Any], input_path: str | Path | None = None, output_path: str | Path | None = None
) -> list[str]:
    run_dir = resolve(config, config["output"]["directory"])
    model = config["model"]
    command = [
        "uv",
        "run",
        "vllm",
        "run-batch",
        "-i",
        str(input_path or run_dir / "requests.jsonl"),
        "-o",
        str(output_path or run_dir / "responses.jsonl"),
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
    if model.get("language_model_only", False):
        command.append("--language-model-only")
    return command


def batch_command(config: dict[str, Any]) -> str:
    return shlex.join(batch_command_args(config))


def response_from_api(completion: Any, variant: dict[str, Any]) -> Response:
    choice = completion.choices[0]
    raw = str(getattr(choice.message, "content", None) or "")
    usage = getattr(completion, "usage", None)
    token_count = int(getattr(usage, "completion_tokens", 0) or 0)
    observed: dict[str, float] = {}
    logprobs = getattr(choice, "logprobs", None)
    for token in getattr(logprobs, "content", None) or []:
        for candidate in getattr(token, "top_logprobs", None) or []:
            observed[str(getattr(candidate, "token", "")).strip()] = float(getattr(candidate, "logprob", -float("inf")))
    if variant["request_mode"] == "candidate_logprobs":
        scores = {candidate: observed.get(str(candidate).strip(), -float("inf")) for candidate in variant["candidates"]}
        return Response(json.dumps({"candidates": scores}), token_count, scores)
    return Response(raw, token_count)


def benchmark_rows(config: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    if "datasets" in config:
        raise ValueError("benchmark currently accepts one input; benchmark each dataset separately")
    rows = rows_for_source(config, config["input"], limit)
    requested = limit if limit is not None else config.get("benchmark", {}).get("rows", len(rows))
    requested = int(requested)
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
    backend = FakeBackend() if config["model"].get("backend") == "fake" else VLLMBackend(config["model"])
    load_seconds = time.perf_counter() - load_started
    entries: list[tuple[dict[str, Any], list[str]]] = []
    for variant in config["variants"]:
        schema = variant_schema(config, variant)
        request_variant = {**variant, "_schema": schema}
        prompts = [request_for_row(config, variant, row, schema)["body"]["messages"][0]["content"] for row in rows]
        entries.append((request_variant, prompts))
    try:
        if warmup:
            for variant, prompts in entries:
                backend.generate(prompts[:warmup], variant)
        measurements: list[dict[str, Any]] = []
        for repeat in range(repeats):
            before = gpu()
            started = time.perf_counter()
            token_count = 0
            completed = 0
            for variant, prompts in entries:
                for offset in range(0, len(prompts), batch_size):
                    responses = backend.generate(prompts[offset : offset + batch_size], variant)
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


def benchmark_api(config: dict[str, Any], requests: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The API benchmark requires the `openai` package.") from exc
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
    variants = {variant["id"]: variant for variant in config["variants"]}

    def call(request: dict[str, Any]) -> tuple[int, str | None]:
        try:
            completion = client.chat.completions.create(**request["body"])
            response = response_from_api(completion, variants[request["custom_id"].split(":", 1)[0]])
            return response.token_count, None
        except Exception as exc:
            return 0, str(exc)

    for request in requests[:warmup]:
        call(request)
    measurements: list[dict[str, Any]] = []
    for repeat in range(repeats):
        before = gpu()
        started = time.perf_counter()
        token_count = 0
        errors = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(call, requests))
        for tokens, error in results:
            token_count += tokens
            errors += int(error is not None)
        elapsed = max(time.perf_counter() - started, 1e-9)
        measurements.append(
            {
                "repeat": repeat + 1,
                "requests": len(requests),
                "completed": len(requests) - errors,
                "errors": errors,
                "tokens": token_count,
                "wall_seconds": elapsed,
                "requests_per_second": len(requests) / elapsed,
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


def benchmark_run_batch(config: dict[str, Any], requests: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    if config["model"].get("backend") != "local_vllm":
        raise RuntimeError("The run-batch benchmark requires model.backend=local_vllm")
    settings = config.get("benchmark", {})
    repeats = int(settings.get("repeats", 1))
    timeout = float(settings.get("run_batch_timeout_seconds", 86400))
    if repeats < 1:
        raise ValueError("benchmark.repeats must be positive")
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    request_path = benchmark_dir / "requests.jsonl"
    request_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests), encoding="utf-8")
    measurements: list[dict[str, Any]] = []
    for repeat in range(repeats):
        response_path = benchmark_dir / f"responses-{repeat + 1:02d}.jsonl"
        command = batch_command_args(config, request_path, response_path)
        before = gpu()
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=config["_root"],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            errors = 0
            response_count = 0
            token_count = 0
            for line in response_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                response_count += 1
                item = json.loads(line)
                errors += int(bool(item.get("error")))
                usage = (item.get("response") or {}).get("body", {}).get("usage") or {}
                token_count += int(usage.get("completion_tokens", 0) or 0)
            command_error = None
        except Exception as exc:
            completed = None
            response_count = 0
            token_count = 0
            errors = 1
            command_error = str(exc)
        elapsed = max(time.perf_counter() - started, 1e-9)
        measurements.append(
            {
                "repeat": repeat + 1,
                "requests": len(requests),
                "completed": response_count,
                "errors": errors,
                "tokens": token_count,
                "wall_seconds": elapsed,
                "requests_per_second": response_count / elapsed,
                "tokens_per_second": token_count / elapsed,
                "gpu_before": before,
                "gpu_after": gpu(),
                "command": shlex.join(command),
                "error": command_error,
                "stdout_tail": completed.stdout[-1000:] if completed else None,
                "stderr_tail": completed.stderr[-1000:] if completed else None,
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


def benchmark(config: dict[str, Any], approaches: list[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    allowed = {"api", "run-batch", "python"}
    settings = config.get("benchmark", {})
    selected = approaches or settings.get("approaches", sorted(allowed))
    selected = [str(item) for item in selected]
    if not selected or any(item not in allowed for item in selected):
        raise ValueError("benchmark approaches must be selected from api, run-batch, and python")
    rows = benchmark_rows(config, limit)
    requests = build_requests(config, rows)
    output_dir = resolve(config, config["output"]["directory"])
    path = resolve(config, settings.get("output", output_dir / "benchmark.json"))
    results: dict[str, Any] = {
        "run_id": config["run"]["id"],
        "model": config["model"],
        "rows": len(rows),
        "variants": len(config["variants"]),
        "requests": len(requests),
        "approaches": {},
    }
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            same_workload = (
                previous.get("run_id") == results["run_id"]
                and previous.get("rows") == results["rows"]
                and previous.get("variants") == results["variants"]
                and previous.get("model", {}).get("name") == results["model"].get("name")
            )
            if same_workload:
                results["approaches"].update(previous.get("approaches", {}))
        except (OSError, json.JSONDecodeError):
            pass
    for approach in selected:
        if approach == "python":
            result = benchmark_python(config, rows)
        elif approach == "api":
            result = benchmark_api(config, requests)
        else:
            result = benchmark_run_batch(config, requests, output_dir)
        result["summary"] = summarise_benchmark(result)
        results["approaches"][approach] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results


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
        benchmark_result = benchmark(test, ["python"], 2)
        if benchmark_result["requests"] != len(test["variants"]) * 2:
            raise RuntimeError("benchmark self-test failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "run-matrix", "prepare", "prepare-matrix", "batch-command", "self-test"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
        command.add_argument("--run-id")
        command.add_argument("--model")
        command.add_argument("--backend", choices=["local_vllm", "fake"])
        command.add_argument("--output")
        if name == "batch-command":
            command.add_argument("--dataset", help="Dataset id when using a matrix configuration")
        if name in {"run", "run-matrix"}:
            command.add_argument("--rows", type=int, default=None, help="Optional row limit")
        if name in {"run-matrix", "prepare-matrix"}:
            command.add_argument("--datasets", help="Comma-separated dataset ids to run (default: all)")
    benchmark_parser = commands.add_parser("benchmark")
    benchmark_parser.add_argument("--config", required=True)
    benchmark_parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    benchmark_parser.add_argument("--run-id")
    benchmark_parser.add_argument("--model")
    benchmark_parser.add_argument("--backend", choices=["local_vllm", "fake"])
    benchmark_parser.add_argument("--output")
    benchmark_parser.add_argument(
        "--approaches",
        default=None,
        help="Comma-separated subset of api,run-batch,python (default: all configured approaches)",
    )
    benchmark_parser.add_argument("--rows", type=int, default=None, help="Limit benchmark input rows")
    parse = commands.add_parser("parse")
    parse.add_argument("--config", required=True)
    parse.add_argument("--responses", required=True)
    parse.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parse.add_argument("--run-id")
    parse.add_argument("--model")
    parse.add_argument("--backend", choices=["local_vllm", "fake"])
    parse.add_argument("--output")
    parse.add_argument("--dataset", help="Dataset id when using a matrix configuration")
    args = parser.parse_args()
    config = load_config(args.config, config_overrides(args))
    if getattr(args, "dataset", None):
        config = select_dataset(config, args.dataset)
    if args.command == "validate":
        print(f"Valid configuration: {config['run']['id']}")
    elif args.command == "prepare":
        print(prepare(config))
    elif args.command == "prepare-matrix":
        selected = args.datasets.split(",") if args.datasets else None
        print(json.dumps([str(path) for path in prepare_matrix(config, selected)], indent=2))
    elif args.command == "run-matrix":
        selected = args.datasets.split(",") if args.datasets else None
        print(json.dumps(run_matrix(config, args.rows, selected), indent=2))
    elif args.command == "batch-command":
        print(batch_command(config))
    elif args.command == "parse":
        print(json.dumps(parse_batch(config, args.responses), indent=2))
    elif args.command == "self-test":
        self_test(config)
        print("Self-test passed")
    elif args.command == "benchmark":
        approaches = args.approaches.split(",") if args.approaches else None
        print(json.dumps(benchmark(config, approaches, args.rows), indent=2))
    else:
        print(json.dumps(run(config, row_limit=getattr(args, "rows", None)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
