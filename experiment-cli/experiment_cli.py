#!/usr/bin/env python3
"""Generic, YAML-configured local LLM experiment runner.

All application-specific choices live in a YAML file and Markdown/JSON assets.
This file owns the complete execution path so the CLI is easy to copy and audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import os
import re
import shlex
import sqlite3
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
RESULT_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("dataset_id", pa.string()),
        ("variant_id", pa.string()),
        ("input_row_id", pa.string()),
        ("source_position", pa.int64()),
        ("input_text", pa.string()),
        ("gold_labels", pa.string()),
        ("prompt_hash", pa.string()),
        ("config_hash", pa.string()),
        ("prompt_group_id", pa.string()),
        ("attempt_count", pa.int64()),
        ("raw_response", pa.string()),
        ("parsed_output", pa.string()),
        ("validation_status", pa.string()),
        ("validation_errors", pa.string()),
        ("final_status", pa.string()),
        ("batch_size", pa.int64()),
        ("latency_seconds", pa.float64()),
        ("rows_per_second", pa.float64()),
        ("token_count", pa.int64()),
        ("gpu_snapshot", pa.string()),
        ("candidate_logprobs", pa.string()),
    ]
)
TOKEN = re.compile(
    r"{{\s*(text|row_id|dataset_id|labels|candidate_mapping|question|definitions|theory|output_schema|raw_response|validation_errors|candidates)\s*}}"
)
UNRESOLVED_TOKEN = re.compile(r"{{\s*[^{}]+\s*}}")


def resolve(config: dict[str, Any], value: str | Path) -> Path:
    expanded = os.path.expandvars(str(value))
    # Configs are portable between local machines and shared GPU environments.
    # When the optional TFM_ROOT variable is not set locally, infer it from the
    # repository layout instead of leaving a literal ``${TFM_ROOT}`` path.
    if "${TFM_ROOT}" in expanded:
        expanded = expanded.replace("${TFM_ROOT}", str(Path(config["_root"]).parent))
    path = Path(expanded)
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


def load_config(path: str | Path, overrides: list[str] | None = None, *, check_files: bool = True) -> dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    config = deepcopy(config)
    config["_root"] = str(path.parent.parent if path.parent.name in {"config", "experiments"} else path.parent)
    override_keys: list[str] = []
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must use KEY=VALUE syntax: {item}")
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError(f"Override key is empty: {item}")
        _set_path(config, key, yaml.safe_load(raw_value))
        override_keys.append(key)
    config["_override_keys"] = override_keys
    validate_config(config, check_files=check_files)
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
        if (
            variant["request_mode"] == "candidate_logprobs"
            and not variant.get("candidates")
            and not variant.get("candidates_from")
        ):
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
                for pair in source.get("additional_pairs", []):
                    paths.extend([pair["path"], pair["labels_path"]])
            paths.extend(str(item) for item in source.get("prompt_parts", {}).values())
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
        pairs = [(path, Path(source["labels_path"]))]
        pairs.extend((Path(pair["path"]), Path(pair["labels_path"])) for pair in source.get("additional_pairs", []))
        columns = source.get("label_columns")
        rows = []
        for argument_path, label_path in pairs:
            with argument_path.open(encoding="utf-8", newline="") as handle:
                arguments = {row[id_column]: dict(row) for row in csv.DictReader(handle, delimiter="\t")}
            with label_path.open(encoding="utf-8", newline="") as handle:
                labels = {row[id_column]: dict(row) for row in csv.DictReader(handle, delimiter="\t")}
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
                            key
                            for key in selected
                            if str(label_row.get(key, "0")).strip() in {"1", "1.0", "true", "True"}
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


def dataset_runtime(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_labels": list(source.get("labels", [])),
        "code_labels": dict(source.get("code_labels", {})),
        "binary_question": source.get("binary_question", "Does this text express the target value?"),
        "prompt_parts": dict(source.get("prompt_parts", {})),
    }


def select_dataset(config: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    for identifier, source in dataset_entries(config):
        if identifier == dataset_id:
            selected = deepcopy(config)
            selected.pop("datasets", None)
            selected["input"] = source
            selected["run"] = {
                **config["run"],
                "id": f"{config['run']['id']}__{identifier}",
                "dataset_id": identifier,
                **dataset_runtime(source),
            }
            base_output = resolve(config, config["output"]["directory"])
            selected["output"] = {**config["output"], "directory": str(base_output / f"dataset={identifier}")}
            return selected
    raise ValueError(f"Unknown dataset id: {dataset_id}")


def require_single_input(config: dict[str, Any], command: str) -> None:
    if "datasets" in config:
        raise ValueError(f"{command} requires one input; use --dataset or {command}-matrix for a matrix config")


def rows_for_source(config: dict[str, Any], source: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    resolved_source = dict(source)
    if resolved_source.get("format") == "paired_tsv" and resolved_source.get("labels_path"):
        resolved_source["labels_path"] = str(resolve(config, resolved_source["labels_path"]))
        resolved_source["additional_pairs"] = [
            {"path": str(resolve(config, pair["path"])), "labels_path": str(resolve(config, pair["labels_path"]))}
            for pair in resolved_source.get("additional_pairs", [])
        ]
    rows = read_rows(
        resolve(config, resolved_source["path"]),
        resolved_source["format"],
        resolved_source["id_column"],
        resolved_source["text_column"],
        resolved_source,
    )
    configured_limit = source.get("limit")
    effective_limit = limit if limit is not None else configured_limit
    if effective_limit is not None:
        if int(effective_limit) < 1:
            raise ValueError("row limit must be positive")
        return rows[: int(effective_limit)]
    return rows


def iter_rows_for_source(config: dict[str, Any], source: dict[str, Any], limit: int | None = None) -> Any:
    """Yield normalised rows without materialising large delimited sources."""
    resolved_source = dict(source)
    if resolved_source.get("format") == "paired_tsv" and resolved_source.get("labels_path"):
        # Paired ValueEval files are small enough to use the existing exact
        # join implementation; the multi-million-row ProtoEthos CSV is not.
        yield from rows_for_source(config, source, limit)
        return
    if resolved_source.get("format") not in {"csv", "tsv"}:
        yield from rows_for_source(config, source, limit)
        return
    path = resolve(config, resolved_source["path"])
    delimiter = resolved_source.get("delimiter", "\t" if resolved_source["format"] == "tsv" else ",")
    configured_limit = source.get("limit")
    effective_limit = limit if limit is not None else configured_limit
    if effective_limit is not None and int(effective_limit) < 1:
        raise ValueError("row limit must be positive")
    emitted = 0
    position = 0
    where = resolved_source.get("where", {})
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle, delimiter=delimiter):
            row = dict(raw)
            if where and any(str(row.get(key)) != str(value) for key, value in where.items()):
                continue
            if resolved_source.get("labels_column") and "_gold_labels" not in row:
                row["_gold_labels"] = _split_labels(row.get(resolved_source["labels_column"]))
            if resolved_source["id_column"] not in row or resolved_source["text_column"] not in row:
                raise ValueError(
                    f"Input row {position} lacks `{resolved_source['id_column']}` or `{resolved_source['text_column']}`"
                )
            row["_source_position"] = position
            position += 1
            emitted += 1
            yield row
            if effective_limit is not None and emitted >= int(effective_limit):
                break


def row_key(row: dict[str, Any], config: dict[str, Any]) -> tuple[str, int]:
    return (str(row[config["input"]["id_column"]]), int(row["_source_position"]))


def saved_position(row: dict[str, Any]) -> int:
    value = row.get("source_position")
    return -1 if value is None else int(value)


def source_provenance(config: dict[str, Any]) -> dict[str, Any]:
    source = config["input"]
    paths = [resolve(config, source["path"])]
    if source.get("format") == "paired_tsv":
        paths.append(resolve(config, source["labels_path"]))
        for pair in source.get("additional_pairs", []):
            paths.extend([resolve(config, pair["path"]), resolve(config, pair["labels_path"])])
    records = []
    for path in paths:
        stat = path.stat()
        records.append({"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    metadata_hash = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"format": source["format"], "files": records, "metadata_hash": metadata_hash}


def render(template: str, values: dict[str, Any]) -> str:
    rendered = TOKEN.sub(lambda match: str(values.get(match.group(1), "")), template)
    unresolved = UNRESOLVED_TOKEN.search(rendered)
    if unresolved:
        raise ValueError(f"Unsupported or unresolved prompt placeholder: {unresolved.group(0)}")
    return rendered


def prompt(config: dict[str, Any], paths: list[str], values: dict[str, Any]) -> str:
    return "\n\n".join(render(resolve(config, path).read_text(encoding="utf-8").strip(), values) for path in paths)


def prompt_part_values(config: dict[str, Any], values: dict[str, Any] | None = None) -> dict[str, str]:
    """Load reusable Markdown context fragments declared by the input.

    A part is deliberately just a named Markdown file.  The file itself may
    contain any supported generic placeholder, so theory and definitions can
    be reused by many variants without copying text into YAML or Python.
    """
    rendered_values: dict[str, Any] = dict(values or {})
    parts: dict[str, str] = {}
    for name, path in config.get("run", {}).get("prompt_parts", {}).items():
        raw = resolve(config, path).read_text(encoding="utf-8").strip()
        parts[str(name)] = render(raw, {**rendered_values, **parts})
    return parts


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
            conversations = [[{"role": "user", "content": item}] for item in prompts]
            outputs = self.llm.chat(conversations, params, use_tqdm=False)
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
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        del self.llm


def tune(
    backend: Any, variant: dict[str, Any], prompts: list[str], batch: dict[str, Any], events: Events, sync: bool
) -> tuple[int, list[dict[str, Any]]]:
    if batch.get("mode", "auto") == "fixed":
        return int(batch.get("size", batch.get("candidates", [1])[0])), []
    attempts: list[dict[str, Any]] = []
    safe: list[tuple[float, int]] = []
    candidates = sorted(set(int(item) for item in batch.get("candidates", [1])))
    warmup_rows = int(batch.get("warmup_rows", 64))
    warmup = prompts[: max(warmup_rows, *candidates)] or prompts[:1]
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


def prompt_group_id(config: dict[str, Any], variant: dict[str, Any], schema: dict[str, Any] | None) -> str:
    """Identify the reusable static prefix shared by rows in a variant."""
    sentinel = {config["input"]["id_column"]: "<row>", config["input"]["text_column"]: "<text>"}
    static = rendered_prompt(config, variant, sentinel, schema)
    return hashlib.sha256(static.replace("<text>", "{{text}}").encode()).hexdigest()[:16]


def variant_config_hash(config: dict[str, Any], variant: dict[str, Any]) -> str:
    assets = {}
    for path in variant.get("prompts", []):
        assets[str(path)] = resolve(config, path).read_text(encoding="utf-8")
    for name, path in config.get("run", {}).get("prompt_parts", {}).items():
        assets[f"part:{name}"] = resolve(config, path).read_text(encoding="utf-8")
    payload = {
        "variant": variant,
        "model": config.get("model"),
        "input": config.get("input"),
        "run": config.get("run", {}).get("dataset_id", "default"),
        "assets": assets,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ResumeIndex:
    """Disk-backed resume keys so multi-million-row runs stay bounded in RAM."""

    def __init__(self, path: Path, fingerprint: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS completed (variant_id TEXT, input_row_id TEXT, source_position INTEGER, PRIMARY KEY (variant_id, input_row_id, source_position))"
        )
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        self.cleared = False
        if fingerprint is not None:
            previous = self.connection.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
            if previous and previous[0] != fingerprint:
                self.connection.execute("DELETE FROM completed")
                self.cleared = True
            self.connection.execute("INSERT OR REPLACE INTO metadata VALUES ('fingerprint', ?)", (fingerprint,))
        self.connection.commit()

    def add(self, key: tuple[str, str, int]) -> None:
        self.connection.execute("INSERT OR IGNORE INTO completed VALUES (?, ?, ?)", key)

    def contains(self, key: tuple[str, str, int]) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM completed WHERE variant_id=? AND input_row_id=? AND source_position=? LIMIT 1", key
            ).fetchone()
            is not None
        )

    def seed_from(self, paths: list[Path], expected_hashes: dict[str, str] | None = None) -> int:
        count = 0
        for path in paths:
            if not path.exists():
                continue
            parquet = pq.ParquetFile(path)
            columns = ["variant_id", "input_row_id", "source_position"]
            if expected_hashes and "config_hash" in parquet.schema.names:
                columns.append("config_hash")
            for batch in parquet.iter_batches(columns=columns):
                records = batch.to_pylist()
                if expected_hashes:
                    records = [
                        row for row in records if row.get("config_hash") == expected_hashes.get(str(row["variant_id"]))
                    ]
                self.connection.executemany(
                    "INSERT OR IGNORE INTO completed VALUES (?, ?, ?)",
                    [(str(row["variant_id"]), str(row["input_row_id"]), saved_position(row)) for row in records],
                )
                count += len(records)
        self.connection.commit()
        return count

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class PartWriter:
    def __init__(self, run_dir: Path, variant_id: str, target_rows: int = 4096) -> None:
        self.directory = run_dir / "parts" / f"variant={variant_id}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.target_rows = max(1, target_rows)
        self.rows: list[dict[str, Any]] = []
        self.index = len(list(self.directory.glob("part-*.parquet")))

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.target_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        path = self.directory / f"part-{self.index:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(self.rows, schema=RESULT_SCHEMA), path, compression="zstd")
        self.index += 1
        self.rows.clear()

    def close(self) -> None:
        self.flush()


def merge_parts(run_dir: Path, expected_hashes: dict[str, str] | None = None) -> int:
    """Create compatibility result files from append-only variant parts."""
    files = sorted((run_dir / "parts").glob("variant=*/part-*.parquet")) if (run_dir / "parts").exists() else []
    if not files:
        return 0
    writer: pq.ParquetWriter | None = None
    variant_writers: dict[str, pq.ParquetWriter] = {}
    count = 0
    try:
        for path in files:
            table = pq.read_table(path)
            if expected_hashes and "config_hash" in table.column_names:
                variant_id = path.parent.name.split("=", 1)[-1]
                table = table.filter(pa.compute.equal(table["config_hash"], expected_hashes.get(variant_id, "")))
                if table.num_rows == 0:
                    continue
            if writer is None:
                writer = pq.ParquetWriter(run_dir / "results.parquet", table.schema, compression="zstd")
            writer.write_table(table)
            variant = path.parent.name.split("=", 1)[-1]
            if variant not in variant_writers:
                variant_writers[variant] = pq.ParquetWriter(
                    run_dir / f"{variant}.parquet", table.schema, compression="zstd"
                )
            variant_writers[variant].write_table(table)
            count += table.num_rows
    finally:
        if writer is not None:
            writer.close()
        for item in variant_writers.values():
            item.close()
    return count


def run_streaming(config: dict[str, Any], backend: Any | None = None, row_limit: int | None = None) -> dict[str, Any]:
    """Execute a source in bounded chunks and append Parquet parts."""
    run_id = config["run"]["id"]
    run_dir = resolve(config, config["output"]["directory"])
    logging_config = config.get("logging", {})
    if any(key in config.get("_override_keys", []) for key in ("run.id", "output.directory")):
        logging_config = dict(logging_config)
        logging_config["file"] = str(run_dir / "logs" / f"{run_id}.log")
        logging_config["events"] = str(run_dir / "logs" / f"{run_id}.events.jsonl")
    events = Events(
        resolve(config, logging_config.get("file", f"logs/{run_id}.log")),
        resolve(config, logging_config.get("events", f"logs/{run_id}.events.jsonl")),
        logging_config.get("level", "INFO"),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    expected_hashes = {
        str(variant["id"]): variant_config_hash(config, materialize_variant(config, variant))
        for variant in config["variants"]
    }
    index = ResumeIndex(
        run_dir / ".resume.sqlite", hashlib.sha256(json.dumps(expected_hashes, sort_keys=True).encode()).hexdigest()
    )
    if index.cleared:
        import shutil

        if (run_dir / "parts").exists():
            shutil.rmtree(run_dir / "parts")
        if (run_dir / "results.parquet").exists():
            (run_dir / "results.parquet").unlink()
        for variant in config["variants"]:
            v_parquet = run_dir / f"{variant['id']}.parquet"
            if v_parquet.exists():
                v_parquet.unlink()
    part_files = list((run_dir / "parts").glob("variant=*/part-*.parquet")) if (run_dir / "parts").exists() else []
    seeded = index.seed_from(
        part_files or ([run_dir / "results.parquet"] if (run_dir / "results.parquet").exists() else []), expected_hashes
    )
    created = False
    selected: dict[str, Any] = {}
    total_input = 0
    total_results = seeded
    try:
        for configured_variant in config["variants"]:
            variant = materialize_variant(config, configured_variant)
            schema = variant_schema(config, variant)
            request_variant = {**variant, "_schema": schema}
            source_iter = iter_rows_for_source(config, config["input"], row_limit)
            batch_config = dict(config.get("batch", {}))
            maximum = int(config["model"].get("max_num_seqs", max(batch_config.get("candidates", [1]))))
            batch_config["candidates"] = [
                item for item in batch_config.get("candidates", [1]) if int(item) <= maximum
            ] or [maximum]
            prefetch_count = max(
                int(batch_config.get("warmup_rows", 64)), *[int(item) for item in batch_config["candidates"]]
            )
            prefetched = list(itertools.islice(source_iter, prefetch_count))
            if not prefetched:
                continue
            if total_input == 0:
                total_input = len(prefetched)
            tune_prompts = [rendered_prompt(config, variant, row, schema) for row in prefetched]
            if backend is None:
                backend = FakeBackend() if config["model"]["backend"] == "fake" else VLLMBackend(config["model"])
                created = True
            size, attempts = tune(
                backend,
                request_variant,
                tune_prompts,
                batch_config,
                events,
                bool(config["model"].get("synchronize_cuda", False)),
            )
            size = min(size, maximum)
            group_id = prompt_group_id(config, variant, schema)
            selected[variant["id"]] = {"selected_batch_size": size, "prompt_group_id": group_id, "tuning": attempts}
            events.emit("variant_started", variant=variant["id"], prompt_group_id=group_id, batch_size=size)
            correction_path = config.get("validation", {}).get("retry", {}).get("correction_prompt")
            correction = resolve(config, correction_path).read_text(encoding="utf-8") if correction_path else None
            writer = PartWriter(run_dir, variant["id"], int(config.get("streaming", {}).get("output_chunk_rows", 4096)))
            pending_rows = itertools.chain(prefetched, source_iter)
            buffer = []
            while True:
                while len(buffer) < size:
                    try:
                        row = next(pending_rows)
                        key = (variant["id"], str(row[config["input"]["id_column"]]), int(row["_source_position"]))
                        if not index.contains(key):
                            buffer.append(row)
                    except StopIteration:
                        break
                if not buffer:
                    break
                chunk = buffer[:size]
                prompts = [rendered_prompt(config, variant, row, schema) for row in chunk]
                started = time.perf_counter()
                while True:
                    try:
                        responses = backend.generate(prompts, request_variant)
                        if len(responses) != len(chunk):
                            raise BackendFailure(
                                f"Backend returned {len(responses)} responses for {len(chunk)} prompts"
                            )
                        break
                    except BackendFailure as exc:
                        minimum = int(batch_config.get("min_size", 1))
                        if len(chunk) <= minimum:
                            raise
                        size = max(minimum, len(chunk) // 2)
                        selected[variant["id"]]["runtime_batch_size"] = size
                        events.emit("batch_runtime_backoff", variant=variant["id"], new_batch_size=size, error=str(exc))
                        chunk = chunk[:size]
                        prompts = prompts[:size]
                elapsed = max(time.perf_counter() - started, 1e-9)
                snapshot = serialise(gpu())
                for row, current_prompt, response in zip(chunk, prompts, responses):
                    parsed, errors = validate_response(response.raw, schema)
                    raw = response.raw
                    attempt_count = 1
                    if request_variant["request_mode"] == "candidate_logprobs":
                        parsed, errors = {"candidates": response.candidate_logprobs or {}}, []

                    if (
                        errors
                        and correction
                        and config.get("validation", {}).get("retry", {}).get("enabled")
                        and request_variant["request_mode"] != "candidate_logprobs"
                    ):
                        events.emit(
                            "retry_started", variant=variant["id"], input_row_id=str(row[config["input"]["id_column"]])
                        )
                        for _ in range(int(config["validation"]["retry"].get("max_attempts", 0))):
                            retry_prompt = render(
                                correction,
                                retry_values(config, variant, row, schema, raw, errors),
                            )
                            raw = backend.generate([retry_prompt], request_variant)[0].raw
                            parsed, errors = validate_response(raw, schema)
                            attempt_count += 1
                            if not errors:
                                break
                        events.emit(
                            "retry_completed",
                            variant=variant["id"],
                            input_row_id=str(row[config["input"]["id_column"]]),
                            attempts=attempt_count,
                            validation_status="valid" if not errors else "invalid",
                        )

                    output = {
                        "run_id": run_id,
                        "dataset_id": config["run"].get("dataset_id", "default"),
                        "variant_id": variant["id"],
                        "input_row_id": str(row[config["input"]["id_column"]]),
                        "source_position": row["_source_position"],
                        "input_text": str(row[config["input"]["text_column"]])
                        if config["output"].get("include_text")
                        else None,
                        "gold_labels": serialise(row.get("_gold_labels")),
                        "prompt_hash": hashlib.sha256(current_prompt.encode()).hexdigest(),
                        "config_hash": variant_config_hash(config, variant),
                        "prompt_group_id": group_id,
                        "attempt_count": attempt_count,
                        "raw_response": raw if config["output"].get("include_raw_response", True) else None,
                        "parsed_output": serialise(parsed),
                        "validation_status": "valid" if not errors else "invalid",
                        "validation_errors": serialise(errors),
                        "final_status": "completed" if not errors else "failed_validation",
                        "batch_size": size,
                        "latency_seconds": elapsed / len(chunk),
                        "rows_per_second": len(chunk) / elapsed,
                        "token_count": response.token_count,
                        "gpu_snapshot": snapshot,
                        "candidate_logprobs": serialise(response.candidate_logprobs),
                    }
                    writer.append(output)
                    index.add((variant["id"], output["input_row_id"], int(output["source_position"])))
                    total_results += 1
                index.connection.commit()
                events.emit(
                    "batch_completed", variant=variant["id"], rows=len(chunk), rows_per_second=len(chunk) / elapsed
                )
                buffer = buffer[len(chunk) :]
            writer.close()
            events.emit("variant_completed", variant=variant["id"])
            if total_input == len(prefetched):
                total_input = sum(1 for _ in iter_rows_for_source(config, config["input"], row_limit))
        merged = merge_parts(run_dir, expected_hashes)
    finally:
        if created and backend is not None:
            backend.close()
        index.close()
    effective = {key: value for key, value in config.items() if not key.startswith("_")}
    (run_dir / "effective_config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "dataset_id": config["run"].get("dataset_id", "default"),
        "input_rows": total_input,
        "result_rows": merged or total_results,
        "model": config["model"],
        "variants": selected,
        "streaming": True,
        "gpu_preflight": gpu(),
        "source_provenance": source_provenance(config),
        "resume_skipped_rows": seeded,
        "event_log": str(events.path),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    events.emit("run_completed", result_rows=manifest["result_rows"], gpu=gpu())
    return manifest


def run(config: dict[str, Any], backend: Any | None = None, row_limit: int | None = None) -> dict[str, Any]:
    if config.get("streaming", {}).get("enabled", False):
        return run_streaming(config, backend, row_limit)
    run_id = config["run"]["id"]
    run_dir = resolve(config, config["output"]["directory"])
    logging_config = config.get("logging", {})
    if any(key in config.get("_override_keys", []) for key in ("run.id", "output.directory")):
        logging_config = dict(logging_config)
        logging_config["file"] = str(run_dir / "logs" / f"{run_id}.log")
        logging_config["events"] = str(run_dir / "logs" / f"{run_id}.events.jsonl")
    events = Events(
        resolve(config, logging_config.get("file", f"logs/{run_id}.log")),
        resolve(config, logging_config.get("events", f"logs/{run_id}.events.jsonl")),
        logging_config.get("level", "INFO"),
    )
    rows = rows_for_source(config, config["input"], row_limit)
    result_path = run_dir / "results.parquet"
    existing = pq.read_table(result_path).to_pylist() if result_path.exists() else []
    expected_hashes = {
        str(item["id"]): variant_config_hash(config, materialize_variant(config, item)) for item in config["variants"]
    }
    existing = [row for row in existing if row.get("config_hash") == expected_hashes.get(str(row["variant_id"]))]
    complete = {
        (str(row["variant_id"]), str(row["input_row_id"]), saved_position(row))
        for row in existing
        if row.get("config_hash") == expected_hashes.get(str(row["variant_id"]))
    }
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
        for configured_variant in config["variants"]:
            variant = materialize_variant(config, configured_variant)
            pending = [row for row in rows if (variant["id"], *row_key(row, config)) not in complete]
            if not pending:
                events.emit("variant_resumed", variant=variant["id"], skipped=len(rows))
                continue
            if backend is None:
                backend = FakeBackend() if config["model"]["backend"] == "fake" else VLLMBackend(config["model"])
                created = True
            schema_path = variant.get("validation", {}).get("schema")
            schema = json.loads(resolve(config, schema_path).read_text(encoding="utf-8")) if schema_path else None
            request_variant = {**variant, "_schema": schema}
            prompts = [rendered_prompt(config, variant, row, schema) for row in pending]
            group_id = prompt_group_id(config, variant, schema)
            batch = dict(config.get("batch", {}))
            maximum = int(config["model"].get("max_num_seqs", max(batch.get("candidates", [1]))))
            batch["candidates"] = [item for item in batch.get("candidates", [1]) if int(item) <= maximum] or [maximum]
            size, attempts = tune(
                backend, request_variant, prompts, batch, events, bool(config["model"].get("synchronize_cuda", False))
            )
            size = min(size, maximum)
            selected[variant["id"]] = {
                "selected_batch_size": size,
                "prompt_group_id": group_id,
                "tuning": attempts,
                "pending_rows": len(pending),
            }
            events.emit(
                "variant_started", variant=variant["id"], prompt_group_id=group_id, rows=len(pending), batch_size=size
            )
            offset = 0
            while offset < len(pending):
                chunk, chunk_prompts = pending[offset : offset + size], prompts[offset : offset + size]
                sync_cuda(bool(config["model"].get("synchronize_cuda", False)))
                started = time.perf_counter()
                while True:
                    try:
                        responses = backend.generate(chunk_prompts, request_variant)
                        if len(responses) != len(chunk):
                            raise BackendFailure(
                                f"Backend returned {len(responses)} responses for {len(chunk)} prompts"
                            )
                        break
                    except BackendFailure as exc:
                        minimum = int(batch.get("min_size", 1))
                        if len(chunk) <= minimum:
                            raise
                        new_size = max(minimum, len(chunk) // 2)
                        events.emit(
                            "batch_runtime_backoff",
                            variant=variant["id"],
                            old_batch_size=len(chunk),
                            new_batch_size=new_size,
                            error=str(exc),
                        )
                        size = new_size
                        selected[variant["id"]]["runtime_batch_size"] = size
                        chunk = pending[offset : offset + size]
                        chunk_prompts = prompts[offset : offset + size]
                sync_cuda(bool(config["model"].get("synchronize_cuda", False)))
                elapsed = max(time.perf_counter() - started, 1e-9)
                batch_gpu = gpu()
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
                                correction,
                                retry_values(config, variant, row, schema, raw, errors),
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
                            "gold_labels": serialise(row.get("_gold_labels")),
                            "prompt_hash": hashlib.sha256(current_prompt.encode()).hexdigest(),
                            "config_hash": variant_config_hash(config, variant),
                            "prompt_group_id": group_id,
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
                            "gpu_snapshot": serialise(batch_gpu),
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
                offset += len(chunk)
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
        "source_provenance": source_provenance(config),
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
                **dataset_runtime(source),
            }
            dataset_config["output"] = {
                **config["output"],
                "directory": str(base_output / f"dataset={dataset_id}"),
            }
            logging_config = dict(config.get("logging", {}))
            # Matrix workers must never share the YAML's default log names;
            # derive them from the effective output directory and run id.
            logging_config["file"] = str(base_output / "logs" / f"{dataset_id}.log")
            logging_config["events"] = str(base_output / "logs" / f"{dataset_id}.events.jsonl")
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
    path = run_dir / "requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for variant in config["variants"]:
            schema = variant_schema(config, variant)
            for row in iter_rows_for_source(config, config["input"]):
                handle.write(json.dumps(request_for_row(config, variant, row, schema), ensure_ascii=False) + "\n")
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
            **dataset_runtime(source),
        }
        dataset_config["output"] = {**config["output"], "directory": str(base_output / f"dataset={dataset_id}")}
        paths.append(prepare(dataset_config))
    return paths


def variant_schema(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any] | None:
    schema_path = variant.get("validation", {}).get("schema")
    schema = json.loads(resolve(config, schema_path).read_text(encoding="utf-8")) if schema_path else None
    enum_from = variant.get("validation", {}).get("enum_from") if schema else None
    labels = list(config.get("run", {}).get("dataset_labels", []))
    if enum_from == "dataset_labels" and labels:

        def replace_enums(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "string" and "enum" in node:
                    node["enum"] = labels
                for child in node.values():
                    replace_enums(child)
            elif isinstance(node, list):
                for child in node:
                    replace_enums(child)

        replace_enums(schema)
    return schema


def materialize_variant(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    materialized = deepcopy(variant)
    source = materialized.get("candidates_from")
    if source == "dataset_labels":
        materialized["candidates"] = list(config.get("run", {}).get("dataset_labels", []))
    elif source == "code_labels":
        mapping = config.get("run", {}).get("code_labels", {})
        materialized["candidates"] = list(mapping)
    return materialized


def rendered_prompt(
    config: dict[str, Any], variant: dict[str, Any], row: dict[str, Any], schema: dict[str, Any] | None = None
) -> str:
    variant = materialize_variant(config, variant)
    if schema is None and variant.get("validation", {}).get("schema"):
        schema = variant_schema(config, variant)
    values = {
        "text": row[config["input"]["text_column"]],
        "row_id": row[config["input"]["id_column"]],
        "dataset_id": config["run"].get("dataset_id", "default"),
        "candidates": ", ".join(str(item) for item in variant.get("candidates", [])),
        "labels": ", ".join(str(item) for item in config.get("run", {}).get("dataset_labels", [])),
        "candidate_mapping": ", ".join(
            f"{code}={label}" for code, label in config.get("run", {}).get("code_labels", {}).items()
        )
        or ", ".join(str(item) for item in variant.get("candidates", [])),
        "question": config.get("run", {}).get("binary_question", "Does this text express the target value?"),
        "output_schema": json.dumps(schema or {}, sort_keys=True),
    }
    values.update(prompt_part_values(config, values))
    content = prompt(config, variant["prompts"], values)
    return content


def retry_values(
    config: dict[str, Any],
    variant: dict[str, Any],
    row: dict[str, Any],
    schema: dict[str, Any] | None,
    raw: str,
    errors: list[str],
) -> dict[str, Any]:
    variant = materialize_variant(config, variant)
    values = {
        "text": row[config["input"]["text_column"]],
        "row_id": row[config["input"]["id_column"]],
        "dataset_id": config["run"].get("dataset_id", "default"),
        "candidates": ", ".join(str(item) for item in variant.get("candidates", [])),
        "labels": ", ".join(str(item) for item in config.get("run", {}).get("dataset_labels", [])),
        "candidate_mapping": ", ".join(
            f"{code}={label}" for code, label in config.get("run", {}).get("code_labels", {}).items()
        )
        or ", ".join(str(item) for item in variant.get("candidates", [])),
        "question": config.get("run", {}).get("binary_question", "Does this text express the target value?"),
        "output_schema": json.dumps(schema or {}, sort_keys=True),
        "raw_response": raw,
        "validation_errors": "; ".join(errors),
    }
    values.update(prompt_part_values(config, values))
    return values


def request_for_row(
    config: dict[str, Any], variant: dict[str, Any], row: dict[str, Any], schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    variant = materialize_variant(config, variant)
    content = rendered_prompt(config, variant, row, schema)
    body: dict[str, Any] = {
        "model": config["model"]["name"],
        "messages": [{"role": "user", "content": content}],
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
        "custom_id": f"{variant['id']}:{row[config['input']['id_column']]}:{row['_source_position']}",
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
    responses: dict[str, dict[str, Any]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = str(item["custom_id"])
        if custom_id in responses:
            raise ValueError(f"Duplicate custom_id in batch response: {custom_id}")
        responses[custom_id] = item
    run_id = config["run"]["id"]
    run_dir = resolve(config, config["output"]["directory"])
    logging_config = config.get("logging", {})
    if any(key in config.get("_override_keys", []) for key in ("run.id", "output.directory")):
        logging_config = dict(logging_config)
        logging_config["file"] = str(run_dir / "logs" / f"{run_id}.log")
        logging_config["events"] = str(run_dir / "logs" / f"{run_id}.events.jsonl")
    events = Events(
        resolve(config, logging_config.get("file", f"logs/{run_id}.log")),
        resolve(config, logging_config.get("events", f"logs/{run_id}.events.jsonl")),
        logging_config.get("level", "INFO"),
    )
    rows = rows_for_source(config, config["input"])
    result_path = run_dir / "results.parquet"
    saved = pq.read_table(result_path).to_pylist() if result_path.exists() else []
    expected_hashes = {
        str(item["id"]): variant_config_hash(config, materialize_variant(config, item)) for item in config["variants"]
    }
    saved = [row for row in saved if row.get("config_hash") == expected_hashes.get(str(row["variant_id"]))]
    complete = {
        (str(row["variant_id"]), str(row["input_row_id"]), saved_position(row))
        for row in saved
        if row.get("config_hash") == expected_hashes.get(str(row["variant_id"]))
    }
    retry_settings = config.get("validation", {}).get("retry", {})
    correction_path = retry_settings.get("correction_prompt")
    correction = resolve(config, correction_path).read_text(encoding="utf-8") if correction_path else None
    retry_requests: list[dict[str, Any]] = []

    def add_retry_request(
        variant: dict[str, Any], row: dict[str, Any], raw: str, errors: list[str], attempt: int
    ) -> None:
        if not correction or not retry_settings.get("enabled") or attempt > int(retry_settings.get("max_attempts", 0)):
            return
        values = retry_values(config, variant, row, variant_schema(config, variant), raw, errors)
        retry_prompt = render(correction, values)
        schema = variant_schema(config, variant)
        body: dict[str, Any] = {
            "model": config["model"]["name"],
            "messages": [{"role": "user", "content": retry_prompt}],
            "temperature": 0,
            "max_completion_tokens": variant.get("max_tokens", 128),
        }
        if variant["request_mode"] == "candidate_logprobs":
            body.update(
                {
                    "max_completion_tokens": 1,
                    "logprobs": True,
                    "top_logprobs": max(20, len(variant.get("candidates", [])) + 5),
                }
            )
        elif schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": variant["id"], "schema": schema, "strict": True},
            }
        retry_requests.append(
            {
                "custom_id": f"retry:{variant['id']}:{row[config['input']['id_column']]}:{row['_source_position']}:{attempt}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )

    for configured_variant in config["variants"]:
        variant = materialize_variant(config, configured_variant)
        schema_path = variant.get("validation", {}).get("schema")
        schema = json.loads(resolve(config, schema_path).read_text(encoding="utf-8")) if schema_path else None
        group_id = prompt_group_id(config, variant, schema)
        for row in rows:
            row_id = str(row[config["input"]["id_column"]])
            key = (variant["id"], *row_key(row, config))
            retry_items = [
                item
                for item_id, item in responses.items()
                if item_id.startswith(f"retry:{variant['id']}:{row_id}:{row['_source_position']}:")
            ]
            if key in complete and not retry_items:
                continue
            current_prompt = rendered_prompt(config, variant, row, schema)
            custom_id = f"{variant['id']}:{row_id}:{row['_source_position']}"
            if custom_id not in responses:
                raw, batch_error, observed = None, "missing_batch_response", None
            else:
                raw, batch_error, observed = _batch_text(responses[custom_id])
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
            attempt_count = 1
            if retry_items:
                retry_id = max(
                    (
                        item_id
                        for item_id in responses
                        if item_id.startswith(f"retry:{variant['id']}:{row_id}:{row['_source_position']}:")
                    ),
                    key=lambda x: int(x.rsplit(":", 1)[-1]),
                )
                retry_raw, retry_error, retry_observed = _batch_text(responses[retry_id])
                if retry_raw is not None:
                    raw = retry_raw
                    parsed, errors = validate_response(raw, schema)
                    attempt_count = int(retry_id.rsplit(":", 1)[-1])
                    if variant["request_mode"] == "candidate_logprobs" and retry_observed is not None:
                        scores = {
                            candidate: retry_observed.get(str(candidate).strip(), -float("inf"))
                            for candidate in variant["candidates"]
                        }
                        parsed, errors = {"candidates": scores}, []
                else:
                    errors = [retry_error or "missing_retry_response"]
                target = next(
                    (
                        item
                        for item in reversed(saved)
                        if item["variant_id"] == variant["id"]
                        and item["input_row_id"] == row_id
                        and saved_position(item) == int(row["_source_position"])
                    ),
                    None,
                )
                if target is not None:
                    target.update(
                        {
                            "attempt_count": attempt_count,
                            "raw_response": raw,
                            "parsed_output": serialise(parsed),
                            "validation_status": "valid" if not errors else "invalid",
                            "validation_errors": serialise(errors),
                            "final_status": "completed" if not errors else "failed_validation",
                            "candidate_logprobs": serialise(scores),
                        }
                    )
                    add_retry_request(variant, row, raw or "", errors, attempt_count + 1)
                    continue
            add_retry_request(variant, row, raw or "", errors, attempt_count + 1)
            saved.append(
                {
                    "run_id": run_id,
                    "dataset_id": config["run"].get("dataset_id", "default"),
                    "variant_id": variant["id"],
                    "input_row_id": row_id,
                    "source_position": row["_source_position"],
                    "input_text": str(row[config["input"]["text_column"]])
                    if config["output"].get("include_text")
                    else None,
                    "gold_labels": serialise(row.get("_gold_labels")),
                    "prompt_hash": hashlib.sha256(current_prompt.encode()).hexdigest(),
                    "config_hash": variant_config_hash(config, variant),
                    "prompt_group_id": group_id,
                    "attempt_count": attempt_count,
                    "raw_response": raw if config["output"].get("include_raw_response", True) else None,
                    "parsed_output": serialise(parsed),
                    "validation_status": "valid" if not errors else "invalid",
                    "validation_errors": serialise(errors),
                    "final_status": "completed"
                    if not errors
                    else (
                        "retry_pending"
                        if correction
                        and retry_settings.get("enabled")
                        and int(retry_settings.get("max_attempts", 0)) >= 1
                        else "failed_validation"
                    ),
                    "batch_size": None,
                    "latency_seconds": None,
                    "rows_per_second": None,
                    "token_count": None,
                    "gpu_snapshot": None,
                    "candidate_logprobs": serialise(scores),
                }
            )
    write_results(run_dir, saved)
    retry_path = None
    if retry_requests:
        retry_path = run_dir / "retry_requests.jsonl"
        retry_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in retry_requests), encoding="utf-8"
        )
    manifest = {
        "run_id": run_id,
        "dataset_id": config["run"].get("dataset_id", "default"),
        "input_rows": len(rows),
        "result_rows": len(saved),
        "model": config["model"],
        "variants": {variant["id"]: {"external_batch": True} for variant in config["variants"]},
        "batch_response_path": str(source),
        "source_provenance": source_provenance(config),
        "resume_skipped_rows": len(complete),
        "event_log": str(events.path),
        "retry_request_path": str(retry_path) if retry_path else None,
        "retry_requests": len(retry_requests),
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
    variants = {variant["id"]: materialize_variant(config, variant) for variant in config["variants"]}

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
                item = json.loads(line)
                if item.get("error"):
                    errors += 1
                    continue
                response_count += 1
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
        "workload_hash": hashlib.sha256(
            json.dumps(
                [variant_config_hash(config, materialize_variant(config, variant)) for variant in config["variants"]],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
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
                and previous.get("workload_hash") == results["workload_hash"]
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
    config = load_config(args.config, config_overrides(args), check_files=False)
    if getattr(args, "dataset", None):
        config = select_dataset(config, args.dataset)
    validate_config(config, check_files=True)
    if args.command == "validate":
        print(f"Valid configuration: {config['run']['id']}")
    elif args.command == "prepare":
        require_single_input(config, "prepare")
        print(prepare(config))
    elif args.command == "prepare-matrix":
        selected = [item.strip() for item in args.datasets.split(",") if item.strip()] if args.datasets else None
        print(json.dumps([str(path) for path in prepare_matrix(config, selected)], indent=2))
    elif args.command == "run-matrix":
        selected = [item.strip() for item in args.datasets.split(",") if item.strip()] if args.datasets else None
        print(json.dumps(run_matrix(config, args.rows, selected), indent=2))
    elif args.command == "batch-command":
        require_single_input(config, "batch-command")
        print(batch_command(config))
    elif args.command == "parse":
        require_single_input(config, "parse")
        print(json.dumps(parse_batch(config, args.responses), indent=2))
    elif args.command == "self-test":
        self_test(config)
        print("Self-test passed")
    elif args.command == "benchmark":
        approaches = [item.strip() for item in args.approaches.split(",") if item.strip()] if args.approaches else None
        print(json.dumps(benchmark(config, approaches, args.rows), indent=2))
    else:
        result = (
            run_matrix(config, row_limit=getattr(args, "rows", None))
            if "datasets" in config
            else run(config, row_limit=getattr(args, "rows", None))
        )
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
