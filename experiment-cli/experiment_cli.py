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
from functools import cache
from pathlib import Path
from typing import Any, Protocol

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


def top_logprobs_count(candidates: list[Any]) -> int:
    """How many top logprobs to ask for to score `candidates`.

    The headroom absorbs tokeniser variants of a candidate (' A' vs 'A'); the
    cap is the maximum most OpenAI-compatible servers accept.
    """
    return min(20, len(candidates) + 5)


def extract_top_logprobs(logprob_content: Any) -> list[tuple[str, float]]:
    """Pull (token, logprob) pairs out of an OpenAI-shaped `logprobs.content`.

    Accepts both plain dicts (HTTP/JSONL responses) and SDK objects, which is
    why every lookup goes through `getattr`-then-`get` rather than one or the
    other.
    """
    observed: list[tuple[str, float]] = []
    for token in logprob_content or []:
        candidates = token.get("top_logprobs") if isinstance(token, dict) else getattr(token, "top_logprobs", None)
        for candidate in candidates or []:
            if isinstance(candidate, dict):
                name, logprob = candidate.get("token", ""), candidate.get("logprob", -float("inf"))
            else:
                name, logprob = getattr(candidate, "token", ""), getattr(candidate, "logprob", -float("inf"))
            observed.append((str(name), float(logprob)))
    return observed


def aggregate_candidate_logprobs(raw_logprobs: list[tuple[str, float]], candidates: list[Any]) -> dict[str, float]:
    """Aggregate token logprobs by stripping whitespace and summing probabilities.

    If multiple tokens strip to the same candidate string (e.g. ' A' and 'A'),
    we aggregate their logprobs using logsumexp to avoid underflow/overflow.
    """
    import math

    grouped: dict[str, list[float]] = {}
    for token, logprob in raw_logprobs:
        stripped = token.strip()
        grouped.setdefault(stripped, []).append(logprob)

    aggregated: dict[str, float] = {}
    for stripped, logprobs in grouped.items():
        if len(logprobs) == 1:
            aggregated[stripped] = logprobs[0]
        else:
            max_lp = max(logprobs)
            sum_exp = sum(math.exp(lp - max_lp) for lp in logprobs)
            aggregated[stripped] = max_lp + math.log(sum_exp)

    return {candidate: aggregated.get(str(candidate).strip(), -float("inf")) for candidate in candidates}


def resolve(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve path relative to project root or expand environment variables."""
    expanded = os.path.expandvars(str(value))

    # Configs are portable between local machines and shared GPU environments.
    # When the optional TFM_ROOT variable is not set locally, infer it from the
    # repository layout instead of leaving a literal ``${TFM_ROOT}`` path.
    if "${TFM_ROOT}" in expanded:
        expanded = expanded.replace("${TFM_ROOT}", str(Path(config["_root"]).parent))
    path = Path(expanded)
    return path if path.is_absolute() else Path(config["_root"]) / path


@cache
def _read_asset(path: str) -> str:
    """Read immutable experiment assets once per process.

    Prompt fragments and schemas are static for one run.  Caching them avoids
    reopening the same Markdown files for every row in a large streamed lane.
    """
    return Path(path).read_text(encoding="utf-8")


def read_asset(config: dict[str, Any], value: str | Path) -> str:
    """Read contents of a config asset file, caching it in memory."""
    return _read_asset(str(resolve(config, value).resolve()))


def _set_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    """Set a value in a nested configuration dictionary using a dotted path."""
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
    """Load and validate the YAML configuration, applying path-based overrides."""
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
    """Validate the structural schema and files in the experiment configuration."""
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
    if config["model"].get("backend") not in {"local_vllm", "nvidia_api", "fake"}:
        raise ValueError("model.backend must be local_vllm, nvidia_api, or fake")
    vllm_environment = config["model"].get("vllm_environment", {})
    if not isinstance(vllm_environment, dict):
        raise ValueError("model.vllm_environment must be a mapping")
    if any(not isinstance(key, str) or not key.startswith("VLLM_") for key in vllm_environment):
        raise ValueError("model.vllm_environment keys must start with VLLM_")
    if any(not isinstance(value, (str, int, float, bool)) for value in vllm_environment.values()):
        raise ValueError("model.vllm_environment values must be scalar")
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
    resources = config.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("resources must be a mapping")
    cpu = resources.get("cpu", {})
    if not isinstance(cpu, dict):
        raise ValueError("resources.cpu must be a mapping")
    cores = cpu.get("cores", "auto")
    if cores not in {"auto", "all"} and (isinstance(cores, bool) or not isinstance(cores, int) or cores < 1):
        raise ValueError("resources.cpu.cores must be auto, all, or a positive integer")
    if (
        isinstance(cpu.get("reserve_cores", 2), bool)
        or not isinstance(cpu.get("reserve_cores", 2), int)
        or int(cpu.get("reserve_cores", 2)) < 0
    ):
        raise ValueError("resources.cpu.reserve_cores must be a non-negative integer")
    if (
        isinstance(cpu.get("thread_pool_size", 1), bool)
        or not isinstance(cpu.get("thread_pool_size", 1), int)
        or int(cpu.get("thread_pool_size", 1)) < 1
    ):
        raise ValueError("resources.cpu.thread_pool_size must be a positive integer")
    if not isinstance(cpu.get("affinity", True), bool):
        raise ValueError("resources.cpu.affinity must be a boolean")
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
            paths.extend(system_prompt_paths(variant))
            if schema := variant.get("validation", {}).get("schema"):
                paths.append(schema)
        retry = config.get("validation", {}).get("retry", {}).get("correction_prompt")
        if retry:
            paths.append(retry)
        for item in paths:
            if not resolve(config, item).is_file():
                raise ValueError(f"Configured file does not exist: {item}")


def _validate_source(source: dict[str, Any], name: str) -> None:
    """Validate file format and mandatory columns for a dataset source definition."""
    for key in ("path", "format", "id_column", "text_column"):
        if not source.get(key):
            raise ValueError(f"{name}.{key} is required")
    if source["format"] not in {"csv", "tsv", "jsonl", "parquet", "nested_json", "paired_tsv"}:
        raise ValueError(f"{name}.format is unsupported")
    if source["format"] == "paired_tsv" and not source.get("labels_path"):
        raise ValueError(f"{name}.labels_path is required for paired_tsv")


def _split_labels(value: Any) -> list[str]:
    """Parse labels from string, comma-separated string, JSON list, or list format."""
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
    """Read dataset rows according to format (CSV, TSV, JSONL, Parquet, or nested JSON)."""
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
    """Return all dataset IDs and input sources defined in the configuration."""
    if "datasets" not in config:
        return [(str(config.get("run", {}).get("dataset_id", "default")), config["input"])]
    return [(str(item["id"]), item.get("input", item)) for item in config["datasets"]]


def dataset_runtime(source: dict[str, Any]) -> dict[str, Any]:
    """Extract runtime configuration parameters for a specific dataset source."""
    return {
        "dataset_labels": list(source.get("labels", [])),
        "code_labels": dict(source.get("code_labels", {})),
        "binary_question": source.get("binary_question", "Does this text express the target value?"),
        "prompt_parts": dict(source.get("prompt_parts", {})),
    }


def dataset_config(
    config: dict[str, Any], dataset_id: str, source: dict[str, Any], base_output: Path | None = None
) -> dict[str, Any]:
    """Collapse one lane of a matrix config into a standalone single-input config.

    Every matrix entry point needs exactly this shape, so it lives here rather
    than being rebuilt by `select_dataset`, `run_matrix` and `prepare_matrix`.
    """
    if base_output is None:
        base_output = resolve(config, config["output"]["directory"])
    lane = deepcopy(config)
    lane.pop("datasets", None)
    lane["input"] = source
    lane["run"] = {
        **config["run"],
        "id": f"{config['run']['id']}__{dataset_id}",
        "dataset_id": dataset_id,
        **dataset_runtime(source),
    }
    lane["output"] = {**config["output"], "directory": str(base_output / f"dataset={dataset_id}")}
    return lane


def selected_entries(config: dict[str, Any], selected: list[str] | None) -> list[tuple[str, dict[str, Any]]]:
    """Return the requested dataset lanes, rejecting unknown ids."""
    entries = dataset_entries(config)
    if not selected:
        return entries
    wanted = set(selected)
    entries = [(identifier, source) for identifier, source in entries if identifier in wanted]
    missing = wanted - {identifier for identifier, _ in entries}
    if missing:
        raise ValueError(f"Unknown dataset id(s): {', '.join(sorted(missing))}")
    return entries


def select_dataset(config: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    """Filter and return the configuration for a single selected dataset."""
    for identifier, source in dataset_entries(config):
        if identifier == dataset_id:
            return dataset_config(config, identifier, source)
    raise ValueError(f"Unknown dataset id: {dataset_id}")


def require_single_input(config: dict[str, Any], command: str) -> None:
    """Enforce that the configuration contains a single dataset, not a list of datasets."""
    if "datasets" in config:
        raise ValueError(f"{command} requires one input; use --dataset or {command}-matrix for a matrix config")


def rows_for_source(config: dict[str, Any], source: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    """Read all (or up to limit) rows for a specified source definition."""

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
    """Construct a unique identifier tuple key for a dataset row."""
    return (str(row[config["input"]["id_column"]]), int(row["_source_position"]))


def saved_position(row: dict[str, Any]) -> int:
    """Get the source position of a saved result row, returning -1 if missing."""
    value = row.get("source_position")

    return -1 if value is None else int(value)


def source_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Generate metadata provenance mapping for the input dataset file(s)."""
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
    """Substitute placeholder tokens inside a prompt template string."""
    rendered = TOKEN.sub(lambda match: str(values.get(match.group(1), "")), template)

    unresolved = UNRESOLVED_TOKEN.search(rendered)
    if unresolved:
        raise ValueError(f"Unsupported or unresolved prompt placeholder: {unresolved.group(0)}")
    return rendered


def prompt(config: dict[str, Any], paths: list[str], values: dict[str, Any]) -> str:
    """Assemble a full prompt by reading and rendering multiple markdown asset paths."""
    return "\n\n".join(render(read_asset(config, path).strip(), values) for path in paths)


def prompt_part_values(config: dict[str, Any], values: dict[str, Any] | None = None) -> dict[str, str]:
    """Load reusable Markdown context fragments declared by the input.

    A part is deliberately just a named Markdown file.  The file itself may
    contain any supported generic placeholder, so theory and definitions can
    be reused by many variants without copying text into YAML or Python.
    """
    rendered_values: dict[str, Any] = dict(values or {})
    parts: dict[str, str] = {}
    for name, path in config.get("run", {}).get("prompt_parts", {}).items():
        raw = read_asset(config, path).strip()
        parts[str(name)] = render(raw, {**rendered_values, **parts})
    return parts


def check_schema(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    """Recursively validate a value against a JSON Schema, appending error strings."""
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
    """Parse a raw response as JSON and validate it against the optional schema."""
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
        """Initialize the event logger to write both logs and events to files."""
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
        """Emit a structured event record to the event file and log output."""
        record = {"timestamp": time.time(), "event": event, **payload}

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.logger.info("%s %s", event, json.dumps(payload, sort_keys=True))


_gpu_snapshot_cache: dict[str, Any] = {"taken_at": 0.0, "snapshot": None}


def gpu(max_age_seconds: float = 0.0) -> dict[str, Any]:
    """Query local GPU telemetry, optionally reusing a recent snapshot.

    Each call shells out to nvidia-smi and costs ~20 ms. That is irrelevant once
    per run but not once per batch: a 3M-row lane at batch=128 would spend tens
    of minutes of wall clock there, in between batches. Callers that only want
    a rough per-batch reading pass max_age_seconds; the manifest and preflight
    leave it at 0 and always measure.
    """
    now = time.monotonic()
    cached = _gpu_snapshot_cache["snapshot"]
    if max_age_seconds > 0 and cached is not None and now - _gpu_snapshot_cache["taken_at"] <= max_age_seconds:
        return cached
    snapshot = _gpu_query()
    _gpu_snapshot_cache.update({"taken_at": now, "snapshot": snapshot})
    return snapshot


def _gpu_query() -> dict[str, Any]:
    """Query local GPU usage metrics (memory, utilization) using nvidia-smi and torch."""
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


def gpu_preflight() -> dict[str, Any]:
    """Verify both the host driver and the uv-managed CUDA Python stack."""
    nvidia_smi = gpu()
    report: dict[str, Any] = {
        "ready": False,
        "nvidia_smi": nvidia_smi,
        "device_nodes": [str(path) for path in sorted(Path("/dev").glob("nvidia*"))],
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_report: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        }
        if cuda_available:
            torch_report["device_name"] = torch.cuda.get_device_name(0)
        report["torch"] = torch_report
    except Exception as exc:
        report["torch"] = {"cuda_available": False, "error": str(exc)}
    report["ready"] = bool(nvidia_smi.get("available") and report["torch"].get("cuda_available"))
    if not report["ready"]:
        report["remediation"] = (
            "uv installs CUDA user-space libraries, not the kernel driver. Repair the host NVIDIA driver and "
            "ensure /dev/nvidia* is exposed, then rerun this preflight."
        )
    return report


def sync_cuda(enabled: bool) -> None:
    """Block CPU execution until all ongoing CUDA kernels have completed, if enabled."""
    if enabled:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


def available_cpu_ids() -> list[int]:
    """Return CPUs available to this process, respecting a parent cgroup."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def cpu_resource_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve a bounded CPU allocation from the portable YAML settings."""
    settings = config.get("resources", {}).get("cpu", {})
    available = available_cpu_ids()
    requested = settings.get("cores", "auto")
    reserve = int(settings.get("reserve_cores", 2))
    if requested == "all":
        count = len(available)
    elif requested == "auto":
        count = max(1, len(available) - reserve)
    else:
        count = int(requested)
    if count > len(available):
        raise ValueError(f"resources.cpu.cores={count} exceeds the {len(available)} CPU(s) available to this process")
    return {
        "available_cpu_ids": available,
        "cpu_ids": available[:count],
        "requested_cores": requested,
        "reserve_cores": reserve,
        "thread_pool_size": int(settings.get("thread_pool_size", 1)),
        "affinity": bool(settings.get("affinity", True)),
    }


def apply_resource_guard(config: dict[str, Any]) -> dict[str, Any]:
    """Bound CPU affinity and native thread pools before vLLM starts workers.

    GPU capacity remains controlled by the existing model settings.  The
    affinity is inherited by vLLM's child processes, unlike a post-hoc monitor
    that can only notice overload after the laptop is already unresponsive.
    """
    plan = cpu_resource_plan(config)
    threads = str(plan["thread_pool_size"])
    environment = {
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads,
        "RAYON_NUM_THREADS": threads,
        "TOKENIZERS_PARALLELISM": "false",
    }
    os.environ.update(environment)
    plan["environment"] = environment
    plan["affinity_applied"] = False
    if plan["affinity"] and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, plan["cpu_ids"])
        plan["affinity_applied"] = True
    config["_resource_guard"] = plan
    return plan


def configure_torch_cpu_threads(resource_guard: dict[str, Any] | None) -> None:
    """Apply the same native thread cap to PyTorch after it is imported."""
    try:
        import torch

        threads = int((resource_guard or {})["thread_pool_size"])
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(threads)
        except RuntimeError:
            # Some torch builds initialise this pool while importing vLLM.
            pass
    except (ImportError, KeyError):
        pass


def configure_vllm_environment(model: dict[str, Any]) -> dict[str, str]:
    """Apply explicit, portable vLLM runtime switches before importing vLLM."""
    configured = {str(key): str(value) for key, value in model.get("vllm_environment", {}).items()}
    os.environ.update(configured)
    return configured


class BackendFailure(RuntimeError):
    pass


@dataclass
class Response:
    """One backend answer for one prompt.

    `raw` is the model's text. `backend_error`, when set, means the request
    never produced a model answer (an outage, an HTTP failure): it is the single
    source of truth for that condition, so `raw` carries no fabricated payload
    and every consumer must let `backend_error` win over schema validation (see
    `interpret_response`). This is what keeps such a row classified
    `failed_backend` — retryable — rather than `failed_validation`.
    """

    raw: str
    token_count: int
    candidate_logprobs: dict[str, float] | None = None
    backend_error: str | None = None


class Backend(Protocol):
    """The contract every backend implements. The main path is the in-process
    vLLM one (VLLMBackend); nvidia_api and fake are the alternates."""

    def generate(self, prompts: list[str], variant: dict[str, Any]) -> list[Response]: ...

    def close(self) -> None: ...


class FakeBackend:
    def generate(self, prompts: list[str], variant: dict[str, Any]) -> list[Response]:
        if variant["request_mode"] == "candidate_logprobs":
            scores = {candidate: -float(index) for index, candidate in enumerate(variant["candidates"])}
            return [Response(json.dumps({"candidates": scores}), 1, scores) for _ in prompts]
        return [Response(json.dumps(variant.get("fake_response", {"label": "alpha"})), 1) for _ in prompts]

    def close(self) -> None:
        return None


class NvidiaAPIBackend:
    """OpenAI-compatible NVIDIA endpoint backend with bounded HTTP retries."""

    def __init__(self, model: dict[str, Any], resource_guard: dict[str, Any] | None = None) -> None:
        del resource_guard
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("nvidia_api requires the requests package.") from exc
        self.requests = requests
        self.model = model
        self.url = str(model.get("api_base_url", "https://integrate.api.nvidia.com/v1/chat/completions"))
        key_name = str(model.get("api_key_env", "NVIDIA_API_KEY"))
        self.api_key = os.environ.get(key_name)
        if not self.api_key:
            raise RuntimeError(f"nvidia_api requires the {key_name} environment variable.")
        self.timeout_seconds = float(model.get("api_timeout_seconds", 120))
        self.concurrency = max(1, int(model.get("api_concurrency", 4)))
        self.http_retries = max(0, int(model.get("api_http_retries", 2)))
        # Endpoints that reject `response_format` can opt out; the schema then
        # only reaches the model through the rendered {{output_schema}} token.
        self.structured_outputs = bool(model.get("api_structured_outputs", True))
        template_kwargs = model.get("chat_template_kwargs", {})
        if not isinstance(template_kwargs, dict):
            raise ValueError("model.chat_template_kwargs must be a mapping")
        self.chat_template_kwargs = dict(template_kwargs)

    def _generate_one(self, prompt: str, variant: dict[str, Any]) -> Response:
        payload: dict[str, Any] = {
            "model": self.model["name"],
            "messages": conversation(variant.get("_system"), prompt),
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": self.chat_template_kwargs,
        }
        if variant["request_mode"] == "candidate_logprobs":
            payload.update(
                {"max_tokens": 1, "logprobs": True, "top_logprobs": top_logprobs_count(variant["candidates"])}
            )
        else:
            payload["max_tokens"] = int(variant.get("max_tokens", 128))
            # Constrain generation server-side, as the vLLM and run-batch paths
            # already do. Without this the API path is the only one relying on
            # the prompt alone to produce schema-valid JSON.
            if (schema := variant.get("_schema")) and self.structured_outputs:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": variant["id"], "schema": schema, "strict": True},
                }
        error = "unknown NVIDIA API failure"
        for attempt in range(self.http_retries + 1):
            try:
                response = self.requests.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 200:
                    data = response.json()
                    choice = (data.get("choices") or [{}])[0]
                    raw = str((choice.get("message") or {}).get("content") or "")
                    token_count = int((data.get("usage") or {}).get("completion_tokens") or 0)
                    if variant["request_mode"] == "candidate_logprobs":
                        observed = extract_top_logprobs((choice.get("logprobs") or {}).get("content"))
                        scores = aggregate_candidate_logprobs(observed, variant["candidates"])
                        return Response(json.dumps({"candidates": scores}), token_count, scores)
                    return Response(raw, token_count)
                error = f"http_{response.status_code}: {response.text[:500]}"
            except Exception as exc:
                error = f"http_exception: {exc}"
            if attempt < self.http_retries:
                time.sleep(min(8, 2**attempt))
        # One output per input so the retry/error path keeps this row's source
        # identity. `raw` is left empty on purpose: `backend_error` is the only
        # signal of a failed request, so no fabricated JSON can later be mistaken
        # for a real (invalid) model answer and mislabelled `failed_validation`.
        return Response("", 0, None, error)

    def generate(self, prompts: list[str], variant: dict[str, Any]) -> list[Response]:
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(prompts))) as executor:
            return list(executor.map(lambda prompt: self._generate_one(prompt, variant), prompts))

    def close(self) -> None:
        return None


def make_backend(model: dict[str, Any], resource_guard: dict[str, Any] | None = None) -> Backend:
    if model.get("backend") == "fake":
        return FakeBackend()
    if model.get("backend") == "nvidia_api":
        return NvidiaAPIBackend(model, resource_guard)
    return VLLMBackend(model, resource_guard)


class VLLMBackend:
    def __init__(self, model: dict[str, Any], resource_guard: dict[str, Any] | None = None) -> None:
        if not gpu().get("available"):
            raise RuntimeError(
                "GPU preflight failed: nvidia-smi cannot communicate with an NVIDIA driver. "
                "Run `uv run python experiment-cli/experiment_cli.py gpu-preflight` for diagnostics."
            )
        # vLLM V1 starts an EngineCore process.  CUDA must never be inherited
        # through fork after the telemetry preflight has touched torch.
        import multiprocessing as mp

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        self.vllm_environment = configure_vllm_environment(model)
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "local_vllm requires vLLM, which lives in the optional `gpu` dependency group. "
                "Run `uv sync` (it installs that group by default) in an environment built with `--no-group gpu`."
            ) from exc
        configure_torch_cpu_threads(resource_guard)
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
        if "enforce_eager" in model:
            llm_kwargs["enforce_eager"] = bool(model["enforce_eager"])
        if "compilation_config" in model:
            llm_kwargs["compilation_config"] = model["compilation_config"]
        for option in (
            "tokenizer_mode",
            "config_format",
            "load_format",
            "quantization",
            "dtype",
            "tensor_parallel_size",
            "trust_remote_code",
            "model_impl",
            "mm_encoder_attn_backend",
        ):
            if model.get(option) is not None:
                llm_kwargs[option] = model[option]
        self.llm = LLM(
            **llm_kwargs,
        )
        template_kwargs = model.get("chat_template_kwargs", {})
        if not isinstance(template_kwargs, dict):
            raise ValueError("model.chat_template_kwargs must be a mapping")
        self.chat_template_kwargs = dict(template_kwargs)

    def generate(self, prompts: list[str], variant: dict[str, Any]) -> list[Response]:
        if variant["request_mode"] == "candidate_logprobs":
            params = self.params(temperature=0, max_tokens=1, logprobs=top_logprobs_count(variant["candidates"]))
        else:
            kwargs: dict[str, Any] = {"temperature": 0, "max_tokens": variant.get("max_tokens", 128)}
            if schema := variant.get("_schema"):
                from vllm.sampling_params import StructuredOutputsParams

                kwargs["structured_outputs"] = StructuredOutputsParams(
                    json=schema, disable_any_whitespace=True, disable_additional_properties=True
                )
            params = self.params(**kwargs)
        try:
            conversations = [conversation(variant.get("_system"), item) for item in prompts]
            outputs = self.llm.chat(
                conversations,
                params,
                use_tqdm=False,
                chat_template_kwargs=self.chat_template_kwargs or None,
            )
        except Exception as exc:
            if any(word in str(exc).lower() for word in ("out of memory", "oom", "context length", "max model len")):
                raise BackendFailure(str(exc)) from exc
            raise
        result: list[Response] = []
        for output in outputs:
            generated = output.outputs[0]
            if variant["request_mode"] == "candidate_logprobs":
                raw_observed = [
                    (str(getattr(logprob, "decoded_token", token)), float(getattr(logprob, "logprob", logprob)))
                    for token, logprob in ((generated.logprobs or [{}])[0] or {}).items()
                ]
                scores = aggregate_candidate_logprobs(raw_observed, variant["candidates"])
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
    backend: Backend, variant: dict[str, Any], prompts: list[str], batch: dict[str, Any], events: Events, sync: bool
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
    """Identify the reusable static prefix shared by rows in a variant.

    The system turn is part of that prefix, so two variants differing only in
    their system prompt must not share a group id.
    """
    sentinel = {config["input"]["id_column"]: "<row>", config["input"]["text_column"]: "<text>"}
    static = rendered_prompt(config, variant, sentinel, schema)
    system = system_prompt(config, variant, schema)
    # A variant without a system prompt must hash exactly as it always has.
    material = f"{system}\n\n{static}" if system else static
    return hashlib.sha256(material.replace("<text>", "{{text}}").encode()).hexdigest()[:16]


def variant_config_hash(config: dict[str, Any], variant: dict[str, Any]) -> str:
    assets = {}
    for path in list(variant.get("prompts", [])) + system_prompt_paths(variant):
        assets[str(path)] = read_asset(config, path)
    for name, path in config.get("run", {}).get("prompt_parts", {}).items():
        assets[f"part:{name}"] = read_asset(config, path)
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
        # Keys of rows an earlier run left in `failed_backend`. They are held in
        # memory rather than SQLite because they are bounded by the number of
        # failures, not the number of rows, and merge_parts needs them to drop
        # the superseded copy once the row is re-attempted.
        self.retryable_keys: set[tuple[str, str, int]] = set()
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
            if "final_status" in parquet.schema.names:
                columns.append("final_status")
            for batch in parquet.iter_batches(columns=columns):
                records = batch.to_pylist()
                if expected_hashes:
                    records = [
                        row for row in records if row.get("config_hash") == expected_hashes.get(str(row["variant_id"]))
                    ]
                done = [row for row in records if row.get("final_status") != "failed_backend"]
                self.retryable_keys.update(
                    (str(row["variant_id"]), str(row["input_row_id"]), saved_position(row))
                    for row in records
                    if row.get("final_status") == "failed_backend"
                )
                self.connection.executemany(
                    "INSERT OR IGNORE INTO completed VALUES (?, ?, ?)",
                    [(str(row["variant_id"]), str(row["input_row_id"]), saved_position(row)) for row in done],
                )
                count += len(done)
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

    def append(self, row: dict[str, Any]) -> bool:
        """Buffer a row and report whether this call durably published a part."""
        self.rows.append(row)
        if len(self.rows) >= self.target_rows:
            return self.flush()
        return False

    def flush(self) -> bool:
        """Atomically publish buffered rows to the shared result filesystem."""
        if not self.rows:
            return False
        path = self.directory / f"part-{self.index:05d}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(self.rows, schema=RESULT_SCHEMA), temporary, compression="zstd")
        temporary.replace(path)
        self.index += 1
        self.rows.clear()
        return True

    def close(self) -> None:
        self.flush()


def _part_keys(table: pa.Table) -> list[tuple[str, str, int]]:
    columns = zip(
        table["variant_id"].to_pylist(), table["input_row_id"].to_pylist(), table["source_position"].to_pylist()
    )
    return [(str(variant), str(row_id), int(position)) for variant, row_id, position in columns]


def _last_occurrence(files: list[Path], keys: set[tuple[str, str, int]]) -> dict[tuple[str, str, int], tuple[int, int]]:
    """Locate the newest row written for each of `keys`, as (file index, row index).

    Parts are append-only and sorted by name, so the last occurrence is the most
    recent attempt. Only three columns are read and only `keys` is retained, so
    the cost is bounded by the number of retried rows rather than the run size.
    """
    latest: dict[tuple[str, str, int], tuple[int, int]] = {}
    for file_index, path in enumerate(files):
        table = pq.read_table(path, columns=["variant_id", "input_row_id", "source_position"])
        for row_index, key in enumerate(_part_keys(table)):
            if key in keys:
                latest[key] = (file_index, row_index)
    return latest


def _supersede_mask(
    table: pa.Table, surviving: dict[tuple[str, str, int], tuple[int, int]], file_index: int
) -> list[bool]:
    """Keep every row except earlier attempts at a key that was retried."""
    return [
        surviving.get(key, (file_index, row_index)) == (file_index, row_index)
        for row_index, key in enumerate(_part_keys(table))
    ]


def merge_parts(
    run_dir: Path,
    expected_hashes: dict[str, str] | None = None,
    retried_keys: set[tuple[str, str, int]] | None = None,
) -> int:
    """Create compatibility result files from append-only variant parts.

    `retried_keys` are rows a previous run left as `failed_backend` and this run
    re-attempted. Parts are append-only, so both attempts are on disk; only the
    newest survives the merge. When nothing was retried this costs nothing.
    """
    files = sorted((run_dir / "parts").glob("variant=*/part-*.parquet")) if (run_dir / "parts").exists() else []
    if not files:
        return 0
    surviving = _last_occurrence(files, retried_keys) if retried_keys else {}
    writer: pq.ParquetWriter | None = None
    variant_writers: dict[str, pq.ParquetWriter] = {}
    count = 0
    try:
        for file_index, path in enumerate(files):
            table = pq.read_table(path)
            if surviving:
                table = table.filter(_supersede_mask(table, surviving, file_index))
                if table.num_rows == 0:
                    continue
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


def make_events(config: dict[str, Any], run_dir: Path, run_id: str) -> Events:
    """Build the event log for a run.

    Matrix lanes and `--set` overrides must never share the YAML's default log
    names, so once run.id or output.directory is overridden the paths are
    derived from the effective output directory instead.
    """
    logging_config = config.get("logging", {})
    if any(key in config.get("_override_keys", []) for key in ("run.id", "output.directory")):
        logging_config = dict(logging_config)
        logging_config["file"] = str(run_dir / "logs" / f"{run_id}.log")
        logging_config["events"] = str(run_dir / "logs" / f"{run_id}.events.jsonl")
    return Events(
        resolve(config, logging_config.get("file", f"logs/{run_id}.log")),
        resolve(config, logging_config.get("events", f"logs/{run_id}.events.jsonl")),
        logging_config.get("level", "INFO"),
    )


def error_kind(errors: list[str], response: Response, max_tokens: int) -> str:
    """Classify why a response failed validation."""
    if response.token_count >= int(max_tokens or 0):
        return "output_length_limit"
    if any(error.startswith("json_parse_error:") for error in errors):
        return "format"
    return "schema"


def generate_with_backoff(
    backend: Backend,
    request_variant: dict[str, Any],
    prompts: list[str],
    batch_config: dict[str, Any],
    on_backoff: Any | None = None,
) -> tuple[list[Response], int]:
    """Generate, halving the batch until the backend stops failing.

    Returns the responses and how many prompts they cover, so the caller can
    truncate its own chunk to match. Raises once the batch cannot shrink
    further: an OOM at the minimum size is a real failure, not a hiccup.
    """
    minimum = int(batch_config.get("min_size", 1))
    while True:
        try:
            responses = backend.generate(prompts, request_variant)
            if len(responses) != len(prompts):
                raise BackendFailure(f"Backend returned {len(responses)} responses for {len(prompts)} prompts")
            return responses, len(prompts)
        except BackendFailure as exc:
            if len(prompts) <= minimum:
                raise
            new_size = max(minimum, len(prompts) // 2)
            if on_backoff is not None:
                on_backoff(exc, len(prompts), new_size)
            prompts = prompts[:new_size]


BACKEND_ERROR_PREFIX = "backend_error:"


def failure_status(errors: list[str]) -> str:
    """Tell a broken backend apart from a badly behaved model.

    The distinction is what makes resume safe to retry: a `backend_error` is
    the infrastructure failing (an outage, a dropped connection) and is worth
    re-attempting, whereas a schema violation is the model answering
    deterministically and would fail again identically on every resume.
    """
    return "failed_backend" if any(item.startswith(BACKEND_ERROR_PREFIX) for item in errors) else "failed_validation"


def interpret_response(
    response: Response, schema: dict[str, Any] | None, request_mode: str
) -> tuple[Any | None, list[str]]:
    """Turn a backend `Response` into `(parsed, errors)`.

    A `backend_error` always wins over schema validation and over logprob
    extraction: it means the request never reached the model, so `response.raw`
    is not a model answer and must not be judged as one. Every request path — the
    main loop, the inline retry, the deferred retry — must decode responses
    through here, or a retry that hits a fresh outage gets mislabelled
    `failed_validation` and is then never re-attempted on resume.
    """
    if response.backend_error:
        return None, [f"{BACKEND_ERROR_PREFIX} {response.backend_error}"]
    if request_mode == "candidate_logprobs":
        return {"candidates": response.candidate_logprobs or {}}, []
    return validate_response(response.raw, schema)


def result_row(
    config: dict[str, Any],
    *,
    run_id: str,
    variant_id: str,
    config_hash: str,
    group_id: str | None,
    row: dict[str, Any],
    prompt_text: str,
    raw: str | None,
    parsed: Any | None,
    errors: list[str],
    attempt_count: int,
    token_count: int | None = None,
    batch_size: int | None = None,
    latency_seconds: float | None = None,
    rows_per_second: float | None = None,
    gpu_snapshot: str | None = None,
    candidate_logprobs: dict[str, float] | None = None,
    final_status: str | None = None,
) -> dict[str, Any]:
    """Build one row of the Parquet result contract.

    Every write path goes through here so the contract stays defined in exactly
    one place alongside RESULT_SCHEMA.
    """
    source = config["input"]
    return {
        "run_id": run_id,
        "dataset_id": config["run"].get("dataset_id", "default"),
        "variant_id": variant_id,
        "input_row_id": str(row[source["id_column"]]),
        "source_position": row["_source_position"],
        "input_text": str(row[source["text_column"]]) if config["output"].get("include_text") else None,
        "gold_labels": serialise(row.get("_gold_labels")),
        "prompt_hash": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "config_hash": config_hash,
        "prompt_group_id": group_id,
        "attempt_count": attempt_count,
        "raw_response": raw if config["output"].get("include_raw_response", True) else None,
        "parsed_output": serialise(parsed),
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": serialise(errors),
        "final_status": final_status or ("completed" if not errors else failure_status(errors)),
        "batch_size": batch_size,
        "latency_seconds": latency_seconds,
        "rows_per_second": rows_per_second,
        "token_count": token_count,
        "gpu_snapshot": gpu_snapshot,
        "candidate_logprobs": serialise(candidate_logprobs),
    }


def discard_stale_results(run_dir: Path, config: dict[str, Any]) -> None:
    """Remove results that a changed configuration has invalidated."""
    import shutil

    if (run_dir / "parts").exists():
        shutil.rmtree(run_dir / "parts")
    for path in [run_dir / "results.parquet"] + [run_dir / f"{item['id']}.parquet" for item in config["variants"]]:
        if path.exists():
            path.unlink()


def _counted(rows: Any, counter: list[int]) -> Any:
    """Count rows as they are pulled, so a stream never needs a second pass."""
    for row in rows:
        counter[0] += 1
        yield row


class ErrorLog:
    """Append-only per-variant error records, holding the handle open."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def write(self, **record: Any) -> None:
        self.handle.write(json.dumps({"timestamp": time.time(), **record}, ensure_ascii=False, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def run(config: dict[str, Any], backend: Backend | None = None, row_limit: int | None = None) -> dict[str, Any]:
    """Execute every variant of one dataset and write the Parquet result contract.

    Rows always flow through the same bounded chunk loop and land in append-only
    Parquet parts. `streaming.enabled` only decides whether the source is
    materialised first — which lets the manifest report exact input and pending
    counts — or consumed as a true stream, which a multi-million-row CSV needs.
    """
    if backend is None and config["model"].get("backend") == "local_vllm" and "_resource_guard" not in config:
        apply_resource_guard(config)
    streaming = bool(config.get("streaming", {}).get("enabled", False))
    run_id = config["run"]["id"]
    run_dir = resolve(config, config["output"]["directory"])
    run_dir.mkdir(parents=True, exist_ok=True)
    events = make_events(config, run_dir, run_id)

    expected_hashes = {
        str(variant["id"]): variant_config_hash(config, materialize_variant(config, variant))
        for variant in config["variants"]
    }
    index = ResumeIndex(
        run_dir / ".resume.sqlite",
        hashlib.sha256(json.dumps(expected_hashes, sort_keys=True).encode()).hexdigest(),
    )
    if index.cleared:
        discard_stale_results(run_dir, config)
    part_files = list((run_dir / "parts").glob("variant=*/part-*.parquet")) if (run_dir / "parts").exists() else []
    seeded = index.seed_from(
        part_files or ([run_dir / "results.parquet"] if (run_dir / "results.parquet").exists() else []),
        expected_hashes,
    )

    # A materialised source knows its size up front; a stream is counted as it
    # is consumed, which is why neither path needs a second read of the input.
    materialised = None if streaming else rows_for_source(config, config["input"], row_limit)
    pulled = [0]
    total_input = len(materialised) if materialised is not None else 0
    # Every variant re-reads the source, so only the first pass is counted.
    input_counted = materialised is not None
    total_results = seeded
    selected: dict[str, Any] = {}
    retry_settings = config.get("validation", {}).get("retry", {})
    correction_path = retry_settings.get("correction_prompt")
    correction = read_asset(config, correction_path) if correction_path else None
    created = False

    try:
        for configured_variant in config["variants"]:
            variant = materialize_variant(config, configured_variant)
            variant_id = str(variant["id"])
            config_hash = expected_hashes[variant_id]
            schema = variant_schema(config, variant)
            request_variant = {**variant, "_schema": schema, "_system": system_prompt(config, variant, schema)}
            max_tokens = int(request_variant.get("max_tokens", 128))
            allow_retry = bool(correction and retry_settings.get("enabled")) and (
                request_variant["request_mode"] != "candidate_logprobs"
            )

            if materialised is not None:
                source_iter: Any = iter(materialised)
            else:
                source_iter = iter_rows_for_source(config, config["input"], row_limit)
                if not input_counted:
                    source_iter = _counted(source_iter, pulled)

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

            pending_rows = None
            if materialised is not None:
                pending_rows = sum(
                    1
                    for row in materialised
                    if not index.contains(
                        (variant_id, str(row[config["input"]["id_column"]]), int(row["_source_position"]))
                    )
                )
                if pending_rows == 0:
                    events.emit("variant_resumed", variant=variant_id, skipped=len(materialised))
                    continue

            if backend is None:
                backend = make_backend(config["model"], config.get("_resource_guard"))
                created = True

            tune_prompts = [rendered_prompt(config, variant, row, schema) for row in prefetched]
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
            selected[variant_id] = {"selected_batch_size": size, "prompt_group_id": group_id, "tuning": attempts}
            if pending_rows is not None:
                selected[variant_id]["pending_rows"] = pending_rows
            events.emit("variant_started", variant=variant_id, prompt_group_id=group_id, batch_size=size)

            writer = PartWriter(run_dir, variant_id, int(config.get("streaming", {}).get("output_chunk_rows", 4096)))
            errors_log = ErrorLog(run_dir / "errors" / f"variant={variant_id}.jsonl")
            deferred_retries: list[dict[str, Any]] = []
            pending_index_keys: list[tuple[str, str, int]] = []

            def record_error(
                *,
                stage: str,
                row: dict[str, Any],
                response: Response,
                raw: str,
                errors: list[str],
                attempt_count: int,
                batch_size: int,
                max_tokens: int,
                variant_id: str = variant_id,
                errors_log: ErrorLog = errors_log,
            ) -> None:
                errors_log.write(
                    run_id=run_id,
                    dataset_id=config["run"].get("dataset_id", "default"),
                    variant_id=variant_id,
                    input_row_id=str(row[config["input"]["id_column"]]),
                    source_position=int(row["_source_position"]),
                    stage=stage,
                    error_kind=error_kind(errors, response, max_tokens),
                    validation_errors=errors,
                    attempt_count=attempt_count,
                    batch_size=batch_size,
                    max_tokens=max_tokens,
                    token_count=response.token_count,
                    raw_response=raw,
                )

            def append_output(
                output: dict[str, Any],
                variant_id: str = variant_id,
                writer: PartWriter = writer,
                pending_index_keys: list[tuple[str, str, int]] = pending_index_keys,
            ) -> None:
                """Publish a part before marking any of its rows resumable.

                A row the backend never answered is deliberately left out of the
                index: the index outlives the run, so marking it here would make
                the next resume treat an outage as settled work.
                """
                if output["final_status"] != "failed_backend":
                    pending_index_keys.append((variant_id, str(output["input_row_id"]), int(output["source_position"])))
                if writer.append(output):
                    for key in pending_index_keys:
                        index.add(key)
                    pending_index_keys.clear()

            def on_backoff(
                exc: BackendFailure,
                old_size: int,
                new_size: int,
                *,
                stage: str = "initial_batch_backoff",
                variant_id: str = variant_id,
                errors_log: ErrorLog = errors_log,
            ) -> None:
                events.emit(
                    "batch_runtime_backoff",
                    variant=variant_id,
                    old_batch_size=old_size,
                    new_batch_size=new_size,
                    error=str(exc),
                )
                errors_log.write(
                    run_id=run_id,
                    dataset_id=config["run"].get("dataset_id", "default"),
                    variant_id=variant_id,
                    stage=stage,
                    error_kind="backend_batch_failure",
                    error=str(exc),
                    failed_batch_size=old_size,
                    new_batch_size=new_size,
                )

            pending_rows_iter = itertools.chain(prefetched, source_iter)
            buffer: list[dict[str, Any]] = []
            while True:
                while len(buffer) < size:
                    try:
                        row = next(pending_rows_iter)
                    except StopIteration:
                        break
                    key = (variant_id, str(row[config["input"]["id_column"]]), int(row["_source_position"]))
                    if not index.contains(key):
                        buffer.append(row)
                if not buffer:
                    break
                chunk = buffer[:size]
                prompts = [rendered_prompt(config, variant, row, schema) for row in chunk]
                sync_cuda(bool(config["model"].get("synchronize_cuda", False)))
                started = time.perf_counter()
                responses, used = generate_with_backoff(backend, request_variant, prompts, batch_config, on_backoff)
                if used < len(chunk):
                    size = used
                    selected[variant_id]["runtime_batch_size"] = size
                    chunk = chunk[:used]
                sync_cuda(bool(config["model"].get("synchronize_cuda", False)))
                elapsed = max(time.perf_counter() - started, 1e-9)
                snapshot = serialise(gpu(max_age_seconds=5.0))

                for row, current_prompt, response in zip(chunk, prompts, responses):
                    raw = response.raw
                    attempt_count = 1
                    parsed, errors = interpret_response(response, schema, request_variant["request_mode"])
                    if response.backend_error:
                        record_error(
                            stage="backend_response",
                            row=row,
                            response=response,
                            raw=raw,
                            errors=errors,
                            attempt_count=attempt_count,
                            batch_size=size,
                            max_tokens=max_tokens,
                        )

                    if errors and allow_retry and retry_settings.get("deferred", False):
                        record_error(
                            stage="initial_validation",
                            row=row,
                            response=response,
                            raw=raw,
                            errors=errors,
                            attempt_count=attempt_count,
                            batch_size=size,
                            max_tokens=max_tokens,
                        )
                        deferred_retries.append(
                            {
                                "row": row,
                                "prompt": current_prompt,
                                "response": response,
                                "raw": raw,
                                "errors": errors,
                                "attempt_count": attempt_count,
                            }
                        )
                        continue

                    if errors and allow_retry:
                        events.emit(
                            "retry_started", variant=variant_id, input_row_id=str(row[config["input"]["id_column"]])
                        )
                        for _ in range(int(retry_settings.get("max_attempts", 0))):
                            retry_prompt = render(correction, retry_values(config, variant, row, schema, raw, errors))
                            retry_response = backend.generate([retry_prompt], request_variant)[0]
                            raw = retry_response.raw
                            parsed, errors = interpret_response(retry_response, schema, request_variant["request_mode"])
                            attempt_count += 1
                            if not errors:
                                break
                        events.emit(
                            "retry_completed",
                            variant=variant_id,
                            input_row_id=str(row[config["input"]["id_column"]]),
                            attempts=attempt_count,
                            validation_status="valid" if not errors else "invalid",
                        )

                    append_output(
                        result_row(
                            config,
                            run_id=run_id,
                            variant_id=variant_id,
                            config_hash=config_hash,
                            group_id=group_id,
                            row=row,
                            prompt_text=current_prompt,
                            raw=raw,
                            parsed=parsed,
                            errors=errors,
                            attempt_count=attempt_count,
                            token_count=response.token_count,
                            batch_size=size,
                            latency_seconds=elapsed / len(chunk),
                            rows_per_second=len(chunk) / elapsed,
                            gpu_snapshot=snapshot,
                            candidate_logprobs=response.candidate_logprobs,
                        )
                    )
                    total_results += 1
                index.connection.commit()
                events.emit(
                    "batch_completed", variant=variant_id, rows=len(chunk), rows_per_second=len(chunk) / elapsed
                )
                buffer = buffer[len(chunk) :]

            if not input_counted:
                # The loop above drains the source, so this pass saw every row.
                total_input = pulled[0]
                input_counted = True

            total_results += run_deferred_retries(
                config,
                deferred_retries,
                backend=backend,
                events=events,
                index=index,
                errors_log=errors_log,
                append_output=append_output,
                request_variant=request_variant,
                schema=schema,
                variant=variant,
                correction=correction,
                batch_config=batch_config,
                size=size,
                run_id=run_id,
                config_hash=config_hash,
                group_id=group_id,
            )

            if writer.flush():
                for key in pending_index_keys:
                    index.add(key)
                pending_index_keys.clear()
            index.connection.commit()
            writer.close()
            errors_log.close()
            events.emit("variant_completed", variant=variant_id)
        merged = merge_parts(run_dir, expected_hashes, index.retryable_keys)
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
        "cpu_resource_guard": config.get("_resource_guard"),
        "gpu_preflight": gpu(),
        "source_provenance": source_provenance(config),
        "resume_skipped_rows": seeded,
        "event_log": str(events.path),
    }
    if streaming:
        manifest["streaming"] = True
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    events.emit("run_completed", result_rows=manifest["result_rows"], gpu=gpu())
    return manifest


def run_deferred_retries(
    config: dict[str, Any],
    queued: list[dict[str, Any]],
    *,
    backend: Backend,
    events: Events,
    index: ResumeIndex,
    errors_log: ErrorLog,
    append_output: Any,
    request_variant: dict[str, Any],
    schema: dict[str, Any] | None,
    variant: dict[str, Any],
    correction: str | None,
    batch_config: dict[str, Any],
    size: int,
    run_id: str,
    config_hash: str,
    group_id: str,
) -> int:
    """Re-ask the model for rows that failed validation, in shrinking batches.

    Deferring these to the end of a variant keeps the main loop at full batch
    size; retries then run smaller and with a larger token budget, because the
    usual cause of failure is a truncated structured response.
    """
    if not queued:
        return 0
    variant_id = str(variant["id"])
    retry_settings = config.get("validation", {}).get("retry", {})
    max_attempts = int(retry_settings.get("max_attempts", 0))
    retry_divisor = max(2, int(retry_settings.get("batch_size_divisor", 2)))
    token_multiplier = max(1, int(retry_settings.get("max_tokens_multiplier", 2)))
    token_cap = int(retry_settings.get("max_tokens_cap", 256))
    retry_batch = max(int(batch_config.get("min_size", 1)), size // retry_divisor)
    written = 0

    for retry_round in range(1, max_attempts + 1):
        if not queued:
            break
        retry_max_tokens = min(token_cap, int(request_variant.get("max_tokens", 128)) * token_multiplier**retry_round)
        retry_variant = {**request_variant, "max_tokens": retry_max_tokens}
        events.emit(
            "deferred_retry_started",
            variant=variant_id,
            retry_round=retry_round,
            rows=len(queued),
            batch_size=retry_batch,
            max_tokens=retry_max_tokens,
        )

        def on_backoff(exc: BackendFailure, old_size: int, new_size: int, *, retry_round: int = retry_round) -> None:
            nonlocal retry_batch
            retry_batch = new_size
            events.emit(
                "deferred_retry_backoff",
                variant=variant_id,
                retry_round=retry_round,
                new_batch_size=new_size,
                error=str(exc),
            )
            errors_log.write(
                run_id=run_id,
                dataset_id=config["run"].get("dataset_id", "default"),
                variant_id=variant_id,
                stage="deferred_retry_backoff",
                error_kind="backend_batch_failure",
                error=str(exc),
                retry_round=retry_round,
                failed_batch_size=old_size,
                new_batch_size=new_size,
            )

        next_queue: list[dict[str, Any]] = []
        offset = 0
        while offset < len(queued):
            retry_chunk = queued[offset : offset + retry_batch]
            retry_prompts = [
                render(correction, retry_values(config, variant, item["row"], schema, item["raw"], item["errors"]))
                for item in retry_chunk
            ]
            started = time.perf_counter()
            retry_responses, used = generate_with_backoff(
                backend, retry_variant, retry_prompts, batch_config, on_backoff
            )
            retry_chunk = retry_chunk[:used]
            elapsed = max(time.perf_counter() - started, 1e-9)
            snapshot = serialise(gpu(max_age_seconds=5.0))

            for item, response in zip(retry_chunk, retry_responses):
                row = item["row"]
                raw = response.raw
                parsed, errors = interpret_response(response, schema, request_variant["request_mode"])
                attempt_count = int(item["attempt_count"]) + 1
                if errors:
                    errors_log.write(
                        run_id=run_id,
                        dataset_id=config["run"].get("dataset_id", "default"),
                        variant_id=variant_id,
                        input_row_id=str(row[config["input"]["id_column"]]),
                        source_position=int(row["_source_position"]),
                        stage="deferred_validation",
                        error_kind=error_kind(errors, response, retry_max_tokens),
                        validation_errors=errors,
                        attempt_count=attempt_count,
                        batch_size=len(retry_chunk),
                        max_tokens=retry_max_tokens,
                        token_count=response.token_count,
                        raw_response=raw,
                    )
                    if retry_round < max_attempts:
                        next_queue.append(
                            {**item, "response": response, "raw": raw, "errors": errors, "attempt_count": attempt_count}
                        )
                        continue
                append_output(
                    result_row(
                        config,
                        run_id=run_id,
                        variant_id=variant_id,
                        config_hash=config_hash,
                        group_id=group_id,
                        row=row,
                        prompt_text=item["prompt"],
                        raw=raw,
                        parsed=parsed,
                        errors=errors,
                        attempt_count=attempt_count,
                        token_count=response.token_count,
                        batch_size=len(retry_chunk),
                        latency_seconds=elapsed / len(retry_chunk),
                        rows_per_second=len(retry_chunk) / elapsed,
                        gpu_snapshot=snapshot,
                        candidate_logprobs=response.candidate_logprobs,
                    )
                )
                written += 1
            index.connection.commit()
            offset += len(retry_chunk)
        queued = next_queue
        events.emit("deferred_retry_completed", variant=variant_id, retry_round=retry_round, remaining=len(queued))
    return written


def run_matrix(
    config: dict[str, Any], row_limit: int | None = None, selected: list[str] | None = None
) -> dict[str, Any]:
    """Run every configured dataset with one shared backend/model process."""
    if "datasets" not in config:
        return {"datasets": [run(config, row_limit=row_limit)]}
    if config["model"].get("backend") == "local_vllm" and "_resource_guard" not in config:
        apply_resource_guard(config)
    base_output = resolve(config, config["output"]["directory"])
    entries = selected_entries(config, selected)
    shared = make_backend(config["model"], config.get("_resource_guard"))
    manifests: list[dict[str, Any]] = []
    try:
        for dataset_id, source in entries:
            lane = dataset_config(config, dataset_id, source, base_output)
            # Matrix workers must never share the YAML's default log names;
            # derive them from the effective output directory and run id.
            lane["logging"] = {
                **config.get("logging", {}),
                "file": str(base_output / "logs" / f"{dataset_id}.log"),
                "events": str(base_output / "logs" / f"{dataset_id}.events.jsonl"),
            }
            manifests.append(run(lane, shared, row_limit=row_limit))
    finally:
        shared.close()
    summary = {
        "run_id": config["run"]["id"],
        "model": config["model"],
        "cpu_resource_guard": config.get("_resource_guard"),
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
    base_output = resolve(config, config["output"]["directory"])
    return [
        prepare(dataset_config(config, dataset_id, source, base_output))
        for dataset_id, source in selected_entries(config, selected)
    ]


def variant_schema(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any] | None:
    schema_path = variant.get("validation", {}).get("schema")
    schema = json.loads(read_asset(config, schema_path)) if schema_path else None
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
    """Resolve `candidates_from` into a concrete candidate list.

    A shallow copy is enough and is deliberate: this sits in the per-row prompt
    path, where a deepcopy of every variant was pure overhead. Nothing mutates
    the nested config, so sharing it is safe.
    """
    source = variant.get("candidates_from")
    if source == "dataset_labels":
        return {**variant, "candidates": list(config.get("run", {}).get("dataset_labels", []))}
    if source == "code_labels":
        return {**variant, "candidates": list(config.get("run", {}).get("code_labels", {}))}
    return dict(variant)


def prompt_values(
    config: dict[str, Any], variant: dict[str, Any], row: dict[str, Any], schema: dict[str, Any] | None
) -> dict[str, Any]:
    """Build the placeholder values that every prompt template may reference.

    `variant` must already be materialised, so `candidates` reflects any
    `candidates_from` indirection.
    """
    run_config = config.get("run", {})
    candidates = ", ".join(str(item) for item in variant.get("candidates", []))
    return {
        "text": row[config["input"]["text_column"]],
        "row_id": row[config["input"]["id_column"]],
        "dataset_id": run_config.get("dataset_id", "default"),
        "candidates": candidates,
        "labels": ", ".join(str(item) for item in run_config.get("dataset_labels", [])),
        "candidate_mapping": ", ".join(f"{code}={label}" for code, label in run_config.get("code_labels", {}).items())
        or candidates,
        "question": run_config.get("binary_question", "Does this text express the target value?"),
        "output_schema": json.dumps(schema or {}, sort_keys=True),
    }


def rendered_prompt(
    config: dict[str, Any], variant: dict[str, Any], row: dict[str, Any], schema: dict[str, Any] | None = None
) -> str:
    variant = materialize_variant(config, variant)
    if schema is None and variant.get("validation", {}).get("schema"):
        schema = variant_schema(config, variant)
    values = prompt_values(config, variant, row, schema)
    values.update(prompt_part_values(config, values))
    return prompt(config, variant["prompts"], values)


ROW_SPECIFIC_TOKENS = ("text", "row_id")


def system_prompt_paths(variant: dict[str, Any]) -> list[str]:
    """Normalise `system_prompt` to a list; a bare string is allowed for one file."""
    declared = variant.get("system_prompt")
    if not declared:
        return []
    return [declared] if isinstance(declared, str) else list(declared)


def system_prompt(config: dict[str, Any], variant: dict[str, Any], schema: dict[str, Any] | None) -> str | None:
    """Render a variant's system turn, or None when it declares no system prompt.

    The system turn carries instructions, not data, so it is rendered once per
    variant rather than once per row. Referencing a row placeholder is therefore
    rejected outright: it would silently render empty for every row rather than
    fail, which is the kind of prompt bug that only shows up in the results.
    """
    paths = system_prompt_paths(variant)
    if not paths:
        return None
    for path in paths:
        raw = read_asset(config, path)
        for token in ROW_SPECIFIC_TOKENS:
            if f"{{{{{token}}}}}" in raw:
                raise ValueError(
                    f"{variant['id']}: system_prompt may not use {{{{{token}}}}} ({path}). "
                    "It is rendered once per variant; put row placeholders in `prompts`."
                )
    source = config["input"]
    blank_row = {source["id_column"]: "", source["text_column"]: ""}
    values = prompt_values(config, variant, blank_row, schema)
    values.update(prompt_part_values(config, values))
    return prompt(config, paths, values)


def conversation(system: str | None, user: str) -> list[dict[str, str]]:
    """Build the chat turns for one prompt, with the system turn when present."""
    turns = [{"role": "system", "content": system}] if system else []
    return turns + [{"role": "user", "content": user}]


def retry_values(
    config: dict[str, Any],
    variant: dict[str, Any],
    row: dict[str, Any],
    schema: dict[str, Any] | None,
    raw: str,
    errors: list[str],
) -> dict[str, Any]:
    variant = materialize_variant(config, variant)
    values = prompt_values(config, variant, row, schema)
    values["raw_response"] = raw
    values["validation_errors"] = "; ".join(errors)
    values.update(prompt_part_values(config, values))
    return values


def request_for_row(
    config: dict[str, Any], variant: dict[str, Any], row: dict[str, Any], schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    variant = materialize_variant(config, variant)
    content = rendered_prompt(config, variant, row, schema)
    body: dict[str, Any] = {
        "model": config["model"]["name"],
        "messages": conversation(system_prompt(config, variant, schema), content),
        "temperature": 0,
    }
    if variant["request_mode"] == "candidate_logprobs":
        body.update(
            {
                "max_completion_tokens": 1,
                "logprobs": True,
                "top_logprobs": top_logprobs_count(variant["candidates"]),
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


def _batch_text(item: dict[str, Any]) -> tuple[str | None, str | None, list[tuple[str, float]] | None]:
    response = item.get("response") or {}
    if item.get("error") or response.get("status_code", 200) != 200:
        return None, str(item.get("error") or response.get("status_code", "batch_response_error")), None
    choice = ((response.get("body") or {}).get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    if content is None:
        return None, "missing_chat_completion_content", None
    raw_observed = extract_top_logprobs((choice.get("logprobs") or {}).get("content"))
    return str(content), None, raw_observed or None


def _response_key(custom_id: str) -> tuple[tuple[str, str, int], int] | None:
    """Split a retry `custom_id` into its result key and attempt number.

    Retry ids are `retry:<variant>:<row_id>:<position>:<attempt>`. Parsing from
    the outside in keeps row ids that themselves contain a colon intact; only
    the variant id is assumed colon-free, which is what building the id assumed
    in the first place. Returns None for ordinary (non-retry) ids.
    """
    if not custom_id.startswith("retry:"):
        return None
    body, attempt = custom_id.rsplit(":", 1)
    variant_id, remainder = body[len("retry:") :].split(":", 1)
    row_id, position = remainder.rsplit(":", 1)
    return (variant_id, row_id, int(position)), int(attempt)


def parse_batch(config: dict[str, Any], response_path: str | Path) -> dict[str, Any]:
    """Fold externally produced `vllm run-batch` responses into the result contract."""
    source = Path(response_path)
    if source.is_dir():
        source = source / "responses.jsonl"

    # Index the responses once. Scanning them per row instead is quadratic, and
    # the ProtoEthos lane parses millions of rows.
    responses: dict[str, dict[str, Any]] = {}
    retries: dict[tuple[str, str, int], list[tuple[int, dict[str, Any]]]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = str(item["custom_id"])
        if custom_id in responses:
            raise ValueError(f"Duplicate custom_id in batch response: {custom_id}")
        responses[custom_id] = item
        parsed_key = _response_key(custom_id)
        if parsed_key is not None:
            key, attempt = parsed_key
            retries.setdefault(key, []).append((attempt, item))

    run_id = config["run"]["id"]
    run_dir = resolve(config, config["output"]["directory"])
    events = make_events(config, run_dir, run_id)
    rows = rows_for_source(config, config["input"])
    result_path = run_dir / "results.parquet"
    expected_hashes = {
        str(item["id"]): variant_config_hash(config, materialize_variant(config, item)) for item in config["variants"]
    }
    saved = [
        row
        for row in (pq.read_table(result_path).to_pylist() if result_path.exists() else [])
        if row.get("config_hash") == expected_hashes.get(str(row["variant_id"]))
    ]
    # Last write wins, matching the original reverse-order search for a row to update.
    saved_by_key = {(str(row["variant_id"]), str(row["input_row_id"]), saved_position(row)): row for row in saved}
    complete = set(saved_by_key)
    retry_settings = config.get("validation", {}).get("retry", {})
    correction_path = retry_settings.get("correction_prompt")
    correction = read_asset(config, correction_path) if correction_path else None
    retry_pending = bool(
        correction and retry_settings.get("enabled") and int(retry_settings.get("max_attempts", 0)) >= 1
    )
    retry_requests: list[dict[str, Any]] = []

    def add_retry_request(
        variant: dict[str, Any],
        schema: dict[str, Any] | None,
        row: dict[str, Any],
        raw: str,
        errors: list[str],
        attempt: int,
    ) -> None:
        """Queue one bounded retry request for a row that failed validation."""
        # A row that validated has nothing to correct. Without this the retry
        # file holds every parsed row, and re-running the batch re-asks the
        # model questions it already answered correctly.
        if not errors:
            return
        if not correction or not retry_settings.get("enabled") or attempt > int(retry_settings.get("max_attempts", 0)):
            return
        retry_prompt = render(correction, retry_values(config, variant, row, schema, raw, errors))
        body: dict[str, Any] = {
            "model": config["model"]["name"],
            "messages": conversation(system_prompt(config, variant, schema), retry_prompt),
            "temperature": 0,
            "max_completion_tokens": variant.get("max_tokens", 128),
        }
        if variant["request_mode"] == "candidate_logprobs":
            body.update(
                {
                    "max_completion_tokens": 1,
                    "logprobs": True,
                    "top_logprobs": top_logprobs_count(variant.get("candidates", [])),
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
        variant_id = str(variant["id"])
        config_hash = expected_hashes[variant_id]
        # This must be the same schema `prepare` sent to the batch server: it is
        # what constrained the model, what the {{output_schema}} token rendered,
        # and therefore what the response can legitimately be judged against.
        schema = variant_schema(config, variant)
        group_id = prompt_group_id(config, variant, schema)
        for row in rows:
            row_id = str(row[config["input"]["id_column"]])
            key = (variant_id, *row_key(row, config))
            retry_items = retries.get(key, [])
            if key in complete and not retry_items:
                continue
            current_prompt = rendered_prompt(config, variant, row, schema)
            custom_id = f"{variant_id}:{row_id}:{row['_source_position']}"
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
                scores = aggregate_candidate_logprobs(observed, variant["candidates"])
                parsed, errors = {"candidates": scores}, []
            attempt_count = 1

            if retry_items:
                attempt_count, latest = max(retry_items, key=lambda pair: pair[0])
                retry_raw, retry_error, retry_observed = _batch_text(latest)
                if retry_raw is not None:
                    raw = retry_raw
                    parsed, errors = validate_response(raw, schema)
                    if variant["request_mode"] == "candidate_logprobs" and retry_observed is not None:
                        scores = aggregate_candidate_logprobs(retry_observed, variant["candidates"])
                        parsed, errors = {"candidates": scores}, []
                else:
                    attempt_count = 1
                    errors = [retry_error or "missing_retry_response"]
                target = saved_by_key.get(key)
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
                    add_retry_request(variant, schema, row, raw or "", errors, attempt_count + 1)
                    continue

            add_retry_request(variant, schema, row, raw or "", errors, attempt_count + 1)
            row_output = result_row(
                config,
                run_id=run_id,
                variant_id=variant_id,
                config_hash=config_hash,
                group_id=group_id,
                row=row,
                prompt_text=current_prompt,
                raw=raw,
                parsed=parsed,
                errors=errors,
                attempt_count=attempt_count,
                candidate_logprobs=scores,
                final_status=None if not errors else ("retry_pending" if retry_pending else "failed_validation"),
            )
            saved.append(row_output)
            saved_by_key[key] = row_output

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
    for option, flag in (
        ("tokenizer_mode", "--tokenizer-mode"),
        ("config_format", "--config-format"),
        ("load_format", "--load-format"),
    ):
        if option in model:
            command.extend([flag, str(model[option])])
    return command


def batch_command(config: dict[str, Any]) -> str:
    return shlex.join(batch_command_args(config))


def response_from_api(completion: Any, variant: dict[str, Any]) -> Response:
    choice = completion.choices[0]
    raw = str(getattr(choice.message, "content", None) or "")
    usage = getattr(completion, "usage", None)
    token_count = int(getattr(usage, "completion_tokens", 0) or 0)
    raw_observed = extract_top_logprobs(getattr(getattr(choice, "logprobs", None), "content", None))
    if variant["request_mode"] == "candidate_logprobs":
        scores = aggregate_candidate_logprobs(raw_observed, variant["candidates"])
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
    backend = (
        FakeBackend()
        if config["model"].get("backend") == "fake"
        else VLLMBackend(config["model"], config.get("_resource_guard"))
    )
    load_seconds = time.perf_counter() - load_started
    entries: list[tuple[dict[str, Any], list[str]]] = []
    for configured_variant in config["variants"]:
        variant = materialize_variant(config, configured_variant)
        schema = variant_schema(config, variant)
        request_variant = {**variant, "_schema": schema, "_system": system_prompt(config, variant, schema)}
        prompts = [rendered_prompt(config, variant, row, schema) for row in rows]
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
                env={**os.environ, **configure_vllm_environment(config["model"])},
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
    # Every command except gpu-preflight takes the same config/override flags.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    common.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    common.add_argument("--run-id")
    common.add_argument("--model")
    common.add_argument("--backend", choices=["local_vllm", "nvidia_api", "fake"])
    common.add_argument("--output")

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "run-matrix", "prepare", "prepare-matrix", "batch-command", "self-test"):
        command = commands.add_parser(name, parents=[common])
        if name == "batch-command":
            command.add_argument("--dataset", help="Dataset id when using a matrix configuration")
        if name in {"run", "run-matrix"}:
            command.add_argument("--rows", type=int, default=None, help="Optional row limit")
        if name in {"run-matrix", "prepare-matrix"}:
            command.add_argument("--datasets", help="Comma-separated dataset ids to run (default: all)")
    commands.add_parser("gpu-preflight", help="Verify host driver access and CUDA availability in this uv environment")
    benchmark_parser = commands.add_parser("benchmark", parents=[common])
    benchmark_parser.add_argument(
        "--approaches",
        default=None,
        help="Comma-separated subset of api,run-batch,python (default: all configured approaches)",
    )
    benchmark_parser.add_argument("--rows", type=int, default=None, help="Limit benchmark input rows")
    parse = commands.add_parser("parse", parents=[common])
    parse.add_argument("--responses", required=True)
    parse.add_argument("--dataset", help="Dataset id when using a matrix configuration")
    args = parser.parse_args()
    if args.command == "gpu-preflight":
        result = gpu_preflight()
        print(json.dumps(result, indent=2))
        return 0 if result["ready"] else 2
    config = load_config(args.config, config_overrides(args), check_files=False)
    if getattr(args, "dataset", None):
        config = select_dataset(config, args.dataset)
    validate_config(config, check_files=True)
    apply_resource_guard(config)
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
