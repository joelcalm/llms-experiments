"""YAML loading, structural validation, path resolution, and dataset selection."""

from __future__ import annotations

import argparse
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .backends.factory import BACKEND_TYPES
from .inputs.factory import INPUT_READER_TYPES
from .processors import ConfigurationDefaultWarning, create_processor


def resolve(config: dict[str, Any], value: str | Path) -> Path:
    expanded = os.path.expandvars(str(value))
    path = Path(expanded)
    return path if path.is_absolute() else Path(config["_root"]) / path


def _set_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current: Any = config
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current.setdefault(part, {})
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
    configured_root = config.get("config_root")
    if configured_root is not None and not isinstance(configured_root, str):
        raise ValueError("config_root must be a relative path string")
    if configured_root is not None:
        root_path = Path(os.path.expandvars(configured_root))
        if root_path.is_absolute():
            raise ValueError("config_root must be a relative path string")
        root = (path.parent / root_path).resolve()
    else:
        root = path.parent.parent if path.parent.name in {"config", "experiments"} else path.parent
    config["_root"] = str(root)
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


def _validate_source(source: dict[str, Any], name: str) -> None:
    for key in ("path", "format", "id_column", "text_column"):
        if not source.get(key):
            raise ValueError(f"{name}.{key} is required")
    if source["format"] not in INPUT_READER_TYPES:
        raise ValueError(f"{name}.format is unsupported")
    if source["format"] == "paired_tsv" and not source.get("labels_path"):
        raise ValueError(f"{name}.labels_path is required for paired_tsv")


def _system_prompt_paths(variant: dict[str, Any]) -> list[str]:
    declared = variant.get("system_prompt")
    if not declared:
        return []
    return [declared] if isinstance(declared, str) else list(declared)


def validate_config(config: dict[str, Any], *, check_files: bool = False) -> None:
    for key in ("run", "model", "variants", "output"):
        if key not in config:
            raise ValueError(f"Missing required top-level key `{key}`")
    if not config["run"].get("id"):
        raise ValueError("run.id is required")
    if "datasets" in config:
        if not isinstance(config["datasets"], list) or not config["datasets"]:
            raise ValueError("datasets must be a non-empty list")
        sources: list[dict[str, Any]] = []
        identifiers: set[str] = set()
        for dataset in config["datasets"]:
            identifier = str(dataset.get("id", ""))
            if not identifier or identifier in identifiers:
                raise ValueError("Every dataset needs a unique id")
            identifiers.add(identifier)
            source = dataset.get("input", dataset)
            _validate_source(source, f"datasets[{identifier}].input")
            sources.append(source)
    else:
        if "input" not in config:
            raise ValueError("Missing required top-level key `input` (or `datasets`)")
        _validate_source(config["input"], "input")
        sources = [config["input"]]
    backend = str(config["model"].get("backend", ""))
    if backend not in BACKEND_TYPES:
        raise ValueError(f"model.backend must be one of: {', '.join(sorted(BACKEND_TYPES))}")
    environment = config["model"].get("vllm_environment", {})
    if not isinstance(environment, dict):
        raise ValueError("model.vllm_environment must be a mapping")
    if any(not isinstance(key, str) or not key.startswith("VLLM_") for key in environment):
        raise ValueError("model.vllm_environment keys must start with VLLM_")
    if any(not isinstance(value, str | int | float | bool) for value in environment.values()):
        raise ValueError("model.vllm_environment values must be scalar")
    seen: set[str] = set()
    for variant in config["variants"]:
        identifier = str(variant.get("id", ""))
        if not identifier or identifier in seen:
            raise ValueError("Every variant needs a unique id")
        seen.add(identifier)
        if not variant.get("prompts"):
            raise ValueError(f"{identifier}: prompts must not be empty")
        for source in sources:
            create_processor(
                variant,
                root=config["_root"],
                dataset_labels=source.get("labels", config.get("run", {}).get("dataset_labels", [])),
                code_labels=source.get("code_labels", config.get("run", {}).get("code_labels", {})),
            )
    output_format = config["output"].get("format")
    if output_format is None:
        warnings.warn(
            "output.format omitted; using the default parquet output store",
            ConfigurationDefaultWarning,
            stacklevel=2,
        )
        config["output"]["format"] = "parquet"
    elif output_format not in {"parquet", "csv", "jsonl"}:
        raise ValueError("output.format must be parquet, csv, or jsonl")
    sizes = config.get("batch", {}).get("candidates", [1])
    if not sizes or any(not isinstance(size, int) or size < 1 for size in sizes):
        raise ValueError("batch.candidates must contain positive integers")
    benchmark = config.get("benchmark", {})
    approaches = benchmark.get("approaches", ["api", "run-batch", "python"])
    if not approaches or any(item not in {"api", "run-batch", "python"} for item in approaches):
        raise ValueError("benchmark.approaches must contain api, run-batch, or python")
    if int(benchmark.get("rows", 1)) < 1:
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
    reserve = cpu.get("reserve_cores", 2)
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        raise ValueError("resources.cpu.reserve_cores must be a non-negative integer")
    pool = cpu.get("thread_pool_size", 1)
    if isinstance(pool, bool) or not isinstance(pool, int) or pool < 1:
        raise ValueError("resources.cpu.thread_pool_size must be a positive integer")
    if not isinstance(cpu.get("affinity", True), bool):
        raise ValueError("resources.cpu.affinity must be a boolean")
    if check_files:
        paths: list[str] = []
        for source in sources:
            paths.append(str(source["path"]))
            if source.get("format") == "paired_tsv":
                paths.append(str(source["labels_path"]))
                for pair in source.get("additional_pairs", []):
                    paths.extend([str(pair["path"]), str(pair["labels_path"])])
            paths.extend(str(item) for item in source.get("prompt_parts", {}).values())
        for variant in config["variants"]:
            paths.extend(str(item) for item in variant["prompts"])
            paths.extend(_system_prompt_paths(variant))
            for stage in (variant.get("processor") or {}).get("stages", []):
                if stage.get("type") == "json_schema" and stage.get("schema"):
                    paths.append(str(stage["schema"]))
        retry = config.get("validation", {}).get("retry", {}).get("correction_prompt")
        if retry:
            paths.append(str(retry))
        for item in paths:
            if not resolve(config, item).is_file():
                raise ValueError(f"Configured file does not exist: {item}")


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


def dataset_config(
    config: dict[str, Any],
    dataset_id: str,
    source: dict[str, Any],
    base_output: Path | None = None,
) -> dict[str, Any]:
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
    entries = dataset_entries(config)
    if not selected:
        return entries
    wanted = set(selected)
    filtered = [(identifier, source) for identifier, source in entries if identifier in wanted]
    missing = wanted - {identifier for identifier, _ in filtered}
    if missing:
        raise ValueError(f"Unknown dataset id(s): {', '.join(sorted(missing))}")
    return filtered


def select_dataset(config: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    for identifier, source in dataset_entries(config):
        if identifier == dataset_id:
            return dataset_config(config, identifier, source)
    raise ValueError(f"Unknown dataset id: {dataset_id}")


def config_overrides(args: argparse.Namespace) -> list[str]:
    overrides = list(args.overrides or [])
    shortcuts = {
        "run_id": "run.id",
        "model": "model.name",
        "backend": "model.backend",
        "output": "output.directory",
        "output_format": "output.format",
    }
    for argument, key in shortcuts.items():
        value = getattr(args, argument, None)
        if value is not None:
            overrides.append(f"{key}={value}")
    return overrides
