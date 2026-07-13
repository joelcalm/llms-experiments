"""Configuration loading and generic contract checks for experiment-cli."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

MODES = {"generate", "candidate_logprobs"}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    config = deepcopy(config)
    config["_config_path"] = str(path)
    config["_root"] = str(path.parent.parent)
    validate_config(config, check_files=True)
    return config


def root_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_root"]) / path


def validate_config(config: dict[str, Any], *, check_files: bool = False) -> None:
    for key in ("run", "input", "model", "variants", "output"):
        if key not in config:
            raise ValueError(f"Missing required top-level key `{key}`")
    if not isinstance(config["run"], dict) or not config["run"].get("id"):
        raise ValueError("run.id is required")
    input_config = config["input"]
    for key in ("path", "format", "id_column", "text_column"):
        if not input_config.get(key):
            raise ValueError(f"input.{key} is required")
    if input_config["format"] not in {"csv", "jsonl", "parquet"}:
        raise ValueError("input.format must be csv, jsonl, or parquet")
    model = config["model"]
    if model.get("backend") not in {"local_vllm", "fake"}:
        raise ValueError("model.backend must be local_vllm or fake")
    if not isinstance(config["variants"], list) or not config["variants"]:
        raise ValueError("variants must be a non-empty list")
    seen: set[str] = set()
    for variant in config["variants"]:
        variant_id = variant.get("id")
        if not variant_id or variant_id in seen:
            raise ValueError("Every variant needs a unique id")
        seen.add(variant_id)
        if variant.get("request_mode") not in MODES:
            raise ValueError(f"{variant_id}: request_mode must be one of {sorted(MODES)}")
        prompts = variant.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"{variant_id}: prompts must be a non-empty list")
        if variant["request_mode"] == "candidate_logprobs" and not variant.get("candidates"):
            raise ValueError(f"{variant_id}: candidate_logprobs requires candidates")
    batch = config.get("batch", {})
    if batch.get("mode", "auto") not in {"auto", "fixed"}:
        raise ValueError("batch.mode must be auto or fixed")
    candidates = batch.get("candidates", [1])
    if not candidates or any(not isinstance(size, int) or size < 1 for size in candidates):
        raise ValueError("batch.candidates must contain positive integers")
    if check_files:
        input_path = root_path(config, input_config["path"])
        if not input_path.exists():
            raise ValueError(f"Input does not exist: {input_path}")
        for variant in config["variants"]:
            for prompt in variant["prompts"]:
                if not root_path(config, prompt).is_file():
                    raise ValueError(f"{variant['id']}: prompt does not exist: {prompt}")
            schema = variant.get("validation", {}).get("schema")
            if schema and not root_path(config, schema).is_file():
                raise ValueError(f"{variant['id']}: schema does not exist: {schema}")

