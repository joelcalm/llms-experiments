"""Prompt assets, processor materialization, schemas, and batch request rendering."""

from __future__ import annotations

import hashlib
import json
import re
from functools import cache
from pathlib import Path
from typing import Any

from .configuration import resolve
from .processors import Processor, create_processor
from .processors.factory import processor_config_hash_material

TOKEN = re.compile(
    r"{{\s*(text|row_id|dataset_id|labels|candidate_mapping|question|target_label|definitions|theory|output_schema|raw_response|validation_errors|candidates)\s*}}"
)
UNRESOLVED_TOKEN = re.compile(r"{{\s*[^{}]+\s*}}")
ROW_SPECIFIC_TOKENS = ("text", "row_id", "target_label")
CONTRACT_VERSION = "2.0"


@cache
def _read_asset(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_asset(config: dict[str, Any], value: str | Path) -> str:
    return _read_asset(str(resolve(config, value).resolve()))


def render(template: str, values: dict[str, Any]) -> str:
    rendered = TOKEN.sub(lambda match: str(values.get(match.group(1), "")), template)
    unresolved = UNRESOLVED_TOKEN.search(rendered)
    if unresolved:
        raise ValueError(f"Unsupported or unresolved prompt placeholder: {unresolved.group(0)}")
    return rendered


def prompt(config: dict[str, Any], paths: list[str], values: dict[str, Any]) -> str:
    return "\n\n".join(render(read_asset(config, path), values) for path in paths)


def prompt_part_values(config: dict[str, Any], values: dict[str, Any] | None = None) -> dict[str, str]:
    rendered: dict[str, str] = {}
    context = dict(values or {})
    parts = {
        **config.get("run", {}).get("prompt_parts", {}),
        **config.get("input", {}).get("prompt_parts", {}),
    }
    for name, path in parts.items():
        rendered[name] = render(read_asset(config, path), {**context, **rendered})
    return rendered


def materialize_variant(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    if variant.get("_processor") is not None:
        return dict(variant)
    processor = create_processor(
        variant,
        root=config["_root"],
        dataset_labels=config.get("run", {}).get("dataset_labels", []),
        code_labels=config.get("run", {}).get("code_labels", {}),
    )
    requirements = processor.requirements
    return {
        **variant,
        "_processor": processor,
        "_schema": dict(requirements.structured_schema) if requirements.structured_schema is not None else None,
        "_candidates": list(requirements.candidates),
        "_max_tokens": requirements.max_tokens,
        "_top_logprobs": requirements.top_logprobs,
    }


def variant_schema(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any] | None:
    materialized = materialize_variant(config, variant)
    schema = materialized["_processor"].requirements.structured_schema
    return dict(schema) if schema is not None else None


def prompt_values(
    config: dict[str, Any],
    variant: dict[str, Any],
    row: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    materialized = materialize_variant(config, variant)
    run = config.get("run", {})
    candidates = ", ".join(str(item) for item in materialized["_processor"].requirements.candidates)
    return {
        "text": row[config["input"]["text_column"]],
        "row_id": row[config["input"]["id_column"]],
        "dataset_id": run.get("dataset_id", "default"),
        "candidates": candidates,
        "labels": ", ".join(str(item) for item in run.get("dataset_labels", [])),
        "candidate_mapping": ", ".join(f"{code}={label}" for code, label in run.get("code_labels", {}).items())
        or candidates,
        "question": run.get("binary_question", "Does this text express the target value?"),
        "target_label": row.get("_target_label", ""),
        "output_schema": json.dumps(schema or {}, sort_keys=True),
    }


def rendered_prompt(
    config: dict[str, Any],
    variant: dict[str, Any],
    row: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> str:
    materialized = materialize_variant(config, variant)
    if schema is None:
        schema = variant_schema(config, materialized)
    values = prompt_values(config, materialized, row, schema)
    values.update(prompt_part_values(config, values))
    return prompt(config, materialized["prompts"], values)


def system_prompt_paths(variant: dict[str, Any]) -> list[str]:
    declared = variant.get("system_prompt")
    if not declared:
        return []
    return [declared] if isinstance(declared, str) else list(declared)


def system_prompt(
    config: dict[str, Any],
    variant: dict[str, Any],
    schema: dict[str, Any] | None,
) -> str | None:
    paths = system_prompt_paths(variant)
    if not paths:
        return None
    for path in paths:
        raw = read_asset(config, path)
        for token in ROW_SPECIFIC_TOKENS:
            if f"{{{{{token}}}}}" in raw:
                raise ValueError(
                    f"{variant['id']}: system_prompt may not use {{{{{token}}}}} ({path}). "
                    "It is rendered once per variant; put row placeholders in prompts."
                )
    source = config["input"]
    blank = {source["id_column"]: "", source["text_column"]: ""}
    values = prompt_values(config, variant, blank, schema)
    values.update(prompt_part_values(config, values))
    return prompt(config, paths, values)


def conversation(system: str | None, user: str) -> list[dict[str, str]]:
    turns = [{"role": "system", "content": system}] if system else []
    return [*turns, {"role": "user", "content": user}]


def retry_values(
    config: dict[str, Any],
    variant: dict[str, Any],
    row: dict[str, Any],
    schema: dict[str, Any] | None,
    raw: str,
    errors: list[str],
) -> dict[str, Any]:
    values = prompt_values(config, variant, row, schema)
    values["raw_response"] = raw
    values["validation_errors"] = "; ".join(errors)
    values.update(prompt_part_values(config, values))
    return values


def prompt_group_id(
    config: dict[str, Any],
    variant: dict[str, Any],
    schema: dict[str, Any] | None,
) -> str:
    source = config["input"]
    sentinel = {source["id_column"]: "<row>", source["text_column"]: "<text>"}
    static = rendered_prompt(config, variant, sentinel, schema)
    system = system_prompt(config, variant, schema)
    material = f"{system}\n\n{static}" if system else static
    return hashlib.sha256(material.replace("<text>", "{{text}}").encode()).hexdigest()[:16]


def variant_config_hash(config: dict[str, Any], variant: dict[str, Any]) -> str:
    assets = {}
    for path in list(variant.get("prompts", [])) + system_prompt_paths(variant):
        assets[str(path)] = read_asset(config, path)
    for name, path in config.get("run", {}).get("prompt_parts", {}).items():
        assets[f"part:{name}"] = read_asset(config, path)
    cleaned = processor_config_hash_material({key: value for key, value in variant.items() if not key.startswith("_")})
    payload = {
        "contract_version": CONTRACT_VERSION,
        "variant": cleaned,
        "model": config.get("model"),
        "input": config.get("input"),
        "run": config.get("run", {}).get("dataset_id", "default"),
        "assets": assets,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def request_for_row(
    config: dict[str, Any],
    variant: dict[str, Any],
    row: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    materialized = materialize_variant(config, variant)
    processor: Processor = materialized["_processor"]
    schema = schema if schema is not None else variant_schema(config, materialized)
    requirements = processor.requirements
    body: dict[str, Any] = {
        "model": config["model"]["name"],
        "messages": conversation(
            system_prompt(config, materialized, schema), rendered_prompt(config, materialized, row, schema)
        ),
        "temperature": 0,
        "max_completion_tokens": requirements.max_tokens,
    }
    if requirements.capture_logprobs:
        body.update({"logprobs": True, "top_logprobs": requirements.top_logprobs})
    if requirements.structured_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": materialized["id"],
                "schema": dict(requirements.structured_schema),
                "strict": True,
            },
        }
    return {
        "custom_id": f"{materialized['id']}:{row[config['input']['id_column']]}:{row['_source_position']}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def build_requests(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for configured in config["variants"]:
        variant = materialize_variant(config, configured)
        processor: Processor = variant["_processor"]
        requests.extend(request_for_row(config, variant, row) for row in processor.prepare_rows(rows))
    return requests
