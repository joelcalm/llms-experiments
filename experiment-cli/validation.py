"""Optional generic JSON/schema validation; no application taxonomies are embedded here."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_schema(path: Path | None) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def _type_matches(value: Any, declared: str) -> bool:
    return {"object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool), "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool), "null": value is None}.get(declared, True)


def _check(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    declared = schema.get("type")
    if declared and not _type_matches(value, declared):
        errors.append(f"{path}: expected {declared}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not an allowed candidate")
    if isinstance(value, str) and schema.get("pattern") and not re.search(schema["pattern"], value):
        errors.append(f"{path}: does not match pattern")
    if isinstance(value, dict):
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}: missing required key `{required}`")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: unexpected key `{key}`")
        for key, child in properties.items():
            if key in value:
                _check(value[key], child, f"{path}.{key}", errors)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _check(item, schema["items"], f"{path}[{index}]", errors)


def validate_response(raw: str, schema: dict[str, Any] | None) -> tuple[Any | None, list[str]]:
    if schema is None:
        return raw, []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"json_parse_error: {exc.msg}"]
    errors: list[str] = []
    _check(parsed, schema, "$", errors)
    return parsed, errors

