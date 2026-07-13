"""Bounded generic correction-prompt retries."""
from __future__ import annotations

from typing import Any, Callable

from prompts import render


def retry_response(
    initial_raw: str,
    errors: list[str],
    retry_config: dict[str, Any],
    correction_template: str | None,
    request: Callable[[str], str],
    values: dict[str, Any],
    validate: Callable[[str], tuple[Any | None, list[str]]],
) -> tuple[str, Any | None, list[str], int]:
    raw, parsed, current_errors, attempts = initial_raw, None, errors, 1
    if not retry_config.get("enabled") or not correction_template:
        return raw, parsed, current_errors, attempts
    for _ in range(int(retry_config.get("max_attempts", 0))):
        prompt = render(correction_template, {**values, "raw_response": raw, "validation_errors": "; ".join(current_errors)})
        raw = request(prompt)
        parsed, current_errors = validate(raw)
        attempts += 1
        if not current_errors:
            break
    return raw, parsed, current_errors, attempts

