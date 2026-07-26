"""Utility and helper functions for processor stage computations and schema validation."""

from __future__ import annotations

import math
import re
from typing import Any

from .contracts import TokenCandidate, TokenPosition


def logsumexp(values: list[float]) -> float:
    """Compute log-sum-exp over a list of float values in a numerically stable way."""
    # Subtract maximum logprob before exponentiation to prevent numerical overflow
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def aggregate_candidate_logprobs(
    position: TokenPosition | None,
    candidates: tuple[str, ...],
) -> dict[str, float]:
    """Aggregate tokenizer spellings of configured first-token candidates."""
    grouped: dict[str, list[float]] = {}
    if position is not None:
        alternatives = list(position.alternatives)
        sampled = TokenCandidate(position.token, position.logprob)
        # Ensure the actually sampled token is included alongside alternative top logprobs
        if sampled not in alternatives:
            alternatives.append(sampled)
        # Group logprobabilities by whitespace-stripped token text
        for item in alternatives:
            grouped.setdefault(item.token.strip(), []).append(float(item.logprob))
    # Sum probabilities in log-space for each candidate spelling group
    aggregated = {token: logsumexp(values) for token, values in grouped.items()}
    return {candidate: aggregated.get(candidate.strip(), -float("inf")) for candidate in candidates}


def replace_schema_enums(node: Any, labels: tuple[str, ...]) -> None:
    """Recursively replace string enum lists in a JSON schema with the dataset label tuple."""
    # Recursively traverse dict/list structure and replace enum definitions on string fields
    if isinstance(node, dict):
        if node.get("type") == "string" and "enum" in node:
            node["enum"] = list(labels)
        for child in node.values():
            replace_schema_enums(child, labels)
    elif isinstance(node, list):
        for child in node:
            replace_schema_enums(child, labels)


def check_schema(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    """Validate a parsed JSON value against a simplified JSON schema, appending errors."""
    expected = schema.get("type")
    # Verify basic JSON type matching (handling bool distinction from int/number)
    valid_type = (
        expected is None
        or (expected == "object" and isinstance(value, dict))
        or (expected == "array" and isinstance(value, list))
        or (expected == "string" and isinstance(value, str))
        or (expected == "number" and isinstance(value, int | float) and not isinstance(value, bool))
        or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (expected == "boolean" and isinstance(value, bool))
        or (expected == "null" and value is None)
    )
    if not valid_type:
        errors.append(f"{path}: expected {expected}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not in enum")
    # Check object properties and required field constraints recursively
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: missing required property")
        properties = schema.get("properties", {})
        for key, child in value.items():
            if key in properties:
                check_schema(child, properties[key], f"{path}.{key}", errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: additional property is not allowed")
    # Check array element schemas recursively
    if isinstance(value, list) and "items" in schema:
        for index, child in enumerate(value):
            check_schema(child, schema["items"], f"{path}[{index}]", errors)
    # Check numerical min/max boundary constraints
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: value is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: value is above maximum")


# Regex matching a single digit 0-9 amidst optional surrounding whitespace/quotes/brackets
_DIGIT_TOKEN = re.compile(r'^[\s":,\[\]{}]*([0-9])[\s":,\[\]{}]*$')


def digit_from_token(token: str) -> int | None:
    """Extract an integer digit from a token string if it matches a digit pattern."""
    match = _DIGIT_TOKEN.fullmatch(token)
    return int(match.group(1)) if match else None


def digit_logprobs(position: TokenPosition) -> dict[str, float]:
    """Aggregate all tokenizer spellings of digits at one position."""
    grouped: dict[str, list[float]] = {}
    alternatives = list(position.alternatives)
    sampled = TokenCandidate(position.token, position.logprob)
    if sampled not in alternatives:
        alternatives.append(sampled)
    # Group logprobs for all tokens that map to the same digit value 0-9
    for item in alternatives:
        digit = digit_from_token(item.token)
        if digit is not None:
            grouped.setdefault(str(digit), []).append(float(item.logprob))
    return {digit: logsumexp(values) for digit, values in grouped.items()}
