"""Response conversion utilities and backwards-compatibility helpers for RawModelResponse."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .processors import RawModelResponse, TokenCandidate, TokenPosition
from .results import BACKEND_ERROR_PREFIX, validate_response


@dataclass
class Response:
    """Structured response wrapper; callers requiring typed token streams should use ``RawModelResponse``."""

    raw: str
    token_count: int
    candidate_logprobs: dict[str, float] | None = None
    backend_error: str | None = None
    token_logprobs: list[dict[str, Any]] | None = None

    @property
    def text(self) -> str:
        return self.raw

    @property
    def candidate_scores(self) -> dict[str, float] | None:
        return self.candidate_logprobs

    @property
    def token_positions(self) -> tuple[TokenPosition, ...]:
        return tuple(
            TokenPosition(
                token=str(position.get("token", "")),
                logprob=float(position.get("logprob", -float("inf"))),
                alternatives=tuple(
                    TokenCandidate(str(token), float(logprob)) for token, logprob in position.get("top_logprobs", [])
                ),
            )
            for position in self.token_logprobs or []
        )

    def normalized(self) -> RawModelResponse:
        return RawModelResponse(
            text=self.raw,
            token_count=self.token_count,
            token_positions=self.token_positions,
            backend_error=self.backend_error,
            candidate_scores=self.candidate_logprobs,
        )


def top_logprobs_count(candidates: list[Any]) -> int:
    return min(20, len(candidates) + 5)


def extract_top_logprobs(logprob_content: Any) -> list[tuple[str, float]]:
    observed: list[tuple[str, float]] = []
    for token in logprob_content or []:
        candidates = token.get("top_logprobs") if isinstance(token, dict) else getattr(token, "top_logprobs", None)
        for candidate in candidates or []:
            if isinstance(candidate, dict):
                name = candidate.get("token", "")
                logprob = candidate.get("logprob", -float("inf"))
            else:
                name = getattr(candidate, "token", "")
                logprob = getattr(candidate, "logprob", -float("inf"))
            observed.append((str(name), float(logprob)))
    return observed


def extract_position_logprobs(logprob_content: Any) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for item in logprob_content or []:
        sampled_token = item.get("token", "") if isinstance(item, dict) else getattr(item, "token", "")
        sampled_logprob = (
            item.get("logprob", -float("inf")) if isinstance(item, dict) else getattr(item, "logprob", -float("inf"))
        )
        top = item.get("top_logprobs") if isinstance(item, dict) else getattr(item, "top_logprobs", None)
        observed: list[tuple[str, float]] = []
        for candidate in top or []:
            if isinstance(candidate, dict):
                token = candidate.get("token", "")
                logprob = candidate.get("logprob", -float("inf"))
            else:
                token = getattr(candidate, "token", "")
                logprob = getattr(candidate, "logprob", -float("inf"))
            observed.append((str(token), float(logprob)))
        sampled = (str(sampled_token), float(sampled_logprob))
        if sampled[0] and sampled not in observed:
            observed.append(sampled)
        positions.append({"token": sampled[0], "logprob": sampled[1], "top_logprobs": observed})
    return positions


def flatten_position_logprobs(positions: list[dict[str, Any]]) -> list[tuple[str, float]]:
    return [candidate for position in positions for candidate in position.get("top_logprobs", [])]


def aggregate_candidate_logprobs(
    raw_logprobs: list[tuple[str, float]],
    candidates: list[Any],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for token, logprob in raw_logprobs:
        grouped.setdefault(token.strip(), []).append(logprob)
    aggregated: dict[str, float] = {}
    for token, logprobs in grouped.items():
        maximum = max(logprobs)
        aggregated[token] = maximum + math.log(sum(math.exp(value - maximum) for value in logprobs))
    return {str(candidate): aggregated.get(str(candidate).strip(), -float("inf")) for candidate in candidates}


_DIGIT_TOKEN = re.compile(r'^[\s":,\[\]{}]*([0-9])[\s":,\[\]{}]*$')


def _digit_from_token(token: str) -> int | None:
    match = _DIGIT_TOKEN.fullmatch(token)
    return int(match.group(1)) if match else None


def digit_logprobs(position: dict[str, Any]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for token, logprob in position.get("top_logprobs", []):
        digit = _digit_from_token(str(token))
        if digit is not None:
            grouped.setdefault(str(digit), []).append(float(logprob))
    result: dict[str, float] = {}
    for digit_key, values in grouped.items():
        maximum = max(values)
        result[digit_key] = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    return result


def verbalized_confidence(
    parsed: Any,
    token_logprobs: list[dict[str, Any]] | None,
) -> tuple[Any | None, list[str]]:
    if not isinstance(parsed, dict):
        return parsed, ["confidence_parse_error: expected a JSON object"]
    try:
        tens = int(parsed["confidence_tens"])
        units = int(parsed["confidence_units"])
    except (KeyError, TypeError, ValueError):
        return parsed, ["confidence_parse_error: confidence_tens and confidence_units must be digits"]
    if not 0 <= tens <= 9 or not 0 <= units <= 9:
        return parsed, ["confidence_parse_error: confidence digits must be in [0, 9]"]

    positions = token_logprobs or []
    matched: list[dict[str, Any]] = []
    cursor = len(positions) - 1
    for expected in (units, tens):
        while cursor >= 0 and _digit_from_token(str(positions[cursor].get("token", ""))) != expected:
            cursor -= 1
        if cursor < 0:
            return parsed, [
                "confidence_logprobs_missing: could not align generated confidence digits with token logprobs"
            ]
        matched.append(positions[cursor])
        cursor -= 1
    units_position, tens_position = matched
    distributions = {
        "tens": digit_logprobs(tens_position),
        "units": digit_logprobs(units_position),
    }
    if not distributions["tens"] or not distributions["units"]:
        return parsed, ["confidence_logprobs_missing: no digit alternatives were returned"]
    probabilities = {
        place: {digit: math.exp(logprob) for digit, logprob in values.items()}
        for place, values in distributions.items()
    }
    masses = {place: sum(values.values()) for place, values in probabilities.items()}
    expected_tens = sum(int(digit) * probability for digit, probability in probabilities["tens"].items())
    expected_units = sum(int(digit) * probability for digit, probability in probabilities["units"].items())
    return {
        **parsed,
        "verbalized_confidence": (10 * tens + units) / 100,
        "logprob_weighted_confidence": (10 * expected_tens + expected_units) / 100,
        "confidence_digit_logprobs": distributions,
        "confidence_digit_probability_mass": masses,
    }, []


def interpret_response(
    response: Response | RawModelResponse,
    schema: dict[str, Any] | None,
    request_mode: str,
) -> tuple[Any | None, list[str]]:
    backend_error = response.backend_error
    if backend_error:
        return None, [f"{BACKEND_ERROR_PREFIX} {backend_error}"]
    raw = response.raw if isinstance(response, Response) else response.text
    if request_mode == "candidate_logprobs":
        scores = response.candidate_logprobs
        return {"candidates": scores or {}}, []
    parsed, errors = validate_response(raw, schema)
    if not errors and request_mode == "generate_with_logprobs":
        return verbalized_confidence(parsed, response.token_logprobs)
    return parsed, errors


def expanded_rows(rows: Any, labels: list[str]) -> Any:
    width = len(labels)
    for row in rows:
        base = int(row["_source_position"])
        for label_index, label in enumerate(labels):
            yield {
                **row,
                "_target_label": label,
                "_source_position": base * width + label_index,
            }


__all__ = [
    "Response",
    "aggregate_candidate_logprobs",
    "digit_logprobs",
    "expanded_rows",
    "extract_position_logprobs",
    "extract_top_logprobs",
    "flatten_position_logprobs",
    "interpret_response",
    "top_logprobs_count",
    "verbalized_confidence",
]
