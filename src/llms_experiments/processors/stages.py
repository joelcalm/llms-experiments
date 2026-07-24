"""Built-in reusable processing stages."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from typing import Any

from .base import ProcessingStage
from .contracts import (
    ProcessingError,
    ProcessingState,
    ProcessorContext,
    ResponseRequirements,
    TokenCandidate,
    TokenPosition,
)


def _logsumexp(values: list[float]) -> float:
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
        if sampled not in alternatives:
            alternatives.append(sampled)
        for item in alternatives:
            grouped.setdefault(item.token.strip(), []).append(float(item.logprob))
    aggregated = {token: _logsumexp(values) for token, values in grouped.items()}
    return {candidate: aggregated.get(candidate.strip(), -float("inf")) for candidate in candidates}


class IdentityStage(ProcessingStage):
    type_name = "identity"
    required_fields = frozenset({"raw_text"})
    produced_fields = frozenset({"value"})

    def requirements(self, context: ProcessorContext) -> ResponseRequirements:
        return ResponseRequirements(max_tokens=int(self.config.get("max_tokens", 128)))

    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        return replace(state, value=state.response.text)


class FanOutStage(ProcessingStage):
    type_name = "fan_out"
    prepares_rows = True

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        if self.config.get("over") != "dataset_labels":
            raise ValueError("fan_out.over must be 'dataset_labels'")

    def prepare_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        context: ProcessorContext,
    ) -> Iterator[Mapping[str, Any]]:
        labels = context.dataset_labels
        if not labels:
            raise ValueError("fan_out over dataset_labels requires a non-empty dataset label set")
        width = len(labels)
        for row in rows:
            base = int(row["_source_position"])
            for index, label in enumerate(labels):
                yield {
                    **row,
                    "_target_label": label,
                    "_source_position": base * width + index,
                }

    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        return state


class JsonDecodeStage(ProcessingStage):
    type_name = "json_decode"
    required_fields = frozenset({"raw_text"})
    produced_fields = frozenset({"value"})

    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        try:
            return replace(state, value=json.loads(state.response.text))
        except json.JSONDecodeError as exc:
            return state.fail(
                ProcessingError(
                    code="json_parse_error",
                    message=str(exc),
                    stage=self.type_name,
                    category="parsing",
                )
            )


def _replace_schema_enums(node: Any, labels: tuple[str, ...]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "string" and "enum" in node:
            node["enum"] = list(labels)
        for child in node.values():
            _replace_schema_enums(child, labels)
    elif isinstance(node, list):
        for child in node:
            _replace_schema_enums(child, labels)


def _check_schema(value: Any, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    expected = schema.get("type")
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
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: missing required property")
        properties = schema.get("properties", {})
        for key, child in value.items():
            if key in properties:
                _check_schema(child, properties[key], f"{path}.{key}", errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: additional property is not allowed")
    if isinstance(value, list) and "items" in schema:
        for index, child in enumerate(value):
            _check_schema(child, schema["items"], f"{path}[{index}]", errors)
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: value is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: value is above maximum")


class JsonSchemaStage(ProcessingStage):
    type_name = "json_schema"
    required_fields = frozenset({"value"})
    produced_fields = frozenset({"value"})

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        if not self.config.get("schema"):
            raise ValueError("json_schema.schema is required")
        self._schema: dict[str, Any] | None = None

    def _resolved_schema(self, context: ProcessorContext) -> dict[str, Any]:
        if self._schema is None:
            self._schema = json.loads(context.resolve(str(self.config["schema"])).read_text(encoding="utf-8"))
            if self.config.get("enum_from") == "dataset_labels":
                _replace_schema_enums(self._schema, context.dataset_labels)
            elif self.config.get("enum_from") is not None:
                raise ValueError("json_schema.enum_from must be 'dataset_labels'")
        return self._schema

    def requirements(self, context: ProcessorContext) -> ResponseRequirements:
        return ResponseRequirements(structured_schema=self._resolved_schema(context), max_tokens=1)

    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        errors: list[str] = []
        _check_schema(state.value, self._resolved_schema(context), "$", errors)
        if not errors:
            return state
        return state.fail(
            ProcessingError(
                code="schema_validation_error",
                message="; ".join(errors),
                stage=self.type_name,
                category="validation",
            )
        )


class CandidateLogprobsStage(ProcessingStage):
    type_name = "candidate_logprobs"
    required_fields = frozenset({"token_positions"})
    produced_fields = frozenset({"value", "candidate_scores"})

    def _candidates(self, context: ProcessorContext) -> tuple[str, ...]:
        declared = self.config.get("candidates")
        source = self.config.get("candidates_from")
        if declared is not None and source is not None:
            raise ValueError("candidate_logprobs accepts candidates or candidates_from, not both")
        if declared is not None:
            candidates = tuple(str(item) for item in declared)
        elif source == "dataset_labels":
            candidates = context.dataset_labels
        elif source == "code_labels":
            candidates = tuple(str(item) for item in context.code_labels)
        else:
            raise ValueError("candidate_logprobs requires candidates or a supported candidates_from")
        if not candidates:
            raise ValueError("candidate_logprobs candidate set must not be empty")
        return candidates

    def requirements(self, context: ProcessorContext) -> ResponseRequirements:
        candidates = self._candidates(context)
        return ResponseRequirements(
            max_tokens=1,
            capture_logprobs=True,
            top_logprobs=min(20, len(candidates) + 5),
            one_token=True,
            candidates=candidates,
        )

    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        candidates = self._candidates(context)
        if state.response.token_positions:
            scores = aggregate_candidate_logprobs(state.response.token_positions[0], candidates)
        elif state.response.candidate_scores is not None:
            scores = {
                candidate: float(state.response.candidate_scores.get(candidate, -float("inf")))
                for candidate in candidates
            }
        else:
            return state.fail(
                ProcessingError(
                    code="candidate_logprobs_missing",
                    message="backend returned no positional token logprobs",
                    stage=self.type_name,
                    category="processing",
                )
            )
        return replace(state, value={"candidates": scores}, candidate_scores=scores)


_DIGIT_TOKEN = re.compile(r'^[\s":,\[\]{}]*([0-9])[\s":,\[\]{}]*$')


def _digit_from_token(token: str) -> int | None:
    match = _DIGIT_TOKEN.fullmatch(token)
    return int(match.group(1)) if match else None


def digit_logprobs(position: TokenPosition) -> dict[str, float]:
    """Aggregate all tokenizer spellings of digits at one position."""

    grouped: dict[str, list[float]] = {}
    alternatives = list(position.alternatives)
    sampled = TokenCandidate(position.token, position.logprob)
    if sampled not in alternatives:
        alternatives.append(sampled)
    for item in alternatives:
        digit = _digit_from_token(item.token)
        if digit is not None:
            grouped.setdefault(str(digit), []).append(float(item.logprob))
    return {digit: _logsumexp(values) for digit, values in grouped.items()}


class VerbalizedConfidenceStage(ProcessingStage):
    type_name = "verbalized_confidence"
    required_fields = frozenset({"value", "token_positions"})
    produced_fields = frozenset({"value"})

    def requirements(self, context: ProcessorContext) -> ResponseRequirements:
        requested = int(self.config.get("top_logprobs", 20))
        if not 10 <= requested <= 20:
            raise ValueError("verbalized_confidence.top_logprobs must be between 10 and 20")
        return ResponseRequirements(capture_logprobs=True, top_logprobs=requested, max_tokens=1)

    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        if not isinstance(state.value, dict):
            return state.fail(
                ProcessingError(
                    code="confidence_parse_error",
                    message="expected a JSON object",
                    stage=self.type_name,
                    category="processing",
                )
            )
        tens_field = str(self.config.get("tens_field", "confidence_tens"))
        units_field = str(self.config.get("units_field", "confidence_units"))
        try:
            tens = int(state.value[tens_field])
            units = int(state.value[units_field])
        except (KeyError, TypeError, ValueError):
            return state.fail(
                ProcessingError(
                    code="confidence_parse_error",
                    message=f"{tens_field} and {units_field} must be digits",
                    stage=self.type_name,
                    category="processing",
                )
            )
        if not 0 <= tens <= 9 or not 0 <= units <= 9:
            return state.fail(
                ProcessingError(
                    code="confidence_parse_error",
                    message="confidence digits must be in [0, 9]",
                    stage=self.type_name,
                    category="processing",
                )
            )

        matched: list[TokenPosition] = []
        cursor = len(state.response.token_positions) - 1
        for expected in (units, tens):
            while cursor >= 0 and _digit_from_token(state.response.token_positions[cursor].token) != expected:
                cursor -= 1
            if cursor < 0:
                return state.fail(
                    ProcessingError(
                        code="confidence_logprobs_missing",
                        message="could not align generated confidence digits with token logprobs",
                        stage=self.type_name,
                        category="processing",
                    )
                )
            matched.append(state.response.token_positions[cursor])
            cursor -= 1
        units_position, tens_position = matched
        distributions = {
            "tens": digit_logprobs(tens_position),
            "units": digit_logprobs(units_position),
        }
        if not distributions["tens"] or not distributions["units"]:
            return state.fail(
                ProcessingError(
                    code="confidence_logprobs_missing",
                    message="no digit alternatives were returned",
                    stage=self.type_name,
                    category="processing",
                )
            )
        probabilities = {
            place: {digit: math.exp(logprob) for digit, logprob in values.items()}
            for place, values in distributions.items()
        }
        masses = {place: sum(values.values()) for place, values in probabilities.items()}
        expected_tens = sum(int(digit) * probability for digit, probability in probabilities["tens"].items())
        expected_units = sum(int(digit) * probability for digit, probability in probabilities["units"].items())
        enriched = {
            **state.value,
            "verbalized_confidence": (10 * tens + units) / 100,
            "logprob_weighted_confidence": (10 * expected_tens + expected_units) / 100,
            "confidence_digit_logprobs": distributions,
            "confidence_digit_probability_mass": masses,
        }
        return replace(state, value=enriched)


STAGE_TYPES: dict[str, type[ProcessingStage]] = {
    "identity": IdentityStage,
    "fan_out": FanOutStage,
    "json_decode": JsonDecodeStage,
    "json_schema": JsonSchemaStage,
    "candidate_logprobs": CandidateLogprobsStage,
    "verbalized_confidence": VerbalizedConfidenceStage,
}
