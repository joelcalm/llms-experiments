"""Verbalized confidence processing stage implementation."""

from __future__ import annotations

import math
from dataclasses import replace

from ..base import ProcessingStage
from ..contracts import ProcessingError, ProcessingState, ProcessorContext, ResponseRequirements, TokenPosition
from ..utils import digit_from_token, digit_logprobs


class VerbalizedConfidenceStage(ProcessingStage):
    """Aligns verbalized confidence digits with token logprobabilities."""

    type_name = "verbalized_confidence"
    required_fields = frozenset({"value", "token_positions"})
    produced_fields = frozenset({"value"})

    def requirements(self, context: ProcessorContext) -> ResponseRequirements:
        """Return response requirements specifying logprob capture configuration."""
        requested = int(self.config.get("top_logprobs", 20))
        if not 10 <= requested <= 20:
            raise ValueError("verbalized_confidence.top_logprobs must be between 10 and 20")
        return ResponseRequirements(capture_logprobs=True, top_logprobs=requested, max_tokens=1)

    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        """Align confidence digits with logprobs and attach weighted confidence scores."""
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
        # Search backward from the end of token_positions to match units then tens digit tokens
        cursor = len(state.response.token_positions) - 1
        for expected in (units, tens):
            while cursor >= 0 and digit_from_token(state.response.token_positions[cursor].token) != expected:
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
        # Convert log-probabilities to linear probabilities exp(logprob) for expectation calculation
        probabilities = {
            place: {digit: math.exp(logprob) for digit, logprob in values.items()}
            for place, values in distributions.items()
        }
        masses = {place: sum(values.values()) for place, values in probabilities.items()}

        # Compute logprob-weighted expectation across all digit choices 0-9
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
