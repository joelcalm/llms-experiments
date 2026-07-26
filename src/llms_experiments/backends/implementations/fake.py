"""Deterministic fake backend used by CPU-only test suites."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ...processors import GenerationRequest, RawModelResponse, TokenCandidate, TokenPosition
from ..base import Backend
from ..utils import coerce_request


class FakeBackend(Backend):
    """Deterministic mock backend implementation for lightweight testing."""

    def generate(
        self,
        prompts: Sequence[str],
        request: GenerationRequest | Mapping[str, Any],
    ) -> list[RawModelResponse]:
        """Generate mock model responses matching the requested generation plan."""
        plan = coerce_request(request)
        requirements = plan.requirements
        if requirements.one_token:
            scores = {candidate: -float(index) for index, candidate in enumerate(requirements.candidates)}
            sampled = requirements.candidates[0]
            position = TokenPosition(
                sampled,
                scores[sampled],
                tuple(TokenCandidate(candidate, score) for candidate, score in scores.items()),
            )
            text = json.dumps({"candidates": scores})
            return [RawModelResponse(text, 1, (position,)) for _ in prompts]
        if requirements.capture_logprobs:
            payload = plan.options.get(
                "fake_response",
                {"label": "alpha", "confidence_tens": 7, "confidence_units": 5},
            )
            tens = int(payload.get("confidence_tens", 7))
            units = int(payload.get("confidence_units", 5))
            positions = (
                TokenPosition(str(tens), 0.0, (TokenCandidate(str(tens), 0.0),)),
                TokenPosition(str(units), 0.0, (TokenCandidate(str(units), 0.0),)),
            )
            return [RawModelResponse(json.dumps(payload), 2, positions) for _ in prompts]
        payload = plan.options.get("fake_response", {"label": "alpha"})
        return [RawModelResponse(json.dumps(payload), 1) for _ in prompts]
