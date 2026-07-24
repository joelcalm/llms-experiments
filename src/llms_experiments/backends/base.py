"""Backend abstraction and request/response normalization helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from ..processors import (
    GenerationRequest,
    RawModelResponse,
    ResponseRequirements,
    TokenCandidate,
    TokenPosition,
)


class BackendFailure(RuntimeError):
    """A batch-level failure that may be retried at a smaller batch size."""


class Backend(ABC):
    """Abstract transport for one model execution environment."""

    @abstractmethod
    def generate(
        self,
        prompts: Sequence[str],
        request: GenerationRequest | Mapping[str, Any],
    ) -> list[RawModelResponse]:
        """Generate one normalized response per prompt."""

    def close(self) -> None:
        """Release model or network resources."""

    def __enter__(self) -> Backend:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def coerce_request(request: GenerationRequest | Mapping[str, Any]) -> GenerationRequest:
    """Adapt the temporary internal variant mapping to a generation plan."""

    if isinstance(request, GenerationRequest):
        return request
    processor = request.get("_processor")
    if processor is not None:
        requirements = processor.requirements
    else:
        mode = str(request.get("request_mode", "generate"))
        candidates = tuple(str(item) for item in request.get("candidates", []))
        requirements = ResponseRequirements(
            max_tokens=int(request.get("max_tokens", 1 if mode == "candidate_logprobs" else 128)),
            capture_logprobs=mode in {"candidate_logprobs", "generate_with_logprobs"},
            top_logprobs=(
                min(20, len(candidates) + 5)
                if mode == "candidate_logprobs"
                else int(request.get("top_logprobs", 20))
                if mode == "generate_with_logprobs"
                else None
            ),
            structured_schema=None if mode == "candidate_logprobs" else request.get("_schema"),
            one_token=mode == "candidate_logprobs",
            candidates=candidates,
        )
    return GenerationRequest(
        variant_id=str(request.get("id", "variant")),
        requirements=requirements,
        system_prompt=request.get("_system"),
        options={key: value for key, value in request.items() if not key.startswith("_")},
    )


def conversation(system_prompt: str | None, user_prompt: str) -> list[dict[str, str]]:
    turns = [{"role": "system", "content": system_prompt}] if system_prompt else []
    return [*turns, {"role": "user", "content": user_prompt}]


def extract_token_positions(content: Any) -> tuple[TokenPosition, ...]:
    """Normalize OpenAI-shaped dict or SDK logprob objects."""

    positions: list[TokenPosition] = []
    for item in content or []:
        sampled_token = item.get("token", "") if isinstance(item, dict) else getattr(item, "token", "")
        sampled_logprob = (
            item.get("logprob", -float("inf")) if isinstance(item, dict) else getattr(item, "logprob", -float("inf"))
        )
        top = item.get("top_logprobs") if isinstance(item, dict) else getattr(item, "top_logprobs", None)
        alternatives: list[TokenCandidate] = []
        for candidate in top or []:
            token = candidate.get("token", "") if isinstance(candidate, dict) else getattr(candidate, "token", "")
            logprob = (
                candidate.get("logprob", -float("inf"))
                if isinstance(candidate, dict)
                else getattr(candidate, "logprob", -float("inf"))
            )
            normalized = TokenCandidate(str(token), float(logprob))
            if normalized not in alternatives:
                alternatives.append(normalized)
        sampled = TokenCandidate(str(sampled_token), float(sampled_logprob))
        if sampled.token and sampled not in alternatives:
            alternatives.append(sampled)
        positions.append(
            TokenPosition(
                token=sampled.token,
                logprob=sampled.logprob,
                alternatives=tuple(alternatives),
            )
        )
    return tuple(positions)
