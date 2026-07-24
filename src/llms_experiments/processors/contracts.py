"""Typed contracts shared by response processors and inference backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

ErrorCategory = Literal["backend", "parsing", "validation", "processing"]


@dataclass(frozen=True)
class TokenCandidate:
    """One token alternative returned at a generated position."""

    token: str
    logprob: float


@dataclass(frozen=True)
class TokenPosition:
    """The sampled token and alternatives at one generated position."""

    token: str
    logprob: float
    alternatives: tuple[TokenCandidate, ...] = ()


@dataclass(frozen=True)
class RawModelResponse:
    """Backend-neutral evidence returned for one prompt."""

    text: str
    token_count: int
    token_positions: tuple[TokenPosition, ...] = ()
    backend_error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Transitional compatibility for callers that still provide already
    # aggregated scores. New backends leave this unset.
    candidate_scores: Mapping[str, float] | None = None

    @property
    def raw(self) -> str:
        """Compatibility alias for the pre-refactor backend response."""

        return self.text

    @property
    def candidate_logprobs(self) -> dict[str, float] | None:
        return dict(self.candidate_scores) if self.candidate_scores is not None else None

    @property
    def token_logprobs(self) -> list[dict[str, Any]] | None:
        if not self.token_positions:
            return None
        return [
            {
                "token": position.token,
                "logprob": position.logprob,
                "top_logprobs": [(candidate.token, candidate.logprob) for candidate in position.alternatives],
            }
            for position in self.token_positions
        ]


@dataclass(frozen=True)
class ResponseRequirements:
    """Evidence and generation behavior required by a processor pipeline."""

    max_tokens: int = 128
    capture_logprobs: bool = False
    top_logprobs: int | None = None
    structured_schema: Mapping[str, Any] | None = None
    one_token: bool = False
    candidates: tuple[str, ...] = ()

    def merge(self, other: ResponseRequirements) -> ResponseRequirements:
        """Merge compatible requirement fragments from ordered stages."""

        if self.structured_schema is not None and other.structured_schema is not None:
            if dict(self.structured_schema) != dict(other.structured_schema):
                raise ValueError("processor stages declare conflicting structured-output schemas")
        schema = self.structured_schema or other.structured_schema
        one_token = self.one_token or other.one_token
        if one_token and schema is not None:
            raise ValueError("one-token candidate extraction cannot be combined with structured JSON generation")
        candidates = self.candidates or other.candidates
        if self.candidates and other.candidates and self.candidates != other.candidates:
            raise ValueError("processor stages declare conflicting candidate sets")
        requested_top = max(value for value in (self.top_logprobs, other.top_logprobs, 0) if value is not None) or None
        if requested_top is not None and not 1 <= requested_top <= 20:
            raise ValueError("top_logprobs must be between 1 and 20")
        return ResponseRequirements(
            max_tokens=1 if one_token else max(self.max_tokens, other.max_tokens),
            capture_logprobs=self.capture_logprobs or other.capture_logprobs,
            top_logprobs=requested_top,
            structured_schema=schema,
            one_token=one_token,
            candidates=candidates,
        )


@dataclass(frozen=True)
class GenerationRequest:
    """One backend-neutral generation plan compiled from processor stages."""

    variant_id: str
    requirements: ResponseRequirements
    system_prompt: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Temporary mapping-style access for v0.2 custom backends."""

        compatibility = {
            "id": self.variant_id,
            "_system": self.system_prompt,
            "_schema": self.requirements.structured_schema,
            "max_tokens": self.requirements.max_tokens,
            "top_logprobs": self.requirements.top_logprobs,
            "candidates": list(self.requirements.candidates),
            "request_mode": (
                "candidate_logprobs"
                if self.requirements.one_token
                else "generate_with_logprobs"
                if self.requirements.capture_logprobs
                else "generate"
            ),
        }
        if key in compatibility:
            return compatibility[key]
        return self.options.get(key, default)


@dataclass(frozen=True)
class ProcessingError:
    """A stable, typed failure emitted by a backend or processing stage."""

    code: str
    message: str
    stage: str
    category: ErrorCategory
    retryable: bool = False

    def as_contract_string(self) -> str:
        """Render the existing result-contract error representation."""

        if self.category == "backend":
            return f"backend_error: {self.message}"
        return f"{self.code}: {self.message}"


@dataclass(frozen=True)
class ProcessorContext:
    """Static and row-specific values available to processor stages."""

    variant_id: str
    result_type: str
    root: Path
    row: Mapping[str, Any] | None = None
    dataset_labels: tuple[str, ...] = ()
    code_labels: Mapping[str, str] = field(default_factory=dict)

    def with_row(self, row: Mapping[str, Any]) -> ProcessorContext:
        return replace(self, row=row)

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path


@dataclass(frozen=True)
class ProcessingState:
    """Immutable state passed through the ordered stage pipeline."""

    response: RawModelResponse
    value: Any = None
    candidate_scores: Mapping[str, float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    errors: tuple[ProcessingError, ...] = ()

    def fail(self, error: ProcessingError) -> ProcessingState:
        return replace(self, errors=(*self.errors, error))


@dataclass(frozen=True)
class ProcessedResult:
    """The format-independent result produced by one processor pipeline."""

    result_type: str
    value: Any
    candidate_scores: Mapping[str, float] | None
    target_label: str | None
    errors: tuple[ProcessingError, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors
