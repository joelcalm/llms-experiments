"""Backend abstraction and request/response normalization helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from typing import Any

from ..processors import GenerationRequest, RawModelResponse


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
        """Enter context management block for model execution."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit context management block and release backend resources."""
        self.close()


class AsyncStreamingBackend(Backend):
    """Backend that admits individual requests to a continuously batched engine.

    ``generate_stream`` deliberately receives a lazy iterable.  Implementations
    must consume no more than ``max_in_flight`` items ahead of the caller and
    yield each final response as soon as the engine completes it.  The caller
    can therefore publish durable output parts without retaining a whole input
    corpus or waiting for a user-defined batch barrier.
    """

    @abstractmethod
    def generate_stream(
        self,
        prompts: Iterable[tuple[str, str]],
        request: GenerationRequest | Mapping[str, Any],
        *,
        max_in_flight: int,
    ) -> AsyncIterator[tuple[str, RawModelResponse]]:
        """Yield final responses for lazily supplied request identifiers and prompts."""
