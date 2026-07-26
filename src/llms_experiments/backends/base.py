"""Backend abstraction and request/response normalization helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
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
