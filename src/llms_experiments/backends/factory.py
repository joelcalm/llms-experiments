"""Explicit backend factory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import Backend
from .implementations.fake import FakeBackend
from .implementations.llama_cpp import LlamaCppBackend
from .implementations.openai_compatible import OpenAICompatibleBackend
from .implementations.vllm import VLLMBackend

BACKEND_TYPES: dict[str, type[Backend]] = {
    "fake": FakeBackend,
    "openai_compatible": OpenAICompatibleBackend,
    "llama_cpp": LlamaCppBackend,
    "local_vllm": VLLMBackend,
}


def create_backend(
    model: Mapping[str, Any],
    resource_guard: Mapping[str, Any] | None = None,
) -> Backend:
    """Create a backend instance matching the model configuration backend name."""
    name = str(model.get("backend", ""))
    try:
        backend_type = BACKEND_TYPES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(BACKEND_TYPES))
        raise ValueError(f"unsupported backend {name!r}; expected one of: {supported}") from exc
    return backend_type(model, resource_guard) if name != "fake" else backend_type()  # type: ignore[call-arg]
