"""Pluggable inference backends."""

from ..processors import GenerationRequest, RawModelResponse
from .backends import (
    AsyncVLLMBackend,
    FakeBackend,
    LlamaCppBackend,
    OpenAICompatibleBackend,
    VLLMBackend,
)
from .base import AsyncStreamingBackend, Backend, BackendFailure
from .factory import BACKEND_TYPES, create_backend

make_backend = create_backend
Response = RawModelResponse

__all__ = [
    "BACKEND_TYPES",
    "AsyncStreamingBackend",
    "AsyncVLLMBackend",
    "Backend",
    "BackendFailure",
    "FakeBackend",
    "GenerationRequest",
    "LlamaCppBackend",
    "OpenAICompatibleBackend",
    "RawModelResponse",
    "Response",
    "VLLMBackend",
    "create_backend",
    "make_backend",
]
