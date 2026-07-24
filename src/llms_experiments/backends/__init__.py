"""Pluggable inference backends."""

from ..processors import GenerationRequest, RawModelResponse
from .base import Backend, BackendFailure
from .factory import BACKEND_TYPES, create_backend
from .fake import FakeBackend
from .llama_cpp import LlamaCppBackend
from .openai_compatible import OpenAICompatibleBackend
from .vllm import VLLMBackend

make_backend = create_backend
Response = RawModelResponse

__all__ = [
    "BACKEND_TYPES",
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
