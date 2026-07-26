"""CPU-friendly local GGUF backend powered by llama-cpp-python."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from ...processors import GenerationRequest, RawModelResponse
from ..base import Backend
from ..utils import coerce_request, conversation, extract_token_positions


class LlamaCppBackend(Backend):
    """Local backend implementation backed by llama-cpp-python."""

    def __init__(self, model: Mapping[str, Any], resource_guard: Mapping[str, Any] | None = None) -> None:
        """Initialize llama_cpp model runtime and context settings."""
        del resource_guard
        try:
            import llama_cpp
        except ImportError as exc:
            raise RuntimeError(
                "llama_cpp backend requires the llama-cpp-python package. "
                "Install it with: pip install llama-cpp-python or pip install '.[llama-cpp]'"
            ) from exc
        model_path = model.get("model_path") or model.get("path") or model.get("name")
        if not model_path:
            raise ValueError("llama_cpp model config requires 'model_path', 'path', or 'name'.")
        self.llm = llama_cpp.Llama(
            model_path=str(model_path),
            n_ctx=int(model.get("n_ctx", model.get("max_model_len", 2048))),
            n_threads=int(model.get("n_threads", os.cpu_count() or 4)),
            n_batch=int(model.get("n_batch", 512)),
            n_gpu_layers=int(model.get("n_gpu_layers", 0)),
            verbose=bool(model.get("verbose", False)),
            logits_all=True,
        )

    def _generate_one(self, prompt: str, plan: GenerationRequest) -> RawModelResponse:
        """Execute chat completion for a single prompt using llama_cpp."""
        requirements = plan.requirements
        kwargs: dict[str, Any] = {
            "messages": conversation(plan.system_prompt, prompt),
            "temperature": 0.0,
            "max_tokens": requirements.max_tokens,
        }
        if requirements.capture_logprobs:
            kwargs.update({"logprobs": True, "top_logprobs": requirements.top_logprobs})
        if requirements.structured_schema is not None:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": dict(requirements.structured_schema),
            }
        try:
            data = self.llm.create_chat_completion(**kwargs)
            choice = (data.get("choices") or [{}])[0]
            raw = str((choice.get("message") or {}).get("content") or "")
            count = int((data.get("usage") or {}).get("completion_tokens") or 0)
            positions = extract_token_positions((choice.get("logprobs") or {}).get("content"))
            return RawModelResponse(raw, count, positions)
        except Exception as exc:
            return RawModelResponse("", 0, backend_error=f"llama_cpp_error: {exc}")

    def generate(
        self,
        prompts: Sequence[str],
        request: GenerationRequest | Mapping[str, Any],
    ) -> list[RawModelResponse]:
        """Generate responses for multiple input prompts."""
        plan = coerce_request(request)
        return [self._generate_one(prompt, plan) for prompt in prompts]

    def close(self) -> None:
        """Release underlying llama_cpp instance resources."""
        if getattr(self, "llm", None) is not None:
            del self.llm
            self.llm = None
