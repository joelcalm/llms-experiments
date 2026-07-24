"""Vendor-neutral OpenAI-compatible HTTP backend."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..processors import GenerationRequest, RawModelResponse
from .base import Backend, coerce_request, conversation, extract_token_positions


class OpenAICompatibleBackend(Backend):
    def __init__(self, model: Mapping[str, Any], resource_guard: Mapping[str, Any] | None = None) -> None:
        del resource_guard
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("openai_compatible requires the requests package.") from exc
        self.requests = requests
        self.model = dict(model)
        self.url = str(model.get("api_base_url", "http://127.0.0.1:8000/v1/chat/completions"))
        key_name = str(model.get("api_key_env", "OPENAI_API_KEY"))
        self.api_key = os.environ.get(key_name)
        if not self.api_key:
            raise RuntimeError(f"openai_compatible requires the {key_name} environment variable.")
        self.timeout_seconds = float(model.get("api_timeout_seconds", 120))
        self.concurrency = max(1, int(model.get("api_concurrency", 4)))
        self.http_retries = max(0, int(model.get("api_http_retries", 2)))
        self.structured_outputs = bool(model.get("api_structured_outputs", True))
        template_kwargs = model.get("chat_template_kwargs", {})
        if not isinstance(template_kwargs, dict):
            raise ValueError("model.chat_template_kwargs must be a mapping")
        self.chat_template_kwargs = dict(template_kwargs)

    def _generate_one(self, prompt: str, plan: GenerationRequest) -> RawModelResponse:
        requirements = plan.requirements
        payload: dict[str, Any] = {
            "model": self.model["name"],
            "messages": conversation(plan.system_prompt, prompt),
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": self.chat_template_kwargs,
            "max_tokens": requirements.max_tokens,
        }
        if requirements.capture_logprobs:
            payload.update({"logprobs": True, "top_logprobs": requirements.top_logprobs})
        if requirements.structured_schema is not None and self.structured_outputs:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": plan.variant_id,
                    "schema": dict(requirements.structured_schema),
                    "strict": True,
                },
            }
        error = "unknown OpenAI-compatible endpoint failure"
        for attempt in range(self.http_retries + 1):
            try:
                response = self.requests.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 200:
                    data = response.json()
                    choice = (data.get("choices") or [{}])[0]
                    raw = str((choice.get("message") or {}).get("content") or "")
                    count = int((data.get("usage") or {}).get("completion_tokens") or 0)
                    positions = extract_token_positions((choice.get("logprobs") or {}).get("content"))
                    return RawModelResponse(raw, count, positions)
                error = f"http_{response.status_code}: {response.text[:500]}"
            except Exception as exc:
                error = f"http_exception: {exc}"
            if attempt < self.http_retries:
                time.sleep(min(8, 2**attempt))
        return RawModelResponse("", 0, backend_error=error)

    def generate(
        self,
        prompts: Sequence[str],
        request: GenerationRequest | Mapping[str, Any],
    ) -> list[RawModelResponse]:
        plan = coerce_request(request)
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(prompts))) as executor:
            return list(executor.map(lambda prompt: self._generate_one(prompt, plan), prompts))
