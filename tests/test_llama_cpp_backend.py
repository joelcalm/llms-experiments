"""Tests for the CPU-friendly LlamaCppBackend."""

from __future__ import annotations

import json
from typing import Any

import pytest
from llms_experiments._core import LlamaCppBackend, make_backend


class DummyLlama:
    """Mock Llama instance simulating llama-cpp-python's create_chat_completion."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        req_mode_has_candidates = "max_tokens" in kwargs and kwargs["max_tokens"] == 1
        if req_mode_has_candidates:
            return {
                "choices": [
                    {
                        "message": {"content": "A"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "A",
                                    "logprob": -0.2,
                                    "top_logprobs": [
                                        {"token": "A", "logprob": -0.2},
                                        {"token": "B", "logprob": -3.5},
                                    ],
                                }
                            ]
                        },
                    }
                ],
                "usage": {"completion_tokens": 1},
            }

        return {
            "choices": [{"message": {"content": json.dumps({"label": "care"})}, "logprobs": None}],
            "usage": {"completion_tokens": 5},
        }


def test_make_backend_creates_llama_cpp_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import ModuleType

    dummy_module = ModuleType("llama_cpp")
    dummy_module.Llama = DummyLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", dummy_module)

    model_config = {
        "name": "gemma-2-2b-it.gguf",
        "backend": "llama_cpp",
        "model_path": "/path/to/model.gguf",
        "n_ctx": 4096,
        "n_threads": 4,
    }

    backend = make_backend(model_config)
    assert isinstance(backend, LlamaCppBackend)

    # Test generation mode
    responses = backend.generate(
        ["Evaluate this text."],
        {"request_mode": "generate", "id": "var1"},
    )
    assert len(responses) == 1
    assert responses[0].backend_error is None
    assert json.loads(responses[0].text) == {"label": "care"}

    # Test candidate logprobs mode
    candidate_responses = backend.generate(
        ["Choose A or B."],
        {"request_mode": "candidate_logprobs", "id": "var2", "candidates": ["A", "B"]},
    )
    assert len(candidate_responses) == 1
    assert candidate_responses[0].candidate_logprobs == {"A": -0.2, "B": -3.5}

    backend.close()


def test_llama_cpp_backend_missing_package_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "llama_cpp", None)

    model_config = {
        "name": "model.gguf",
        "backend": "llama_cpp",
        "model_path": "/path/to/model.gguf",
    }

    with pytest.raises(RuntimeError, match="llama_cpp backend requires the llama-cpp-python package"):
        make_backend(model_config)
