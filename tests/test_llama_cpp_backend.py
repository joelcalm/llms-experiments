"""Comprehensive tests for the CPU-friendly LlamaCppBackend across all 7 inference strategies."""

from __future__ import annotations

import json
from typing import Any

import pytest
from llms_experiments._core import LlamaCppBackend, make_backend


class DummyLlama:
    """Mock Llama instance simulating llama-cpp-python's create_chat_completion across all strategy payloads."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        req_mode_has_candidates = "max_tokens" in kwargs and kwargs["max_tokens"] == 1
        logprobs = kwargs.get("logprobs")

        # Candidate logprobs strategies (single_label_code_logits, independent_yes_no_logits, soft_multi_label_yes_no_logits)
        if req_mode_has_candidates:
            top_logprobs = kwargs.get("top_logprobs", 5)
            if top_logprobs == 2:  # yes/no logits
                content = [
                    {
                        "token": "yes",
                        "logprob": -0.15,
                        "top_logprobs": [
                            {"token": "yes", "logprob": -0.15},
                            {"token": "no", "logprob": -4.20},
                        ],
                    }
                ]
            else:  # code logits (A, B, C)
                content = [
                    {
                        "token": "A",
                        "logprob": -0.25,
                        "top_logprobs": [
                            {"token": "A", "logprob": -0.25},
                            {"token": "B", "logprob": -2.10},
                            {"token": "C", "logprob": -6.50},
                        ],
                    }
                ]
            return {
                "choices": [{"message": {"content": "A"}, "logprobs": {"content": content}}],
                "usage": {"completion_tokens": 1},
            }

        # Verbalized confidence strategy (generate_with_logprobs)
        if logprobs:
            return {
                "choices": [
                    {
                        "message": {"content": json.dumps({"label": "care", "confidence_tens": 8, "confidence_units": 5})},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "8",
                                    "logprob": -0.1,
                                    "top_logprobs": [
                                        {"token": "8", "logprob": -0.1},
                                        {"token": "7", "logprob": -2.5},
                                    ],
                                },
                                {
                                    "token": "5",
                                    "logprob": -0.2,
                                    "top_logprobs": [
                                        {"token": "5", "logprob": -0.2},
                                        {"token": "0", "logprob": -3.1},
                                    ],
                                },
                            ]
                        },
                    }
                ],
                "usage": {"completion_tokens": 12},
            }

        # Response format schema structured outputs
        resp_fmt = kwargs.get("response_format", {})
        schema = resp_fmt.get("schema", {})

        if "labels" in schema.get("properties", {}):
            payload = {"labels": ["care", "purity"]}
        elif "score" in schema.get("properties", {}):
            payload = {"score": 4}
        else:
            payload = {"label": "fairness"}

        return {
            "choices": [{"message": {"content": json.dumps(payload)}, "logprobs": None}],
            "usage": {"completion_tokens": 8},
        }


@pytest.fixture
def mock_llama_backend(monkeypatch: pytest.MonkeyPatch) -> LlamaCppBackend:
    import sys
    from types import ModuleType

    dummy_module = ModuleType("llama_cpp")
    dummy_module.Llama = DummyLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", dummy_module)

    model_config = {
        "name": "gemma-2-2b-it.gguf",
        "backend": "llama_cpp",
        "model_path": "/models/gemma-2-2b-it.gguf",
        "n_ctx": 2048,
        "n_threads": 4,
    }
    backend = make_backend(model_config)
    assert isinstance(backend, LlamaCppBackend)
    return backend


def test_strategy_1_single_label_json(mock_llama_backend: LlamaCppBackend) -> None:
    variant = {
        "id": "single_label_var",
        "request_mode": "generate",
        "result_type": "single_label",
        "_schema": {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": ["care", "fairness", "loyalty"]}},
            "required": ["label"],
        },
    }
    responses = mock_llama_backend.generate(["Classify this input."], variant)
    assert len(responses) == 1
    assert responses[0].backend_error is None
    assert json.loads(responses[0].text) == {"label": "fairness"}


def test_strategy_2_multi_label_json(mock_llama_backend: LlamaCppBackend) -> None:
    variant = {
        "id": "multi_label_var",
        "request_mode": "generate",
        "result_type": "multi_label",
        "_schema": {
            "type": "object",
            "properties": {"labels": {"type": "array", "items": {"type": "string"}}},
            "required": ["labels"],
        },
    }
    responses = mock_llama_backend.generate(["Classify all applicable labels."], variant)
    assert len(responses) == 1
    assert responses[0].backend_error is None
    assert json.loads(responses[0].text) == {"labels": ["care", "purity"]}


def test_strategy_3_ordinal_score_json(mock_llama_backend: LlamaCppBackend) -> None:
    variant = {
        "id": "ordinal_score_var",
        "request_mode": "generate",
        "result_type": "ordinal_score",
        "_schema": {
            "type": "object",
            "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
            "required": ["score"],
        },
    }
    responses = mock_llama_backend.generate(["Rate intensity from 1 to 5."], variant)
    assert len(responses) == 1
    assert responses[0].backend_error is None
    assert json.loads(responses[0].text) == {"score": 4}


def test_strategy_4_single_label_code_logits(mock_llama_backend: LlamaCppBackend) -> None:
    variant = {
        "id": "code_logits_var",
        "request_mode": "candidate_logprobs",
        "result_type": "categorical_logprobs",
        "candidates": ["A", "B", "C"],
    }
    responses = mock_llama_backend.generate(["Select code A, B, or C."], variant)
    assert len(responses) == 1
    assert responses[0].backend_error is None
    assert responses[0].candidate_logprobs == {"A": -0.25, "B": -2.10, "C": -6.50}


def test_strategy_5_independent_yes_no_logits(mock_llama_backend: LlamaCppBackend) -> None:
    variant = {
        "id": "yes_no_logits_var",
        "request_mode": "candidate_logprobs",
        "result_type": "fixed_binary_probe",
        "candidates": ["yes", "no"],
    }
    responses = mock_llama_backend.generate(["Does this express care? (yes/no)"], variant)
    assert len(responses) == 1
    assert responses[0].backend_error is None
    assert responses[0].candidate_logprobs == {"yes": -0.15, "no": -4.20}


def test_strategy_6_soft_multi_label_yes_no_logits(mock_llama_backend: LlamaCppBackend) -> None:
    variant = {
        "id": "soft_multi_label_var",
        "request_mode": "candidate_logprobs",
        "result_type": "label_yes_no_logprobs",
        "candidates": ["yes", "no"],
        "expand_over": "dataset_labels",
    }
    responses = mock_llama_backend.generate(["Does this text express care?"], variant)
    assert len(responses) == 1
    assert responses[0].backend_error is None
    assert responses[0].candidate_logprobs == {"yes": -0.15, "no": -4.20}


def test_strategy_7_verbalized_confidence(mock_llama_backend: LlamaCppBackend) -> None:
    variant = {
        "id": "verbalized_confidence_var",
        "request_mode": "generate_with_logprobs",
        "result_type": "single_label_verbalized_confidence",
        "top_logprobs": 20,
    }
    responses = mock_llama_backend.generate(["Classify and state confidence."], variant)
    assert len(responses) == 1
    assert responses[0].backend_error is None
    parsed = json.loads(responses[0].text)
    assert parsed["label"] == "care"
    assert parsed["confidence_tens"] == 8
    assert parsed["confidence_units"] == 5
    assert responses[0].token_logprobs is not None
    assert len(responses[0].token_logprobs) == 2
    assert responses[0].token_logprobs[0]["token"] == "8"
    assert responses[0].token_logprobs[1]["token"] == "5"


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
