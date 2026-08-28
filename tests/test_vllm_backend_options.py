"""Unit tests for optional vLLM backend controls."""

from __future__ import annotations

import ast
from pathlib import Path


def _backend_source() -> str:
    path = Path(__file__).parents[1] / "src/llms_experiments/backends/implementations/vllm.py"
    return path.read_text(encoding="utf-8")


def test_vllm_backend_has_model_level_structured_output_switch() -> None:
    """The switch must preserve schema validation while skipping guided decoding."""
    tree = ast.parse(_backend_source())
    source = _backend_source()
    assert 'model.get("structured_outputs", True)' in source
    assert "requirements.structured_schema is not None and self.structured_outputs_enabled" in source
    assert any(isinstance(node, ast.ClassDef) and node.name == "VLLMBackend" for node in tree.body)


def test_vllm_backend_passes_model_generation_config_to_vllm() -> None:
    """Muse needs its checkpoint generation defaults instead of runner defaults."""
    assert '"generation_config"' in _backend_source()


def test_vllm_backend_preserves_model_revisions() -> None:
    """Pinned model and tokenizer revisions must reach the vLLM engine."""
    source = _backend_source()
    assert '"revision"' in source
    assert '"tokenizer_revision"' in source


def test_vllm_backend_passes_scheduler_controls_to_vllm() -> None:
    """Throughput controls must reach the in-process vLLM constructor."""
    source = _backend_source()
    for option in (
        "max_num_batched_tokens",
        "max_num_seqs",
        "enable_chunked_prefill",
        "async_scheduling",
        "kv_cache_dtype",
        "calculate_kv_scales",
        "mamba_cache_dtype",
        "mamba_ssm_cache_dtype",
        "performance_mode",
        "optimization_level",
    ):
        assert f'"{option}"' in source


def test_async_vllm_backend_uses_final_only_bounded_dispatch() -> None:
    """The async path must defer scheduler batching to vLLM itself."""
    source = (
        Path(__file__).parents[1] / "src" / "llms_experiments" / "backends" / "implementations" / "async_vllm.py"
    ).read_text(encoding="utf-8")

    assert "AsyncLLM.from_engine_args" in source
    assert "RequestOutputKind.FINAL_ONLY" in source
    assert "max_in_flight" in source
    assert 'engine_kwargs.pop("calculate_kv_scales", None)' in source


def test_vllm_backend_rejects_non_boolean_structured_output_switch() -> None:
    """Configuration errors should be caught before the GPU engine is created."""
    assert 'raise ValueError("model.structured_outputs must be a boolean")' in _backend_source()
