"""In-process vLLM backend."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ...processors import GenerationRequest, RawModelResponse, TokenCandidate, TokenPosition
from ...runtime import configure_torch_cpu_threads, configure_vllm_environment, gpu
from ..base import Backend, BackendFailure
from ..utils import coerce_request, conversation


def vllm_engine_kwargs(model: Mapping[str, Any]) -> dict[str, Any]:
    """Return the reproducible engine options shared by vLLM backends.

    Options are included only when explicitly configured, apart from the
    repository's safe scheduling defaults.  Keeping this construction shared
    prevents a streaming run from silently using different model, cache, or
    kernel settings than the synchronous reference path.
    """

    kwargs: dict[str, Any] = {
        "model": model["name"],
        "gpu_memory_utilization": model.get("gpu_memory_utilization", 0.9),
        "max_model_len": model.get("max_model_len", 2048),
        "max_num_seqs": model.get("max_num_seqs", 128),
        "enable_prefix_caching": model.get("enable_prefix_caching", True),
    }
    for option in (
        "language_model_only",
        "limit_mm_per_prompt",
        "enforce_eager",
        "compilation_config",
        "tokenizer_mode",
        "config_format",
        "load_format",
        "quantization",
        "dtype",
        "tensor_parallel_size",
        "trust_remote_code",
        "model_impl",
        "chat_template",
        "revision",
        "tokenizer_revision",
        "mm_encoder_attn_backend",
        "attention_backend",
        "moe_backend",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "async_scheduling",
        "kv_cache_dtype",
        "calculate_kv_scales",
        "mamba_cache_dtype",
        "mamba_ssm_cache_dtype",
        "gdn_prefill_backend",
        "generation_config",
        "reasoning_parser",
        "performance_mode",
        "optimization_level",
    ):
        if model.get(option) is not None:
            kwargs[option] = model[option]
    return kwargs


def raw_response_from_vllm_output(output: Any) -> RawModelResponse:
    """Convert one final vLLM output into the backend-neutral result type."""

    generated = output.outputs[0]
    positions: list[TokenPosition] = []
    logprobs = generated.logprobs
    if logprobs is not None:
        for index, candidates in enumerate(logprobs):
            if candidates is None:
                continue
            sampled_id = generated.token_ids[index]
            sampled: Any = candidates.get(sampled_id)
            alternatives = tuple(
                TokenCandidate(
                    str(getattr(logprob_value, "decoded_token", token_id)),
                    float(getattr(logprob_value, "logprob", logprob_value)),
                )
                for token_id, logprob in candidates.items()
                for logprob_value in (cast(Any, logprob),)
            )
            positions.append(
                TokenPosition(
                    token=str(getattr(sampled, "decoded_token", sampled_id)),
                    logprob=float(getattr(sampled, "logprob", sampled)),
                    alternatives=alternatives,
                )
            )
    return RawModelResponse(
        text=generated.text,
        token_count=len(generated.token_ids),
        token_positions=tuple(positions),
    )


class VLLMBackend(Backend):
    """High-throughput in-process GPU backend powered by vLLM."""

    def __init__(self, model: Mapping[str, Any], resource_guard: Mapping[str, Any] | None = None) -> None:
        """Verify GPU availability and initialize vLLM engine instance."""
        if not gpu().get("available"):
            raise RuntimeError(
                "GPU preflight failed: nvidia-smi cannot communicate with an NVIDIA driver. "
                "Run `llms-experiments doctor` for installation and driver diagnostics."
            )
        import multiprocessing as mp

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        self.vllm_environment = configure_vllm_environment(model)
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "local_vllm requires vLLM from the optional `gpu` extra. "
                "Install with `pip install 'llms-experiments[gpu]'` or `uv sync --extra gpu`."
            ) from exc
        configure_torch_cpu_threads(resource_guard)
        self.params = SamplingParams
        structured_outputs = model.get("structured_outputs", True)
        if not isinstance(structured_outputs, bool):
            raise ValueError("model.structured_outputs must be a boolean")
        self.structured_outputs_enabled = structured_outputs
        self.llm = LLM(**vllm_engine_kwargs(model))
        template_kwargs = model.get("chat_template_kwargs", {})
        if not isinstance(template_kwargs, dict):
            raise ValueError("model.chat_template_kwargs must be a mapping")
        self.chat_template_kwargs = dict(template_kwargs)
        # vLLM's chat-template resolution at generation time never reads back
        # the `chat_template` passed to LLM()'s constructor -- it only
        # consults the per-call `chat_template` argument to `.chat()`. Load
        # the template content once here so `generate()` can pass it
        # explicitly on every call.
        from vllm.entrypoints.chat_utils import load_chat_template

        self.chat_template = load_chat_template(model.get("chat_template"))

    def generate(
        self,
        prompts: Sequence[str],
        request: GenerationRequest | Mapping[str, Any],
    ) -> list[RawModelResponse]:
        """Execute batched inference through vLLM engine."""
        plan = coerce_request(request)
        requirements = plan.requirements
        kwargs: dict[str, Any] = {"temperature": 0, "max_tokens": requirements.max_tokens}
        if requirements.capture_logprobs:
            kwargs["logprobs"] = requirements.top_logprobs
        if requirements.structured_schema is not None and self.structured_outputs_enabled:
            from vllm.sampling_params import StructuredOutputsParams

            kwargs["structured_outputs"] = StructuredOutputsParams(
                json=dict(requirements.structured_schema),
                disable_any_whitespace=True,
                disable_additional_properties=True,
            )
        params = self.params(**kwargs)
        try:
            outputs = self.llm.chat(
                cast(Any, [conversation(plan.system_prompt, prompt) for prompt in prompts]),
                params,
                use_tqdm=False,
                chat_template=cast(Any, self.chat_template),
                chat_template_kwargs=self.chat_template_kwargs or None,
            )
        except Exception as exc:
            if any(word in str(exc).lower() for word in ("out of memory", "oom", "context length", "max model len")):
                raise BackendFailure(str(exc)) from exc
            raise
        return [raw_response_from_vllm_output(output) for output in outputs]

    def close(self) -> None:
        """Shut down vLLM engine instance and release GPU memory resources."""
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        del self.llm
