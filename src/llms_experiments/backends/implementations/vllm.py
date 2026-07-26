"""In-process vLLM backend."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from ...processors import GenerationRequest, RawModelResponse, TokenCandidate, TokenPosition
from ...runtime import configure_torch_cpu_threads, configure_vllm_environment, gpu
from ..base import Backend, BackendFailure
from ..utils import coerce_request, conversation


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
            "mm_encoder_attn_backend",
            "attention_backend",
            "moe_backend",
        ):
            if model.get(option) is not None:
                kwargs[option] = model[option]
        self.llm = LLM(**kwargs)
        template_kwargs = model.get("chat_template_kwargs", {})
        if not isinstance(template_kwargs, dict):
            raise ValueError("model.chat_template_kwargs must be a mapping")
        self.chat_template_kwargs = dict(template_kwargs)

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
        if requirements.structured_schema is not None:
            from vllm.sampling_params import StructuredOutputsParams

            kwargs["structured_outputs"] = StructuredOutputsParams(
                json=dict(requirements.structured_schema),
                disable_any_whitespace=True,
                disable_additional_properties=True,
            )
        params = self.params(**kwargs)
        try:
            outputs = self.llm.chat(
                [conversation(plan.system_prompt, prompt) for prompt in prompts],
                params,
                use_tqdm=False,
                chat_template_kwargs=self.chat_template_kwargs or None,
            )
        except Exception as exc:
            if any(word in str(exc).lower() for word in ("out of memory", "oom", "context length", "max model len")):
                raise BackendFailure(str(exc)) from exc
            raise
        responses: list[RawModelResponse] = []
        for output in outputs:
            generated = output.outputs[0]
            positions: list[TokenPosition] = []
            for index, candidates in enumerate(generated.logprobs or []):
                sampled_id = generated.token_ids[index]
                sampled = candidates.get(sampled_id)
                alternatives = tuple(
                    TokenCandidate(
                        str(getattr(logprob, "decoded_token", token_id)),
                        float(getattr(logprob, "logprob", logprob)),
                    )
                    for token_id, logprob in candidates.items()
                )
                positions.append(
                    TokenPosition(
                        token=str(getattr(sampled, "decoded_token", sampled_id)),
                        logprob=float(getattr(sampled, "logprob", sampled)),
                        alternatives=alternatives,
                    )
                )
            responses.append(
                RawModelResponse(
                    text=generated.text,
                    token_count=len(generated.token_ids),
                    token_positions=tuple(positions),
                )
            )
        return responses

    def close(self) -> None:
        """Shut down vLLM engine instance and release GPU memory resources."""
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        del self.llm
