"""Backend interfaces. vLLM is imported lazily so validation/tests need no GPU stack."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from logging_utils import gpu_snapshot


class BackendFailure(RuntimeError):
    pass


def synchronize_cuda(enabled: bool) -> None:
    """Synchronise only when requested; normal throughput runs remain asynchronous."""
    if not enabled:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


@dataclass
class Response:
    raw: str
    token_count: int
    candidate_logprobs: dict[str, float] | None = None


class FakeBackend:
    """Deterministic backend used by tests and offline contract demonstrations."""
    def generate(self, prompts: list[str], variant: dict[str, Any]) -> list[Response]:
        if variant["request_mode"] == "candidate_logprobs":
            candidates = variant["candidates"]
            return [Response(json.dumps({"candidates": {candidate: -float(index) for index, candidate in enumerate(candidates)}}), 1,
                             {candidate: -float(index) for index, candidate in enumerate(candidates)}) for _ in prompts]
        payload = variant.get("fake_response", {"label": "alpha"})
        return [Response(json.dumps(payload), 1) for _ in prompts]

    def close(self) -> None:
        return None


class LocalVLLMBackend:
    def __init__(self, model: dict[str, Any]) -> None:
        status = gpu_snapshot()
        if not status["available"]:
            raise RuntimeError("GPU preflight failed: nvidia-smi cannot communicate with an NVIDIA driver. Run this configuration on the intended local GPU environment.")
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("local_vllm requires vLLM. Install it in the execution environment before running.") from exc
        self._sampling_params = __import__("vllm", fromlist=["SamplingParams"]).SamplingParams
        kwargs = {"model": model["name"], "gpu_memory_utilization": model.get("gpu_memory_utilization", 0.9),
                  "max_model_len": model.get("max_model_len", 2048), "max_num_seqs": model.get("max_num_seqs", 128),
                  "enable_prefix_caching": model.get("enable_prefix_caching", True)}
        self.llm = LLM(**kwargs)

    def generate(self, prompts: list[str], variant: dict[str, Any]) -> list[Response]:
        mode = variant["request_mode"]
        if mode == "candidate_logprobs":
            prompts = [prompt + "\n\nAnswer with exactly one candidate token:" for prompt in prompts]
            params = self._sampling_params(temperature=0, max_tokens=1, logprobs=max(20, len(variant["candidates"]) + 5))
        else:
            sampling_kwargs: dict[str, Any] = {"temperature": 0, "max_tokens": variant.get("max_tokens", 128)}
            schema = variant.get("_json_schema")
            if schema:
                from vllm.sampling_params import StructuredOutputsParams
                sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                    json=schema,
                    disable_any_whitespace=True,
                    disable_additional_properties=True,
                )
            params = self._sampling_params(**sampling_kwargs)
        try:
            outputs = self.llm.generate(prompts, params, use_tqdm=False)
        except Exception as exc:
            message = str(exc)
            if any(token in message.lower() for token in ("out of memory", "oom", "context length", "max model len")):
                raise BackendFailure(message) from exc
            raise
        responses: list[Response] = []
        for output in outputs:
            generated = output.outputs[0]
            scores: dict[str, float] | None = None
            if mode == "candidate_logprobs":
                top = (generated.logprobs or [{}])[0] or {}
                observed: dict[str, float] = {}
                for token, logprob in top.items():
                    decoded = getattr(logprob, "decoded_token", None)
                    token_text = str(decoded if decoded is not None else token).strip()
                    observed[token_text] = float(getattr(logprob, "logprob", logprob))
                scores = {candidate: observed.get(str(candidate).strip(), -float("inf")) for candidate in variant["candidates"]}
                raw = json.dumps({"candidates": scores})
            else:
                raw = generated.text
            responses.append(Response(raw=raw, token_count=len(generated.token_ids), candidate_logprobs=scores))
        return responses

    def close(self) -> None:
        del self.llm


def create_backend(model: dict[str, Any]) -> Any:
    return FakeBackend() if model.get("backend") == "fake" else LocalVLLMBackend(model)
