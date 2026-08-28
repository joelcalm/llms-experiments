"""Bounded asynchronous vLLM backend for continuously batched inference."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from typing import Any, cast

from ...processors import GenerationRequest, RawModelResponse
from ...runtime import configure_torch_cpu_threads, configure_vllm_environment, gpu
from ..base import AsyncStreamingBackend, BackendFailure
from ..utils import coerce_request, conversation
from .vllm import raw_response_from_vllm_output, vllm_engine_kwargs


class AsyncVLLMBackend(AsyncStreamingBackend):
    """Use vLLM's ``AsyncLLM`` engine without application-level batch barriers.

    The backend admits requests lazily up to a bounded producer queue.  vLLM
    owns the actual scheduler batch and continuously fills GPU slots as each
    short classification request completes.
    """

    def __init__(self, model: Mapping[str, Any], resource_guard: Mapping[str, Any] | None = None) -> None:
        """Create one asynchronously scheduled vLLM engine for a local GPU."""

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
            from vllm import SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.entrypoints.chat_utils import load_chat_template
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError as exc:
            raise RuntimeError(
                "local_vllm_async requires a vLLM installation with AsyncLLM. "
                "Install the optional GPU environment before selecting this backend."
            ) from exc
        configure_torch_cpu_threads(resource_guard)
        self.params = SamplingParams
        engine_kwargs = vllm_engine_kwargs(model)
        # Chat templates belong to the per-request renderer.  ``LLM`` accepts
        # the convenience constructor argument, while ``AsyncEngineArgs`` does
        # not, so pass the same template explicitly in ``_render_chat``.
        engine_kwargs.pop("chat_template", None)
        # ``calculate_kv_scales`` was accepted by the synchronous constructor
        # but is not an ``AsyncEngineArgs`` field in vLLM 0.28.  The current
        # campaign uses fixed FP8 cache scales, so omitting it preserves the
        # requested default and keeps the dedicated async runtime compatible.
        engine_kwargs.pop("calculate_kv_scales", None)
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))
        template_kwargs = model.get("chat_template_kwargs", {})
        if not isinstance(template_kwargs, dict):
            raise ValueError("model.chat_template_kwargs must be a mapping")
        self.chat_template_kwargs = dict(template_kwargs)
        self.chat_template = load_chat_template(model.get("chat_template"))
        self._closed = False

    def _sampling_params(self, request: GenerationRequest) -> Any:
        """Build deterministic final-only sampling parameters for one variant."""

        requirements = request.requirements
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
        try:
            from vllm.sampling_params import RequestOutputKind

            kwargs["output_kind"] = RequestOutputKind.FINAL_ONLY
        except ImportError:
            # Older vLLM versions yield final outputs when no streaming client
            # is attached; the dedicated 0.28 runtime used in production takes
            # the explicit final-only path above.
            pass
        return self.params(**kwargs)

    def _render_chat(self, request: GenerationRequest, prompt: str) -> str:
        """Render one identical chat prompt before it enters the async engine."""

        tokenizer = self.engine.get_tokenizer()
        return cast(
            str,
            tokenizer.apply_chat_template(
                cast(Any, conversation(request.system_prompt, prompt)),
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self.chat_template,
                **self.chat_template_kwargs,
            ),
        )

    async def generate_stream(
        self,
        prompts: Iterable[tuple[str, str]],
        request: GenerationRequest | Mapping[str, Any],
        *,
        max_in_flight: int,
    ) -> AsyncIterator[tuple[str, RawModelResponse]]:
        """Yield final requests while retaining at most ``max_in_flight`` tasks."""

        if max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        plan = coerce_request(request)
        params = self._sampling_params(plan)
        iterator = iter(prompts)
        pending: dict[asyncio.Task[tuple[str, RawModelResponse]], str] = {}

        async def infer(
            request_id: str,
            prompt: str,
            engine_request_id: str,
        ) -> tuple[str, RawModelResponse]:
            final: Any | None = None
            try:
                async for output in self.engine.generate(
                    self._render_chat(plan, prompt),
                    params,
                    request_id=engine_request_id,
                ):
                    final = output
            except Exception as exc:
                message = str(exc).lower()
                if any(term in message for term in ("out of memory", "oom", "context length", "max model len")):
                    raise BackendFailure(str(exc)) from exc
                raise
            if final is None:
                raise BackendFailure(f"vLLM produced no final output for request {request_id}")
            return request_id, raw_response_from_vllm_output(final)

        def fill_pending() -> None:
            while len(pending) < max_in_flight:
                try:
                    request_id, prompt = next(iterator)
                except StopIteration:
                    return
                engine_request_id = f"{time.time_ns()}-{request_id}"
                task = asyncio.create_task(infer(request_id, prompt, engine_request_id))
                pending[task] = engine_request_id

        try:
            fill_pending()
            while pending:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    pending.pop(task)
                    yield task.result()
                fill_pending()
        except BaseException:
            for task, engine_request_id in pending.items():
                # vLLM 0.28 exposes ``abort`` as a coroutine.  Older
                # releases returned synchronously, so retain compatibility
                # without leaking a coroutine during interruption cleanup.
                with contextlib.suppress(Exception):
                    abort_result = self.engine.abort(engine_request_id)
                    if inspect.isawaitable(abort_result):
                        await abort_result
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise

    def generate(
        self,
        prompts: Sequence[str],
        request: GenerationRequest | Mapping[str, Any],
    ) -> list[RawModelResponse]:
        """Collect a bounded fallback batch for rare retry operations.

        Main matrix inference calls :meth:`generate_stream`; this compatibility
        method is intentionally reserved for small deferred validation retries.
        """

        async def collect() -> list[RawModelResponse]:
            responses: list[RawModelResponse | None] = [None] * len(prompts)
            items = ((str(index), prompt) for index, prompt in enumerate(prompts))
            async for request_id, response in self.generate_stream(
                items,
                request,
                max_in_flight=max(1, min(len(prompts), 64)),
            ):
                responses[int(request_id)] = response
            if any(response is None for response in responses):
                raise BackendFailure("vLLM asynchronous fallback returned incomplete responses")
            return [cast(RawModelResponse, response) for response in responses]

        return asyncio.run(collect())

    def close(self) -> None:
        """Stop the background engine after all durable parts are flushed."""

        if self._closed:
            return
        self._closed = True
        self.engine.shutdown()
