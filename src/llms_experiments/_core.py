"""Deprecated v0.2 compatibility facade.

Runtime code is organized by responsibility in the sibling packages. Existing
imports from ``llms_experiments._core`` continue to resolve for one migration
cycle, but new code should import from the public modules.
"""

from __future__ import annotations

from typing import Any

from .backends import (
    Backend,
    BackendFailure,
    FakeBackend,
    LlamaCppBackend,
    OpenAICompatibleBackend,
    VLLMBackend,
    create_backend,
)
from .batching import (
    batch_command,
    batch_command_args,
    parse_batch,
    prepare,
    prepare_matrix,
)
from .benchmarking import benchmark
from .compatibility import (
    Response,
    aggregate_candidate_logprobs,
    digit_logprobs,
    expanded_rows,
    extract_position_logprobs,
    extract_top_logprobs,
    flatten_position_logprobs,
    interpret_response,
    top_logprobs_count,
    verbalized_confidence,
)
from .configuration import (
    config_overrides,
    dataset_config,
    dataset_entries,
    dataset_runtime,
    load_config,
    resolve,
    select_dataset,
    selected_entries,
    validate_config,
)
from .events import ErrorLog, Events, make_events
from .inputs import (
    create_input_reader,
    iter_rows_for_source,
    normalize_gold_labels,
    read_rows,
    rows_for_source,
    source_provenance,
    split_labels,
)
from .orchestration import (
    generate_with_backoff,
    generation_request,
    run,
    run_deferred_retries,
    run_matrix,
    tune,
)
from .outputs import (
    RESULT_SCHEMA,
    PartWriter,
    ResumeIndex,
    create_output_store,
    merge_parts,
)
from .processors import (
    GenerationRequest,
    ProcessedResult,
    ProcessingError,
    ProcessingStage,
    Processor,
    RawModelResponse,
    ResponseRequirements,
    create_processor,
    create_stage,
)
from .prompting import (
    _read_asset,
    build_requests,
    conversation,
    materialize_variant,
    prompt,
    prompt_group_id,
    prompt_part_values,
    prompt_values,
    read_asset,
    render,
    rendered_prompt,
    request_for_row,
    retry_values,
    system_prompt,
    system_prompt_paths,
    variant_config_hash,
    variant_schema,
)
from .results import (
    BACKEND_ERROR_PREFIX,
    CONTRACT_VERSION,
    TOOL_VERSION,
    check_schema,
    error_kind,
    failure_status,
    result_row,
    semantic_result_type,
    serialise,
    validate_response,
)
from .runtime import (
    apply_resource_guard,
    available_cpu_ids,
    configure_torch_cpu_threads,
    configure_vllm_environment,
    cpu_resource_plan,
    gpu,
    gpu_preflight,
    sync_cuda,
)

_split_labels = split_labels
make_backend = create_backend
NvidiaAPIBackend = OpenAICompatibleBackend


def variant_expansion_labels(config: dict[str, Any], variant: dict[str, Any]) -> list[str] | None:
    processor = materialize_variant(config, variant)["_processor"]
    if any(stage.type_name == "fan_out" for stage in processor.stages):
        return list(config.get("run", {}).get("dataset_labels", []))
    return None


__all__ = [
    "BACKEND_ERROR_PREFIX",
    "CONTRACT_VERSION",
    "RESULT_SCHEMA",
    "TOOL_VERSION",
    "Backend",
    "BackendFailure",
    "ErrorLog",
    "Events",
    "FakeBackend",
    "GenerationRequest",
    "LlamaCppBackend",
    "NvidiaAPIBackend",
    "OpenAICompatibleBackend",
    "PartWriter",
    "ProcessedResult",
    "ProcessingError",
    "ProcessingStage",
    "Processor",
    "RawModelResponse",
    "Response",
    "ResponseRequirements",
    "ResumeIndex",
    "VLLMBackend",
    "_read_asset",
    "_split_labels",
    "aggregate_candidate_logprobs",
    "apply_resource_guard",
    "available_cpu_ids",
    "batch_command",
    "batch_command_args",
    "benchmark",
    "build_requests",
    "check_schema",
    "config_overrides",
    "configure_torch_cpu_threads",
    "configure_vllm_environment",
    "conversation",
    "cpu_resource_plan",
    "create_backend",
    "create_input_reader",
    "create_output_store",
    "create_processor",
    "create_stage",
    "dataset_config",
    "dataset_entries",
    "dataset_runtime",
    "digit_logprobs",
    "error_kind",
    "expanded_rows",
    "extract_position_logprobs",
    "extract_top_logprobs",
    "failure_status",
    "flatten_position_logprobs",
    "generate_with_backoff",
    "generation_request",
    "gpu",
    "gpu_preflight",
    "interpret_response",
    "iter_rows_for_source",
    "load_config",
    "make_backend",
    "make_events",
    "materialize_variant",
    "merge_parts",
    "normalize_gold_labels",
    "parse_batch",
    "prepare",
    "prepare_matrix",
    "prompt",
    "prompt_group_id",
    "prompt_part_values",
    "prompt_values",
    "read_asset",
    "read_rows",
    "render",
    "rendered_prompt",
    "request_for_row",
    "resolve",
    "result_row",
    "retry_values",
    "rows_for_source",
    "run",
    "run_deferred_retries",
    "run_matrix",
    "select_dataset",
    "selected_entries",
    "semantic_result_type",
    "serialise",
    "source_provenance",
    "split_labels",
    "sync_cuda",
    "system_prompt",
    "system_prompt_paths",
    "top_logprobs_count",
    "tune",
    "validate_config",
    "validate_response",
    "variant_config_hash",
    "variant_expansion_labels",
    "variant_schema",
    "verbalized_confidence",
]
