"""Construct processor pipelines from validated YAML mappings."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .base import ProcessingStage, Processor
from .contracts import ProcessorContext
from .implementations.candidate_logprobs import CandidateLogprobsStage
from .implementations.fan_out import FanOutStage
from .implementations.identity import IdentityStage
from .implementations.json_decode import JsonDecodeStage
from .implementations.json_schema import JsonSchemaStage
from .implementations.verbalized_confidence import VerbalizedConfidenceStage

STAGE_TYPES: dict[str, type[ProcessingStage]] = {
    "identity": IdentityStage,
    "fan_out": FanOutStage,
    "json_decode": JsonDecodeStage,
    "json_schema": JsonSchemaStage,
    "candidate_logprobs": CandidateLogprobsStage,
    "verbalized_confidence": VerbalizedConfidenceStage,
}

LEGACY_PROCESSOR_FIELDS = ("request_mode", "result_type", "expand_over")


class ConfigurationDefaultWarning(UserWarning):
    """A visible warning that an extension default was selected."""


class LegacyConfigurationWarning(UserWarning):
    """A visible warning that unhandled configuration fields were ignored."""


def create_stage(specification: Mapping[str, Any]) -> ProcessingStage:
    """Create one stage from its explicit type mapping."""
    if not isinstance(specification, Mapping):
        raise ValueError("every processor stage must be a mapping")
    stage_type = str(specification.get("type", ""))
    if stage_type not in STAGE_TYPES:
        supported = ", ".join(sorted(STAGE_TYPES))
        raise ValueError(f"unsupported processor stage {stage_type!r}; expected one of: {supported}")
    return STAGE_TYPES[stage_type]({key: value for key, value in specification.items() if key != "type"})


def create_processor(
    variant: Mapping[str, Any],
    *,
    root: str | Path,
    dataset_labels: list[str] | tuple[str, ...] = (),
    code_labels: Mapping[str, str] | None = None,
) -> Processor:
    """Build and validate one YAML-defined processor pipeline."""
    for name in LEGACY_PROCESSOR_FIELDS:
        if name in variant:
            warnings.warn(
                f"{variant.get('id', '<variant>')}: {name} is unhandled; processing is controlled by processor.stages",
                LegacyConfigurationWarning,
                stacklevel=2,
            )
    configured = variant.get("processor")
    if configured is None:
        warnings.warn(
            f"{variant.get('id', '<variant>')}: processor omitted; using identity raw-output processing",
            ConfigurationDefaultWarning,
            stacklevel=2,
        )
        configured = {"result": "raw", "stages": [{"type": "identity"}]}
    if not isinstance(configured, Mapping):
        raise ValueError("processor must be a mapping with result and stages")
    result_type = str(configured.get("result", ""))
    specifications = configured.get("stages")
    if not isinstance(specifications, list) or not specifications:
        raise ValueError("processor.stages must be a non-empty list")
    context = ProcessorContext(
        variant_id=str(variant.get("id", "")),
        result_type=result_type,
        root=Path(root),
        dataset_labels=tuple(str(item) for item in dataset_labels),
        code_labels=dict(code_labels or {}),
    )
    return Processor(
        result_type=result_type,
        stages=[create_stage(item) for item in specifications],
        context=context,
        max_tokens=int(variant.get("max_tokens", 128)),
    )


def processor_config_hash_material(variant: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return behavior-defining processor configuration fields for config hashing."""
    return {key: value for key, value in variant.items() if key not in LEGACY_PROCESSOR_FIELDS}
