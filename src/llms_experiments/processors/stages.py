"""Compatibility re-exports for processor stage implementations."""

from __future__ import annotations

from .base import ProcessingStage
from .implementations.candidate_logprobs import CandidateLogprobsStage
from .implementations.fan_out import FanOutStage
from .implementations.identity import IdentityStage
from .implementations.json_decode import JsonDecodeStage
from .implementations.json_schema import JsonSchemaStage
from .implementations.verbalized_confidence import VerbalizedConfidenceStage
from .utils import (
    aggregate_candidate_logprobs,
    check_schema as _check_schema,
    digit_from_token as _digit_from_token,
    digit_logprobs,
    logsumexp as _logsumexp,
    replace_schema_enums as _replace_schema_enums,
)

STAGE_TYPES: dict[str, type[ProcessingStage]] = {
    "identity": IdentityStage,
    "fan_out": FanOutStage,
    "json_decode": JsonDecodeStage,
    "json_schema": JsonSchemaStage,
    "candidate_logprobs": CandidateLogprobsStage,
    "verbalized_confidence": VerbalizedConfidenceStage,
}

__all__ = [
    "STAGE_TYPES",
    "CandidateLogprobsStage",
    "FanOutStage",
    "IdentityStage",
    "JsonDecodeStage",
    "JsonSchemaStage",
    "VerbalizedConfidenceStage",
    "_check_schema",
    "_digit_from_token",
    "_logsumexp",
    "_replace_schema_enums",
    "aggregate_candidate_logprobs",
    "digit_logprobs",
]
