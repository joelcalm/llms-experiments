"""Composable YAML-configured response processing pipelines."""

from .base import ProcessingStage, Processor
from .contracts import (
    GenerationRequest,
    ProcessedResult,
    ProcessingError,
    ProcessingState,
    ProcessorContext,
    RawModelResponse,
    ResponseRequirements,
    TokenCandidate,
    TokenPosition,
)
from .factory import (
    ConfigurationDefaultWarning,
    LegacyConfigurationWarning,
    create_processor,
    create_stage,
)

__all__ = [
    "ConfigurationDefaultWarning",
    "GenerationRequest",
    "LegacyConfigurationWarning",
    "ProcessedResult",
    "ProcessingError",
    "ProcessingStage",
    "ProcessingState",
    "Processor",
    "ProcessorContext",
    "RawModelResponse",
    "ResponseRequirements",
    "TokenCandidate",
    "TokenPosition",
    "create_processor",
    "create_stage",
]
