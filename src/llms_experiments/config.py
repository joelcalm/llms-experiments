"""Typed configuration models, loading, validation, and selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import configuration


class ConfigSection(BaseModel):
    """A validated section that preserves backend-specific extension keys."""

    model_config = ConfigDict(extra="allow")


class RunConfig(ConfigSection):
    id: str = Field(min_length=1)


class InputConfig(ConfigSection):
    path: str
    format: str
    id_column: str
    text_column: str


class DatasetConfig(ConfigSection):
    id: str = Field(min_length=1)
    input: InputConfig


class ModelConfig(ConfigSection):
    name: str = Field(min_length=1)
    backend: Literal["local_vllm", "openai_compatible", "fake", "llama_cpp"]


class ProcessingStageConfig(ConfigSection):
    type: Literal[
        "identity",
        "fan_out",
        "json_decode",
        "json_schema",
        "candidate_logprobs",
        "verbalized_confidence",
    ]


class ProcessorConfig(ConfigSection):
    result: str = Field(min_length=1)
    stages: list[ProcessingStageConfig] = Field(min_length=1)


class VariantConfig(ConfigSection):
    id: str = Field(min_length=1)
    prompts: list[str] = Field(min_length=1)
    processor: ProcessorConfig | None = None


class OutputConfig(ConfigSection):
    directory: str
    format: Literal["parquet", "csv", "jsonl"] | None = None


class ExperimentConfig(ConfigSection):
    """Top-level run contract for one input or a dataset matrix."""

    run: RunConfig
    model: ModelConfig
    variants: list[VariantConfig] = Field(min_length=1)
    output: OutputConfig
    input: InputConfig | None = None
    datasets: list[DatasetConfig] | None = None

    @model_validator(mode="after")
    def one_input_shape(self) -> ExperimentConfig:
        if (self.input is None) == (self.datasets is None):
            raise ValueError("configuration requires exactly one of input or datasets")
        identifiers = [variant.id for variant in self.variants]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("variant IDs must be unique")
        if self.datasets is not None:
            datasets = [dataset.id for dataset in self.datasets]
            if len(datasets) != len(set(datasets)):
                raise ValueError("dataset IDs must be unique")
        return self


def load_config(path: str | Path, overrides: list[str] | None = None, *, check_files: bool = True) -> dict[str, Any]:
    """Load YAML configuration and enforce the typed ExperimentConfig schema."""

    config = configuration.load_config(path, overrides, check_files=check_files)
    ExperimentConfig.model_validate(config)
    return config


def configuration_schema() -> dict[str, Any]:
    """Return the machine-readable JSON Schema for experiment configurations."""

    return ExperimentConfig.model_json_schema()


config_overrides = configuration.config_overrides
dataset_config = configuration.dataset_config
select_dataset = configuration.select_dataset
validate_config = configuration.validate_config

__all__ = [
    "DatasetConfig",
    "ExperimentConfig",
    "InputConfig",
    "ModelConfig",
    "OutputConfig",
    "ProcessingStageConfig",
    "ProcessorConfig",
    "RunConfig",
    "VariantConfig",
    "config_overrides",
    "configuration_schema",
    "dataset_config",
    "load_config",
    "select_dataset",
    "validate_config",
]
