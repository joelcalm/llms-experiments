"""Processor pipeline and processing-stage abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any

from .contracts import (
    ProcessedResult,
    ProcessingError,
    ProcessingState,
    ProcessorContext,
    RawModelResponse,
    ResponseRequirements,
)


class ProcessingStage(ABC):
    """One reusable row-preparation or response-processing operation."""

    type_name: str
    required_fields: frozenset[str] = frozenset()
    produced_fields: frozenset[str] = frozenset()
    prepares_rows: bool = False

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def requirements(self, context: ProcessorContext) -> ResponseRequirements:
        return ResponseRequirements(max_tokens=1)

    def prepare_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        context: ProcessorContext,
    ) -> Iterator[Mapping[str, Any]]:
        yield from rows

    @abstractmethod
    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        """Transform processing state or return it unchanged for preparation-only stages."""


class Processor:
    """Compiled, ordered processing pipeline for one configured variant."""

    def __init__(
        self,
        *,
        result_type: str,
        stages: Sequence[ProcessingStage],
        context: ProcessorContext,
        max_tokens: int = 128,
    ) -> None:
        if not result_type:
            raise ValueError("processor.result is required")
        if not stages:
            raise ValueError("processor.stages must not be empty")
        self.result_type = result_type
        self.stages = tuple(stages)
        self.context = replace(context, result_type=result_type)
        self._validate_chain()
        requirements = ResponseRequirements(max_tokens=max(1, int(max_tokens)))
        for stage in self.stages:
            requirements = requirements.merge(stage.requirements(self.context))
        self.requirements = requirements

    def _validate_chain(self) -> None:
        identity_count = sum(stage.type_name == "identity" for stage in self.stages)
        if identity_count and len(self.stages) != 1:
            raise ValueError("identity must be the only stage in its processor pipeline")
        response_started = False
        available = {"raw_text", "token_positions", "row"}
        for stage in self.stages:
            if stage.prepares_rows:
                if response_started:
                    raise ValueError(f"row-preparation stage {stage.type_name!r} must precede response stages")
                continue
            response_started = True
            missing = stage.required_fields - available
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"processor stage {stage.type_name!r} requires unavailable field(s): {names}")
            available.update(stage.produced_fields)

    def prepare_rows(self, rows: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
        prepared: Iterable[Mapping[str, Any]] = rows
        for stage in self.stages:
            if stage.prepares_rows:
                prepared = stage.prepare_rows(prepared, self.context)
        yield from prepared

    def process(self, response: RawModelResponse, row: Mapping[str, Any]) -> ProcessedResult:
        context = self.context.with_row(row)
        target_label = str(row["_target_label"]) if row.get("_target_label") is not None else None
        if response.backend_error:
            error = ProcessingError(
                code="backend_error",
                message=response.backend_error,
                stage="backend",
                category="backend",
                retryable=True,
            )
            return ProcessedResult(self.result_type, None, None, target_label, (error,))
        state = ProcessingState(response=response)
        for stage in self.stages:
            if stage.prepares_rows:
                continue
            state = stage.process(state, context)
            if state.errors:
                break
        return ProcessedResult(
            result_type=self.result_type,
            value=state.value,
            candidate_scores=state.candidate_scores,
            target_label=target_label,
            errors=state.errors,
            metadata=state.metadata,
        )
