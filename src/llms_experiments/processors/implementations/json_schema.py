"""JSON Schema validation stage implementation."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..base import ProcessingStage
from ..contracts import ProcessingError, ProcessingState, ProcessorContext, ResponseRequirements
from ..utils import check_schema, replace_schema_enums


class JsonSchemaStage(ProcessingStage):
    """Validates decoded JSON value against declared JSON Schema specification."""

    type_name = "json_schema"
    required_fields = frozenset({"value"})
    produced_fields = frozenset({"value"})

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        """Initialize stage and verify schema configuration presence."""
        super().__init__(config)
        if not self.config.get("schema"):
            raise ValueError("json_schema.schema is required")
        self._schema: dict[str, Any] | None = None

    def _resolved_schema(self, context: ProcessorContext) -> dict[str, Any]:
        """Load and resolve schema file content with dataset label enums if configured."""
        if self._schema is None:
            self._schema = json.loads(context.resolve(str(self.config["schema"])).read_text(encoding="utf-8"))
            if self.config.get("enum_from") == "dataset_labels":
                replace_schema_enums(self._schema, context.dataset_labels)
            elif self.config.get("enum_from") is not None:
                raise ValueError("json_schema.enum_from must be 'dataset_labels'")
        return self._schema

    def requirements(self, context: ProcessorContext) -> ResponseRequirements:
        """Return response requirements containing structured output schema."""
        return ResponseRequirements(structured_schema=self._resolved_schema(context), max_tokens=1)

    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        """Validate state value against schema and set error if invalid."""
        errors: list[str] = []
        check_schema(state.value, self._resolved_schema(context), "$", errors)
        if not errors:
            return state
        return state.fail(
            ProcessingError(
                code="schema_validation_error",
                message="; ".join(errors),
                stage=self.type_name,
                category="validation",
            )
        )
