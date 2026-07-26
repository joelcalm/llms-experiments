"""JSON decode processing stage implementation."""

from __future__ import annotations

import json
from dataclasses import replace

from ..base import ProcessingStage
from ..contracts import ProcessingError, ProcessingState, ProcessorContext


class JsonDecodeStage(ProcessingStage):
    """Parses raw text response into a structured JSON object."""

    type_name = "json_decode"
    required_fields = frozenset({"raw_text"})
    produced_fields = frozenset({"value"})

    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        """Decode response text into JSON value or set parsing failure error."""
        try:
            return replace(state, value=json.loads(state.response.text))
        except json.JSONDecodeError as exc:
            return state.fail(
                ProcessingError(
                    code="json_parse_error",
                    message=str(exc),
                    stage=self.type_name,
                    category="parsing",
                )
            )
