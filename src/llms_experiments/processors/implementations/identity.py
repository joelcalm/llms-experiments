"""Identity processing stage implementation."""

from __future__ import annotations

from dataclasses import replace

from ..base import ProcessingStage
from ..contracts import ProcessingState, ProcessorContext, ResponseRequirements


class IdentityStage(ProcessingStage):
    """Pass-through stage that extracts raw output response text directly into state value."""

    type_name = "identity"
    required_fields = frozenset({"raw_text"})
    produced_fields = frozenset({"value"})

    def requirements(self, context: ProcessorContext) -> ResponseRequirements:
        """Return response requirements specifying configured max token length."""
        return ResponseRequirements(max_tokens=int(self.config.get("max_tokens", 128)))

    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        """Assign raw response text directly as the value in processing state."""
        return replace(state, value=state.response.text)
