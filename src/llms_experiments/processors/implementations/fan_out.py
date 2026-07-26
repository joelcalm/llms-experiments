"""Fan-out row preparation stage implementation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from ..base import ProcessingStage
from ..contracts import ProcessingState, ProcessorContext


class FanOutStage(ProcessingStage):
    """Row-preparation stage multiplying input rows across configured dataset labels."""

    type_name = "fan_out"
    prepares_rows = True

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        """Initialize fan_out configuration and validate target domain."""
        super().__init__(config)
        if self.config.get("over") != "dataset_labels":
            raise ValueError("fan_out.over must be 'dataset_labels'")

    def prepare_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        context: ProcessorContext,
    ) -> Iterator[Mapping[str, Any]]:
        """Yield multiplied row dicts for each dataset label."""
        labels = context.dataset_labels
        if not labels:
            raise ValueError("fan_out over dataset_labels requires a non-empty dataset label set")
        width = len(labels)
        for row in rows:
            base = int(row["_source_position"])
            for index, label in enumerate(labels):
                yield {
                    **row,
                    "_target_label": label,
                    "_source_position": base * width + index,
                }

    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        """Return state unchanged as fan_out operates during row preparation."""
        return state
