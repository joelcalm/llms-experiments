"""Paired TSV input reader implementation."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from typing import Any

from ..base import InputReader


class PairedTsvInputReader(InputReader):
    """Read aligned argument and label TSV files keyed by the same row id."""

    format_name = "paired_tsv"

    def validate(self) -> None:
        """Require the extra labels file used by paired TSV sources."""
        super().validate()
        if not self.source.get("labels_path"):
            raise ValueError("input.labels_path is required for paired_tsv")

    def _pairs(self) -> list[tuple[Any, Any]]:
        """Collect the main pair and any additional argument/label file pairs."""
        pairs = [(self.path, self.resolve(str(self.source["labels_path"])))]
        pairs.extend(
            (self.resolve(str(pair["path"])), self.resolve(str(pair["labels_path"])))
            for pair in self.source.get("additional_pairs", [])
        )
        return pairs

    def provenance_paths(self) -> list[Any]:
        """Return every argument and label file that contributes to provenance."""
        return [path for pair in self._pairs() for path in pair]

    def iter_rows(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        """Join argument rows with their labels and emit normalized examples."""
        effective_limit = self.effective_limit(limit)
        id_column = str(self.source["id_column"])
        text_column = str(self.source["text_column"])
        selected_columns = self.source.get("label_columns")
        emitted = 0
        position = 0
        for argument_path, label_path in self._pairs():
            with argument_path.open(encoding="utf-8", newline="") as handle:
                arguments = {row[id_column]: dict(row) for row in csv.DictReader(handle, delimiter="\t")}
            with label_path.open(encoding="utf-8", newline="") as handle:
                labels = {row[id_column]: dict(row) for row in csv.DictReader(handle, delimiter="\t")}
            for row_id, argument in arguments.items():
                if row_id not in labels:
                    continue
                label_row = labels[row_id]
                columns = selected_columns or [key for key in label_row if key != id_column]
                row = {
                    id_column: row_id,
                    text_column: argument.get(text_column, ""),
                    "_gold_labels": [
                        key for key in columns if str(label_row.get(key, "0")).strip() in {"1", "1.0", "true", "True"}
                    ],
                }
                yield self.normalize(row, position)
                emitted += 1
                position += 1
                if effective_limit is not None and emitted >= effective_limit:
                    return
