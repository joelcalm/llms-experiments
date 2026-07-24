"""Built-in input-reader implementations."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq

from .base import InputReader, split_labels


class DelimitedInputReader(InputReader):
    delimiter = ","

    def iter_rows(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        effective_limit = self.effective_limit(limit)
        delimiter = str(self.source.get("delimiter", self.delimiter))
        emitted = 0
        position = 0
        where = dict(self.source.get("where", {}))
        with self.path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle, delimiter=delimiter):
                if where and any(str(raw.get(key)) != str(value) for key, value in where.items()):
                    continue
                yield self.normalize(raw, position)
                emitted += 1
                position += 1
                if effective_limit is not None and emitted >= effective_limit:
                    break


class CsvInputReader(DelimitedInputReader):
    format_name = "csv"
    delimiter = ","


class TsvInputReader(DelimitedInputReader):
    format_name = "tsv"
    delimiter = "\t"


class JsonLinesInputReader(InputReader):
    format_name = "jsonl"

    def iter_rows(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        effective_limit = self.effective_limit(limit)
        where = dict(self.source.get("where", {}))
        emitted = 0
        position = 0
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if where and any(str(row.get(key)) != str(value) for key, value in where.items()):
                    continue
                yield self.normalize(row, position)
                emitted += 1
                position += 1
                if effective_limit is not None and emitted >= effective_limit:
                    break


class ParquetInputReader(InputReader):
    format_name = "parquet"

    def iter_rows(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        effective_limit = self.effective_limit(limit)
        where = dict(self.source.get("where", {}))
        emitted = 0
        position = 0
        parquet = pq.ParquetFile(self.path)
        for batch in parquet.iter_batches():
            for row in batch.to_pylist():
                if where and any(str(row.get(key)) != str(value) for key, value in where.items()):
                    continue
                yield self.normalize(row, position)
                emitted += 1
                position += 1
                if effective_limit is not None and emitted >= effective_limit:
                    return


class NestedJsonInputReader(InputReader):
    format_name = "nested_json"

    def iter_rows(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        effective_limit = self.effective_limit(limit)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        records_key = str(self.source.get("records_key", "Tweets"))
        labels_key = str(self.source.get("labels_column", "annotations"))
        label_value_key = str(self.source.get("label_value_key", "annotation"))
        where = dict(self.source.get("where", {}))
        emitted = 0
        position = 0
        for parent in payload:
            for record in parent.get(records_key, []):
                if where and any(str(record.get(key)) != str(value) for key, value in where.items()):
                    continue
                labels: list[str] = []
                for annotation in record.get(labels_key, []):
                    value = annotation.get(label_value_key) if isinstance(annotation, dict) else annotation
                    labels.extend(split_labels(value))
                yield self.normalize({**record, "_gold_labels": sorted(set(labels))}, position)
                emitted += 1
                position += 1
                if effective_limit is not None and emitted >= effective_limit:
                    return


class PairedTsvInputReader(InputReader):
    format_name = "paired_tsv"

    def validate(self) -> None:
        super().validate()
        if not self.source.get("labels_path"):
            raise ValueError("input.labels_path is required for paired_tsv")

    def _pairs(self) -> list[tuple[Any, Any]]:
        pairs = [(self.path, self.resolve(str(self.source["labels_path"])))]
        pairs.extend(
            (self.resolve(str(pair["path"])), self.resolve(str(pair["labels_path"])))
            for pair in self.source.get("additional_pairs", [])
        )
        return pairs

    def provenance_paths(self) -> list[Any]:
        return [path for pair in self._pairs() for path in pair]

    def iter_rows(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
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
