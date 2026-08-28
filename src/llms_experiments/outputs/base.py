"""Durable output-store abstraction and format-independent lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .schema import normalize_result_row

_FINAL_WRITE_BATCH_ROWS = 4096


class ResultFileWriter(ABC):
    @abstractmethod
    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Write another bounded batch."""

    @abstractmethod
    def close(self) -> None:
        """Close the underlying file."""


class OutputStore(ABC):
    """Abstract durable store for append-only parts and final projections."""

    format_name: str
    extension: str

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def result_path(self) -> Path:
        return self.run_dir / f"results.{self.extension}"

    def variant_path(self, variant_id: str) -> Path:
        return self.run_dir / f"{variant_id}.{self.extension}"

    def part_directory(self, variant_id: str) -> Path:
        return self.run_dir / "parts" / f"variant={variant_id}"

    def part_paths(self) -> list[Path]:
        root = self.run_dir / "parts"
        return sorted(root.glob(f"variant=*/part-*.{self.extension}")) if root.exists() else []

    @abstractmethod
    def open_writer(self, path: Path) -> ResultFileWriter:
        """Open a bounded-batch writer for one temporary file."""

    @abstractmethod
    def iter_file(self, path: Path) -> Iterator[dict[str, Any]]:
        """Yield logical rows from one store file."""

    def iter_rows(self, paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
        for path in paths:
            if path.exists():
                yield from self.iter_file(path)

    def write_atomic(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        writer = self.open_writer(temporary)
        try:
            writer.write(rows)
        finally:
            writer.close()
        temporary.replace(path)

    def part_writer(self, variant_id: str, target_rows: int = 4096) -> AtomicPartWriter:
        return AtomicPartWriter(self, variant_id, target_rows)

    def read_final(self) -> list[dict[str, Any]]:
        path = self.result_path()
        return list(self.iter_file(path)) if path.exists() else []

    def _latest_retries(
        self,
        paths: Sequence[Path],
        keys: set[tuple[str, str, int]],
    ) -> dict[tuple[str, str, int], tuple[int, int]]:
        latest: dict[tuple[str, str, int], tuple[int, int]] = {}
        for file_index, path in enumerate(paths):
            for row_index, row in enumerate(self.iter_file(path)):
                key = (str(row["variant_id"]), str(row["input_row_id"]), int(row["source_position"]))
                if key in keys:
                    latest[key] = (file_index, row_index)
        return latest

    def _publish(
        self,
        rows: Iterable[tuple[int, int, dict[str, Any]]],
        *,
        expected_hashes: Mapping[str, str] | None = None,
        latest_retries: Mapping[tuple[str, str, int], tuple[int, int]] | None = None,
        publish_combined: bool = True,
    ) -> int:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        result_temporary = self.result_path().with_suffix(f".{self.extension}.tmp")
        result_writer = self.open_writer(result_temporary) if publish_combined else None
        variant_writers: dict[str, ResultFileWriter] = {}
        variant_temporaries: dict[str, Path] = {}
        result_rows: list[dict[str, Any]] = []
        variant_rows: dict[str, list[dict[str, Any]]] = {}
        count = 0

        def flush_result_rows() -> None:
            if result_writer is not None and result_rows:
                result_writer.write(result_rows)
                result_rows.clear()

        def flush_variant_rows(variant: str) -> None:
            rows = variant_rows[variant]
            if rows:
                variant_writers[variant].write(rows)
                rows.clear()

        try:
            for file_index, row_index, row in rows:
                normalized = normalize_result_row(row)
                variant = str(normalized["variant_id"])
                if expected_hashes and normalized.get("config_hash") != expected_hashes.get(variant):
                    continue
                key = (variant, str(normalized["input_row_id"]), int(normalized["source_position"]))
                if latest_retries and latest_retries.get(key, (file_index, row_index)) != (file_index, row_index):
                    continue
                if variant not in variant_writers:
                    temporary = self.variant_path(variant).with_suffix(f".{self.extension}.tmp")
                    variant_temporaries[variant] = temporary
                    variant_writers[variant] = self.open_writer(temporary)
                    variant_rows[variant] = []
                result_rows.append(normalized)
                variant_rows[variant].append(normalized)
                if result_writer is not None and len(result_rows) >= _FINAL_WRITE_BATCH_ROWS:
                    flush_result_rows()
                if len(variant_rows[variant]) >= _FINAL_WRITE_BATCH_ROWS:
                    flush_variant_rows(variant)
                count += 1
        finally:
            flush_result_rows()
            for variant in variant_writers:
                flush_variant_rows(variant)
            if result_writer is not None:
                result_writer.close()
            for writer in variant_writers.values():
                writer.close()
        if count:
            if publish_combined:
                result_temporary.replace(self.result_path())
            for variant, temporary in variant_temporaries.items():
                temporary.replace(self.variant_path(variant))
        elif publish_combined and result_temporary.exists():
            result_temporary.unlink()
        return count

    def finalize(
        self,
        expected_hashes: Mapping[str, str] | None = None,
        retried_keys: set[tuple[str, str, int]] | None = None,
        *,
        publish_combined: bool = True,
    ) -> int:
        paths = self.part_paths()
        if not paths:
            return 0
        latest = self._latest_retries(paths, retried_keys) if retried_keys else {}

        def indexed_rows() -> Iterator[tuple[int, int, dict[str, Any]]]:
            for file_index, path in enumerate(paths):
                for row_index, row in enumerate(self.iter_file(path)):
                    yield file_index, row_index, row

        return self._publish(
            indexed_rows(),
            expected_hashes=expected_hashes,
            latest_retries=latest,
            publish_combined=publish_combined,
        )

    def write_snapshot(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return self._publish((0, index, dict(row)) for index, row in enumerate(rows))

    def discard(self, variant_ids: Iterable[str]) -> None:
        import shutil

        parts = self.run_dir / "parts"
        if parts.exists():
            shutil.rmtree(parts)
        for path in [self.result_path(), *(self.variant_path(identifier) for identifier in variant_ids)]:
            if path.exists():
                path.unlink()

    def discard_variants(self, variant_ids: Iterable[str]) -> None:
        """Discard only the selected variant parts and projections.

        This is used by concurrently isolated strategy jobs when a changed
        configuration invalidates one variant's resume index. Other strategy
        parts in the same matrix directory remain recoverable.
        """

        import shutil

        for variant_id in variant_ids:
            parts = self.part_directory(str(variant_id))
            if parts.exists():
                shutil.rmtree(parts)
            path = self.variant_path(str(variant_id))
            if path.exists():
                path.unlink()


class AtomicPartWriter:
    """Buffer rows and atomically publish immutable result parts."""

    def __init__(self, store: OutputStore, variant_id: str, target_rows: int = 4096) -> None:
        self.store = store
        self.variant_id = variant_id
        self.directory = store.part_directory(variant_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.target_rows = max(1, target_rows)
        self.rows: list[dict[str, Any]] = []
        self.index = len(list(self.directory.glob(f"part-*.{store.extension}")))

    def append(self, row: Mapping[str, Any]) -> bool:
        self.rows.append(normalize_result_row(dict(row)))
        return self.flush() if len(self.rows) >= self.target_rows else False

    def flush(self) -> bool:
        if not self.rows:
            return False
        path = self.directory / f"part-{self.index:05d}.{self.store.extension}"
        self.store.write_atomic(path, self.rows)
        self.index += 1
        self.rows.clear()
        return True

    def close(self) -> None:
        self.flush()
