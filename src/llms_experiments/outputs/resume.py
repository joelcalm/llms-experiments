"""Format-independent SQLite resume index."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from .base import OutputStore


class ResumeIndex:
    def __init__(self, path: Path, fingerprint: str | None = None, store: OutputStore | None = None) -> None:
        self.store = store
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS completed (variant_id TEXT, input_row_id TEXT, source_position INTEGER, PRIMARY KEY (variant_id, input_row_id, source_position))"
        )
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        self.cleared = False
        self.retryable_keys: set[tuple[str, str, int]] = set()
        if fingerprint is not None:
            previous = self.connection.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
            if previous and previous[0] != fingerprint:
                self.connection.execute("DELETE FROM completed")
                self.cleared = True
            self.connection.execute("INSERT OR REPLACE INTO metadata VALUES ('fingerprint', ?)", (fingerprint,))
        self.connection.commit()

    def add(self, key: tuple[str, str, int]) -> None:
        self.connection.execute("INSERT OR IGNORE INTO completed VALUES (?, ?, ?)", key)

    def contains(self, key: tuple[str, str, int]) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM completed WHERE variant_id=? AND input_row_id=? AND source_position=? LIMIT 1",
                key,
            ).fetchone()
            is not None
        )

    def seed_from(
        self,
        paths: Sequence[Path],
        expected_hashes: Mapping[str, str] | None = None,
        store: OutputStore | None = None,
    ) -> int:
        active_store = store or self.store
        if active_store is None:
            from .factory import create_output_store_for_path

            active_store = create_output_store_for_path(paths[0]) if paths else None
        if active_store is None:
            return 0
        count = 0
        for row in active_store.iter_rows(paths):
            variant = str(row["variant_id"])
            if expected_hashes and row.get("config_hash") != expected_hashes.get(variant):
                continue
            key = (variant, str(row["input_row_id"]), int(row.get("source_position", -1)))
            if row.get("final_status") == "failed_backend":
                self.retryable_keys.add(key)
                continue
            self.add(key)
            count += 1
        self.connection.commit()
        return count

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()
