"""Format-independent SQLite resume index."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from .base import OutputStore

_SEED_BATCH_SIZE = 4096


class ResumeIndex:
    def __init__(self, path: Path, fingerprint: str | None = None, store: OutputStore | None = None) -> None:
        self.store = store
        path.parent.mkdir(parents=True, exist_ok=True)
        self.recovered_from_corruption = False
        self.connection = self._connect(path)
        self.cleared = False
        self.retryable_keys: set[tuple[str, str, int]] = set()
        if fingerprint is not None:
            previous = self.connection.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
            if previous and previous[0] != fingerprint:
                self.connection.execute("DELETE FROM completed")
                self.cleared = True
            self.connection.execute("INSERT OR REPLACE INTO metadata VALUES ('fingerprint', ?)", (fingerprint,))
        self.connection.commit()

    def _connect(self, path: Path) -> sqlite3.Connection:
        """Open the index, preserving durable parts if SQLite reports corruption."""

        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS completed (variant_id TEXT, input_row_id TEXT, source_position INTEGER, PRIMARY KEY (variant_id, input_row_id, source_position))"
            )
            connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
            return connection
        except sqlite3.DatabaseError as exc:
            message = str(exc).lower()
            if "malformed" not in message and "not a database" not in message:
                raise
            if connection is not None:
                connection.close()
            backup = path.with_name(f"{path.name}.corrupt-{time.time_ns()}")
            path.replace(backup)
            self.recovered_from_corruption = True
            fresh = sqlite3.connect(path)
            fresh.execute(
                "CREATE TABLE completed (variant_id TEXT, input_row_id TEXT, source_position INTEGER, PRIMARY KEY (variant_id, input_row_id, source_position))"
            )
            fresh.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            return fresh

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
        retry_failed_validation: bool = False,
    ) -> int:
        """Seed completed keys from durable output parts.

        Backend failures are always left retryable.  Validation failures keep
        the historical durable-by-default behaviour, but a campaign may opt
        into retrying them when a corrected prompt or token budget is known to
        be available.

        Parameters
        ----------
        paths:
            Existing output files or immutable parts to inspect.
        expected_hashes:
            Optional variant-to-configuration-hash mapping used to ignore
            stale rows.
        store:
            Output store used to iterate the files.
        retry_failed_validation:
            Leave ``failed_validation`` keys out of the completed index so a
            subsequent run can generate replacement rows.
        """
        active_store = store or self.store
        if active_store is None:
            from .factory import create_output_store_for_path

            active_store = create_output_store_for_path(paths[0]) if paths else None
        if active_store is None:
            return 0
        count = 0
        pending_keys: list[tuple[str, str, int]] = []
        for row in active_store.iter_rows(paths):
            variant = str(row["variant_id"])
            if expected_hashes and row.get("config_hash") != expected_hashes.get(variant):
                continue
            key = (variant, str(row["input_row_id"]), int(row.get("source_position", -1)))
            status = row.get("final_status")
            if status == "failed_backend" or (retry_failed_validation and status == "failed_validation"):
                self.connection.execute(
                    "DELETE FROM completed WHERE variant_id=? AND input_row_id=? AND source_position=?",
                    key,
                )
                self.retryable_keys.add(key)
                continue
            pending_keys.append(key)
            if len(pending_keys) >= _SEED_BATCH_SIZE:
                self.connection.executemany("INSERT OR IGNORE INTO completed VALUES (?, ?, ?)", pending_keys)
                pending_keys.clear()
            count += 1
        if pending_keys:
            self.connection.executemany("INSERT OR IGNORE INTO completed VALUES (?, ?, ?)", pending_keys)
        self.connection.commit()
        return count

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()
