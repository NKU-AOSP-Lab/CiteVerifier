from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class RuntimeStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._initialized = False
        self._ensure_initialized()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            conn = self._connect()
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_counters (
                        name TEXT PRIMARY KEY,
                        value INTEGER NOT NULL DEFAULT 0,
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS single_search_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_title TEXT NOT NULL,
                        query_hash TEXT,
                        found INTEGER NOT NULL DEFAULT 0,
                        max_candidates INTEGER NOT NULL,
                        duration_ms INTEGER,
                        error_message TEXT,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS batch_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_input INTEGER NOT NULL,
                        total_processed INTEGER NOT NULL DEFAULT 0,
                        found_count INTEGER NOT NULL DEFAULT 0,
                        max_candidates INTEGER NOT NULL,
                        duration_ms INTEGER,
                        status TEXT NOT NULL DEFAULT 'running',
                        error_message TEXT,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS batch_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id INTEGER NOT NULL,
                        item_index INTEGER NOT NULL,
                        query_title TEXT NOT NULL,
                        found INTEGER NOT NULL DEFAULT 0,
                        dblp_id INTEGER,
                        dblp_title TEXT,
                        dblp_title_similarity REAL,
                        year TEXT,
                        venue TEXT,
                        pub_type TEXT,
                        duration_ms INTEGER,
                        error_message TEXT,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        FOREIGN KEY(run_id) REFERENCES batch_runs(id) ON DELETE CASCADE
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS event_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        detail_json TEXT,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """
                )
                conn.commit()
                self._initialized = True
            finally:
                conn.close()

    def increment_counter(self, name: str, delta: int = 1) -> int:
        self._ensure_initialized()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO runtime_counters (name, value, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(name) DO UPDATE SET
                    value = runtime_counters.value + excluded.value,
                    updated_at = datetime('now')
                """,
                (name, delta),
            )
            row = conn.execute(
                "SELECT value FROM runtime_counters WHERE name = ?",
                (name,),
            ).fetchone()
            conn.commit()
            return int(row["value"]) if row else 0
        finally:
            conn.close()

    def start_batch_run(self, total_input: int, max_candidates: int) -> int:
        self._ensure_initialized()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO batch_runs (total_input, max_candidates, status, created_at, updated_at)
                VALUES (?, ?, 'running', datetime('now'), datetime('now'))
                """,
                (total_input, max_candidates),
            )
            run_id = int(cur.lastrowid)
            conn.commit()
            return run_id
        finally:
            conn.close()

    def finish_batch_run(
        self,
        run_id: int,
        *,
        total_processed: int,
        found_count: int,
        duration_ms: int,
        status: str,
        error_message: str | None,
    ) -> None:
        self._ensure_initialized()
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE batch_runs
                SET total_processed = ?,
                    found_count = ?,
                    duration_ms = ?,
                    status = ?,
                    error_message = ?,
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (
                    int(total_processed),
                    int(found_count),
                    int(duration_ms),
                    status,
                    error_message,
                    int(run_id),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def record_batch_item(
        self,
        run_id: int,
        *,
        item_index: int,
        query_title: str,
        found: bool,
        dblp_id: int | None,
        dblp_title: str | None,
        dblp_title_similarity: float | None,
        year: Any,
        venue: str | None,
        pub_type: str | None,
        duration_ms: int,
        error_message: str | None,
    ) -> None:
        self._ensure_initialized()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO batch_items (
                    run_id,
                    item_index,
                    query_title,
                    found,
                    dblp_id,
                    dblp_title,
                    dblp_title_similarity,
                    year,
                    venue,
                    pub_type,
                    duration_ms,
                    error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(run_id),
                    int(item_index),
                    query_title,
                    1 if found else 0,
                    dblp_id,
                    dblp_title,
                    dblp_title_similarity,
                    None if year is None else str(year),
                    venue,
                    pub_type,
                    int(duration_ms),
                    error_message,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def record_single_search(
        self,
        *,
        query_title: str,
        query_hash: str,
        found: bool,
        max_candidates: int,
        duration_ms: int,
        error_message: str | None,
    ) -> None:
        self._ensure_initialized()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO single_search_events (
                    query_title,
                    query_hash,
                    found,
                    max_candidates,
                    duration_ms,
                    error_message
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    query_title,
                    query_hash,
                    1 if found else 0,
                    int(max_candidates),
                    int(duration_ms),
                    error_message,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def log_event(self, level: str, message: str, detail: dict[str, Any] | None = None) -> None:
        self._ensure_initialized()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO event_logs (level, message, detail_json)
                VALUES (?, ?, ?)
                """,
                (
                    level.upper(),
                    message,
                    json.dumps(detail or {}, ensure_ascii=False, separators=(",", ":")),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def stats(self) -> dict[str, Any]:
        self._ensure_initialized()
        conn = self._connect()
        try:
            counters = {
                row["name"]: int(row["value"])
                for row in conn.execute("SELECT name, value FROM runtime_counters")
            }
            single_count = conn.execute("SELECT COUNT(1) AS c FROM single_search_events").fetchone()
            batch_count = conn.execute("SELECT COUNT(1) AS c FROM batch_runs").fetchone()
            item_count = conn.execute("SELECT COUNT(1) AS c FROM batch_items").fetchone()
            return {
                "counters": counters,
                "single_search_events": int(single_count["c"]) if single_count else 0,
                "batch_runs": int(batch_count["c"]) if batch_count else 0,
                "batch_items": int(item_count["c"]) if item_count else 0,
            }
        finally:
            conn.close()
