from __future__ import annotations

import csv
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from dblp_builder.pipeline import PipelineConfig, run_pipeline

APP_VERSION = "0.1.0"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).expanduser().resolve()
DEFAULT_DB_PATH = DATA_DIR / "dblp.sqlite"

DB_PATH = Path(os.getenv("DB_PATH", str(DEFAULT_DB_PATH))).expanduser().resolve()
DB_BUSY_TIMEOUT_MS = int(os.getenv("DB_BUSY_TIMEOUT_MS", "30000"))

DEFAULT_XML_GZ_URL = os.getenv("DBLP_XML_GZ_URL", "https://dblp.org/xml/dblp.xml.gz")
DEFAULT_DTD_URL = os.getenv("DBLP_DTD_URL", "https://dblp.org/xml/dblp.dtd")
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
DEFAULT_PROGRESS_EVERY = int(os.getenv("PROGRESS_EVERY", "10000"))
MAX_LOG_LINES = int(os.getenv("MAX_LOG_LINES", "1000"))

MAX_LIMIT = int(os.getenv("MAX_LIMIT", "200"))
MAX_ENTRIES_PER_SIDE = min(int(os.getenv("MAX_ENTRIES_PER_SIDE", "50")), 50)
MAX_AUTHOR_RESOLVE = int(os.getenv("MAX_AUTHOR_RESOLVE", "800"))

FULLMETA_PUBLICATION_COLUMNS = {"id", "title", "year", "venue", "pub_type", "raw_xml"}

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_visit_lock = threading.Lock()
_visit_count = 0


def _next_visit() -> int:
    global _visit_count
    with _visit_lock:
        _visit_count += 1
        return _visit_count

def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:8090").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


app = FastAPI(
    title="DblpService",
    description="DBLP build + query backend (CoAuthors-compatible /api endpoints).",
    version=APP_VERSION,
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

origins = _parse_cors_origins()
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

logger = logging.getLogger("dblp_service")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


@app.get("/", response_class=HTMLResponse)
@app.get("/bootstrap", response_class=HTMLResponse)
def bootstrap_console(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "bootstrap.html",
        {
            "request": request,
            "app_version": APP_VERSION,
            "visit_count": _next_visit(),
            "default_xml_gz_url": DEFAULT_XML_GZ_URL,
            "default_dtd_url": DEFAULT_DTD_URL,
            "default_batch_size": DEFAULT_BATCH_SIZE,
            "default_progress_every": DEFAULT_PROGRESS_EVERY,
            "data_dir": str(DATA_DIR),
            "api_base": "",
        },
    )


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _sanitize_author_entries(entries: list[str]) -> list[str]:
    cleaned = [_normalize(e) for e in entries]
    cleaned = [e for e in cleaned if e]
    seen = set()
    out: list[str] = []
    for e in cleaned:
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def _fts_query_from_text(text: str, max_terms: int = 6) -> str:
    cleaned = "".join((ch.lower() if ch.isalnum() else " ") for ch in text)
    tokens = cleaned.split()
    uniq: list[str] = []
    seen = set()
    for t in tokens:
        if len(t) < 2:
            continue
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
        if len(uniq) >= max_terms:
            break
    return " ".join(uniq)


def _get_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Database file is not available.")
    try:
        conn = sqlite3.connect(
            str(DB_PATH),
            timeout=max(DB_BUSY_TIMEOUT_MS / 1000.0, 1.0),
            check_same_thread=False,
        )
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail=f"Cannot open database: {exc}") from exc
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {DB_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA temp_store = MEMORY;")
    return conn


def _ensure_fullmeta_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
    tables = {row["name"] for row in cur.fetchall()}
    required = {"publications", "authors", "pub_authors"}
    if not required.issubset(tables):
        raise HTTPException(status_code=503, detail="Database schema is incomplete.")

    cur.execute("PRAGMA table_info(publications);")
    columns = {row["name"] for row in cur.fetchall()}
    missing = FULLMETA_PUBLICATION_COLUMNS - columns
    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                "Current database is not fullmeta-compatible. "
                f"Missing columns: {', '.join(sorted(missing))}"
            ),
        )


def _detect_data_date() -> str:
    override = os.getenv("DATA_DATE", "").strip()
    if override:
        return override
    try:
        st = DB_PATH.stat()
    except OSError:
        return "unknown"
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
    ts = datetime.fromtimestamp(mtime_ns / 1_000_000_000, tz=timezone.utc)
    return ts.strftime("%Y-%m-%d")


def _resolve_author_ids(
    conn: sqlite3.Connection,
    name_query: str,
    limit: int | None = None,
    exact_base_match: bool = False,
) -> list[int]:
    normalized = _normalize(name_query)
    if not normalized:
        return []

    cur = conn.cursor()
    cur.execute("SELECT id FROM authors WHERE name = ? LIMIT 1;", (normalized,))
    row = cur.fetchone()
    if row:
        return [int(row["id"])]
    if exact_base_match:
        return []

    lim = MAX_AUTHOR_RESOLVE if limit is None else max(1, min(int(limit), MAX_AUTHOR_RESOLVE))

    fts = _fts_query_from_text(normalized)
    if fts:
        try:
            cur.execute(
                "SELECT rowid AS id FROM author_fts WHERE author_fts MATCH ? LIMIT ?;",
                (fts, lim),
            )
            ids = [int(r["id"]) for r in cur.fetchall()]
            if ids:
                return ids
        except sqlite3.Error:
            pass

    cur.execute("SELECT id FROM authors WHERE name LIKE ? LIMIT ?;", (f"%{normalized}%", lim))
    return [int(r["id"]) for r in cur.fetchall()]


def _placeholders(items: list[int]) -> str:
    return ",".join("?" for _ in items) if items else "NULL"


def _clamp_limit(value: int | None, default: int) -> int:
    if value is None:
        return default
    try:
        n = int(value)
    except Exception:
        return default
    return max(1, min(n, MAX_LIMIT))


class CoauthoredPairsRequest(BaseModel):
    left: list[str] = Field(default_factory=list)
    right: list[str] = Field(default_factory=list)
    limit_per_pair: int | None = None
    author_limit: int | None = None
    exact_base_match: bool = True
    year_min: int | None = None


class StartRequest(BaseModel):
    xml_gz_url: str = Field(default=DEFAULT_XML_GZ_URL)
    dtd_url: str = Field(default=DEFAULT_DTD_URL)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=100)
    progress_every: int = Field(default=DEFAULT_PROGRESS_EVERY, ge=1000)
    rebuild: bool = True


@dataclass(slots=True)
class PipelineState:
    status: str = "idle"
    step: str = "idle"
    message: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PipelineManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = PipelineState()

    def _append_log_locked(self, msg: str) -> None:
        self._state.logs.append(msg)
        if len(self._state.logs) > MAX_LOG_LINES:
            self._state.logs = self._state.logs[-MAX_LOG_LINES :]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._state.status,
                "step": self._state.step,
                "message": self._state.message,
                "started_at": self._state.started_at,
                "finished_at": self._state.finished_at,
                "progress": dict(self._state.progress),
                "logs": list(self._state.logs),
            }

    def start(self, req: StartRequest) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise HTTPException(status_code=409, detail="Pipeline is already running.")
            self._stop.clear()
            self._state = PipelineState(
                status="running",
                step="starting",
                message="",
                started_at=_now_iso(),
                finished_at=None,
                progress={},
                logs=[],
            )
            self._append_log_locked("Pipeline start requested.")

            config = PipelineConfig(
                xml_gz_url=req.xml_gz_url,
                dtd_url=req.dtd_url,
                data_dir=DATA_DIR,
                batch_size=req.batch_size,
                progress_every=req.progress_every,
                rebuild=req.rebuild,
            )

            self._thread = threading.Thread(
                target=self._run_job,
                args=(config,),
                daemon=True,
            )
            self._thread.start()
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        with self._lock:
            if self._state.status == "running":
                self._state.status = "stopping"
                self._state.message = "Stop requested."
                self._append_log_locked("Stop requested.")
        return self.snapshot()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise HTTPException(status_code=409, detail="Cannot reset while running.")
            self._stop.clear()
            self._state = PipelineState()
            self._append_log_locked("Reset.")
        return self.snapshot()

    def _run_job(self, config: PipelineConfig) -> None:
        def _log(msg: str) -> None:
            with self._lock:
                self._append_log_locked(msg)
                self._state.message = msg

        def _progress(phase: str, payload: dict[str, Any]) -> None:
            with self._lock:
                self._state.step = phase
                self._state.progress.update(payload)

        def _should_stop() -> bool:
            return self._stop.is_set()

        try:
            result = run_pipeline(
                config=config,
                log=_log,
                progress=_progress,
                should_stop=_should_stop,
            )
            with self._lock:
                self._state.status = "completed"
                self._state.step = "completed"
                self._state.message = "Completed."
                self._state.finished_at = _now_iso()
                self._state.progress.update(result)
                self._append_log_locked("Pipeline completed.")
        except InterruptedError:
            with self._lock:
                self._state.status = "stopped"
                self._state.step = "stopped"
                self._state.message = "Stopped."
                self._state.finished_at = _now_iso()
                self._append_log_locked("Pipeline stopped.")
        except Exception as exc:
            with self._lock:
                self._state.status = "error"
                self._state.step = "error"
                self._state.message = str(exc)
                self._state.finished_at = _now_iso()
                self._append_log_locked(f"Pipeline error: {exc}")


manager = PipelineManager()


def _safe_file_info(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
    except OSError:
        return {"exists": False, "size": None, "path": str(path)}
    return {"exists": True, "size": st.st_size, "path": str(path)}


def _load_pc_members() -> list[dict[str, str]]:
    csv_path = Path(os.getenv("PC_MEMBERS_CSV", str(BASE_DIR / "pc-members.csv"))).expanduser()
    if not csv_path.exists():
        return []
    members: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = (row.get("reviewer") or "").strip()
            affiliation = (row.get("affiliation") or "").strip()
            if name:
                members.append({"name": name, "affiliation": affiliation})
    return members


PC_MEMBERS = _load_pc_members()


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    conn = _get_connection()
    try:
        _ensure_fullmeta_schema(conn)
    finally:
        conn.close()
    return {"status": "ok"}


@app.get("/api/stats")
def api_stats() -> dict[str, Any]:
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM publications;")
        pub_count = int(cur.fetchone()["cnt"])
        cur.execute("SELECT COUNT(*) AS cnt FROM authors;")
        author_count = int(cur.fetchone()["cnt"])
    finally:
        conn.close()
    return {
        "publications": pub_count,
        "authors": author_count,
        "data_source": "DBLP",
        "data_date": _detect_data_date(),
    }


@app.get("/api/pc-members")
def api_pc_members() -> dict[str, Any]:
    return {"members": PC_MEMBERS, "count": len(PC_MEMBERS)}


@app.post("/api/coauthors/pairs")
def api_coauthors_pairs(
    payload: CoauthoredPairsRequest,
) -> dict[str, Any]:
    left_entries = _sanitize_author_entries(payload.left)
    right_entries = _sanitize_author_entries(payload.right)
    if not left_entries or not right_entries:
        raise HTTPException(status_code=400, detail="Both left and right author lists are required.")
    if len(left_entries) > MAX_ENTRIES_PER_SIDE or len(right_entries) > MAX_ENTRIES_PER_SIDE:
        raise HTTPException(status_code=400, detail=f"Too many authors. Max {MAX_ENTRIES_PER_SIDE} per side is allowed.")

    limit_per_pair = None if payload.limit_per_pair is None else _clamp_limit(payload.limit_per_pair, default=20)
    author_limit = payload.author_limit
    if author_limit is not None:
        author_limit = min(int(author_limit), MAX_AUTHOR_RESOLVE)
    year_min = payload.year_min

    conn = _get_connection()
    try:
        _ensure_fullmeta_schema(conn)

        left_ids: dict[str, list[int]] = {}
        right_ids: dict[str, list[int]] = {}

        for entry in left_entries:
            left_ids[entry] = _resolve_author_ids(
                conn,
                entry,
                limit=author_limit,
                exact_base_match=payload.exact_base_match,
            )
        for entry in right_entries:
            right_ids[entry] = _resolve_author_ids(
                conn,
                entry,
                limit=author_limit,
                exact_base_match=payload.exact_base_match,
            )

        matrix: dict[str, dict[str, int]] = {left: {} for left in left_entries}
        pair_pubs: list[dict[str, Any]] = []

        cur = conn.cursor()
        for left_entry, left_author_ids in left_ids.items():
            for right_entry, right_author_ids in right_ids.items():
                if not left_author_ids or not right_author_ids:
                    items: list[dict[str, Any]] = []
                else:
                    limit_sql = "" if limit_per_pair is None else "LIMIT ?"
                    year_filter_sql = "" if year_min is None else "AND p.year >= ?"
                    params: tuple[Any, ...] = (*left_author_ids, *right_author_ids)
                    if year_min is not None:
                        params = (*params, int(year_min))
                    if limit_per_pair is not None:
                        params = (*params, int(limit_per_pair))

                    order_sql = "ORDER BY (p.year IS NULL) ASC, p.year DESC, p.title ASC"
                    cur.execute(
                        f"""
                        SELECT DISTINCT p.title, p.year, p.venue, p.pub_type
                        FROM pub_authors pa1
                        JOIN pub_authors pa2 ON pa1.pub_id = pa2.pub_id
                        JOIN publications p ON p.id = pa1.pub_id
                        WHERE pa1.author_id IN ({_placeholders(left_author_ids)})
                          AND pa2.author_id IN ({_placeholders(right_author_ids)})
                        {year_filter_sql}
                        {order_sql}
                        {limit_sql};
                        """,
                        params,
                    )
                    rows = cur.fetchall()
                    items = [
                        {
                            "title": row["title"],
                            "year": row["year"],
                            "venue": row["venue"],
                            "pub_type": row["pub_type"],
                        }
                        for row in rows
                    ]

                matrix[left_entry][right_entry] = len(items)
                pair_pubs.append(
                    {
                        "left": left_entry,
                        "right": right_entry,
                        "count": len(items),
                        "items": items,
                    }
                )

        return {
            "limit_per_pair": limit_per_pair,
            "exact_base_match": payload.exact_base_match,
            "left_authors": left_entries,
            "right_authors": right_entries,
            "matrix": matrix,
            "pair_pubs": pair_pubs,
            "pair_count": len(pair_pubs),
        }
    finally:
        conn.close()


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    return {
        "default_xml_gz_url": DEFAULT_XML_GZ_URL,
        "default_dtd_url": DEFAULT_DTD_URL,
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "default_progress_every": DEFAULT_PROGRESS_EVERY,
        "data_dir": str(DATA_DIR),
    }


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    return manager.snapshot()


@app.get("/api/files")
def api_files() -> dict[str, Any]:
    return {
        "data_dir": str(DATA_DIR),
        "files": {
            "xml_gz": _safe_file_info(DATA_DIR / "dblp.xml.gz"),
            "xml": _safe_file_info(DATA_DIR / "dblp.xml"),
            "dtd": _safe_file_info(DATA_DIR / "dblp.dtd"),
            "db": _safe_file_info(DATA_DIR / "dblp.sqlite"),
            "db_wal": _safe_file_info(DATA_DIR / "dblp.sqlite-wal"),
            "db_shm": _safe_file_info(DATA_DIR / "dblp.sqlite-shm"),
        },
    }


@app.post("/api/start")
def api_start(req: StartRequest) -> dict[str, Any]:
    return manager.start(req)


@app.post("/api/stop")
def api_stop() -> dict[str, Any]:
    return manager.stop()


@app.post("/api/reset")
def api_reset() -> dict[str, Any]:
    return manager.reset()
