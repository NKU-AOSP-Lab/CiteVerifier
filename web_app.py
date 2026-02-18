from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from dblp_match import (
    _db_has_word_index,
    _sqlite_readonly_fast,
    load_all_titles_from_db,
    search_dblp_brute_force,
    search_dblp_by_index,
)
from runtime_store import RuntimeStore

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = Path(os.getenv("DBLP_DB_PATH", "dblp.sqlite")).expanduser().resolve()
DATA_DIR = Path(os.getenv("CITEVERIFIER_DATA_DIR", str(BASE_DIR / "data"))).expanduser().resolve()
RUNTIME_DB_PATH = Path(
    os.getenv("CITEVERIFIER_RUNTIME_DB", str(DATA_DIR / "runtime.sqlite"))
).expanduser().resolve()
MAX_BATCH_TITLES = 200

APP_VERSION = "0.1.0"

app = FastAPI(
    title="CiteVerifier Web",
    description="Web UI for DBLP title lookup in CiteVerifier.",
    version=APP_VERSION,
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

_brute_cache_lock = threading.Lock()
_brute_cache: dict[str, list[tuple[int, str, str]]] = {}
runtime_store = RuntimeStore(RUNTIME_DB_PATH)


class TitleSearchRequest(BaseModel):
    title: str = Field(..., min_length=1)
    max_candidates: int = Field(default=100000, ge=1, le=500000)


class BatchTitleSearchRequest(BaseModel):
    titles: list[str] = Field(default_factory=list)
    max_candidates: int = Field(default=100000, ge=1, le=500000)


def _resolve_db_path() -> Path:
    return DEFAULT_DB_PATH


def _normalize_title(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _normalize_title_list(values: list[str]) -> list[str]:
    dedup: list[str] = []
    seen: set[str] = set()
    for raw in values:
        title = _normalize_title(raw)
        if not title:
            continue
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(title)
    return dedup


def _title_hash(title: str) -> str:
    return hashlib.sha256(title.casefold().encode("utf-8")).hexdigest()[:24]


def _fetch_publication_meta(db_path: Path, pub_id: int | None) -> dict[str, Any]:
    if pub_id is None:
        return {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='publications'")
        if cur.fetchone() is None:
            return {}
        cur.execute(
            "SELECT year, venue, pub_type FROM publications WHERE id = ? LIMIT 1",
            (int(pub_id),),
        )
        row = cur.fetchone()
        if row is None:
            return {}
        return {
            "year": row["year"],
            "venue": row["venue"],
            "pub_type": row["pub_type"],
        }
    finally:
        conn.close()


def _search_title(db_path: Path, title: str, max_candidates: int) -> tuple[dict[str, Any] | None, str]:
    conn = sqlite3.connect(str(db_path))
    try:
        if _db_has_word_index(conn):
            _sqlite_readonly_fast(conn)
            result = search_dblp_by_index(conn, title, max_candidates=max_candidates)
            return result, "indexed"
    finally:
        conn.close()

    cache_key = str(db_path)
    with _brute_cache_lock:
        all_titles = _brute_cache.get(cache_key)
        if all_titles is None:
            all_titles = load_all_titles_from_db(db_path, quiet=True)
            _brute_cache[cache_key] = all_titles
    result = search_dblp_brute_force(all_titles, title)
    return result, "bruteforce"


def _single_search_result(db_path: Path, title: str, max_candidates: int) -> dict[str, Any]:
    match, _ = _search_title(db_path, title, max_candidates)
    if not match:
        return {
            "found": False,
            "query_title": title,
        }

    pub_id = match.get("dblp_id")
    meta = _fetch_publication_meta(db_path, pub_id if isinstance(pub_id, int) else None)
    return {
        "found": True,
        "query_title": title,
        "dblp_id": match.get("dblp_id"),
        "dblp_title": match.get("dblp_title"),
        "dblp_title_similarity": match.get("dblp_title_similarity"),
        **meta,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    visit_count = runtime_store.increment_counter("web_page_views")
    return templates.TemplateResponse(
        "web_index.html",
        {
            "request": request,
            "app_version": APP_VERSION,
            "visit_count": visit_count,
        },
    )


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    path = _resolve_db_path()
    if not path.exists():
        return {
            "status": "error",
            "detail": "DBLP SQLite file not found.",
        }
    return {
        "status": "ok",
    }


@app.get("/api/runtime/stats")
def api_runtime_stats() -> dict[str, Any]:
    return runtime_store.stats()


@app.post("/api/search/title")
def api_search_title(payload: TitleSearchRequest) -> dict[str, Any]:
    title = _normalize_title(payload.title)
    if not title:
        raise HTTPException(status_code=400, detail="Title is required.")

    started_at = time.perf_counter()
    found = False
    error_message: str | None = None
    try:
        db_path = _resolve_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="DBLP database not found.")
        result = _single_search_result(db_path, title, payload.max_candidates)
        found = bool(result.get("found"))
        return result
    except HTTPException as exc:
        error_message = str(exc.detail)
        raise
    except Exception as exc:
        error_message = str(exc)
        runtime_store.log_event(
            "ERROR",
            "single_title_search_failed",
            {"title": title, "error": str(exc)},
        )
        raise HTTPException(status_code=500, detail="Internal server error.") from exc
    finally:
        duration_ms = max(0, int((time.perf_counter() - started_at) * 1000))
        runtime_store.increment_counter("single_search_requests")
        if found:
            runtime_store.increment_counter("single_search_found")
        if error_message:
            runtime_store.increment_counter("single_search_errors")
        runtime_store.record_single_search(
            query_title=title,
            query_hash=_title_hash(title),
            found=found,
            max_candidates=payload.max_candidates,
            duration_ms=duration_ms,
            error_message=error_message,
        )


@app.post("/api/search/title/batch")
def api_search_title_batch(payload: BatchTitleSearchRequest) -> dict[str, Any]:
    titles = _normalize_title_list(payload.titles)
    if not titles:
        raise HTTPException(status_code=400, detail="At least one title is required.")
    if len(titles) > MAX_BATCH_TITLES:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit ({MAX_BATCH_TITLES}).",
        )

    db_path = _resolve_db_path()
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="DBLP database not found.")

    runtime_store.increment_counter("batch_search_requests")
    run_id = runtime_store.start_batch_run(
        total_input=len(titles),
        max_candidates=payload.max_candidates,
    )

    started_at = time.perf_counter()
    found_count = 0
    items: list[dict[str, Any]] = []

    for idx, title in enumerate(titles, start=1):
        item_started_at = time.perf_counter()
        error_message: str | None = None
        result: dict[str, Any]
        try:
            result = _single_search_result(db_path, title, payload.max_candidates)
        except Exception as exc:
            error_message = str(exc)
            result = {
                "found": False,
                "query_title": title,
            }

        duration_ms = max(0, int((time.perf_counter() - item_started_at) * 1000))
        found = bool(result.get("found"))
        if found:
            found_count += 1

        runtime_store.record_batch_item(
            run_id,
            item_index=idx,
            query_title=title,
            found=found,
            dblp_id=result.get("dblp_id") if isinstance(result.get("dblp_id"), int) else None,
            dblp_title=result.get("dblp_title"),
            dblp_title_similarity=result.get("dblp_title_similarity"),
            year=result.get("year"),
            venue=result.get("venue"),
            pub_type=result.get("pub_type"),
            duration_ms=duration_ms,
            error_message=error_message,
        )

        items.append(
            {
                "index": idx,
                **result,
                "duration_ms": duration_ms,
                "error_message": error_message,
            }
        )

    total_duration_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    runtime_store.finish_batch_run(
        run_id,
        total_processed=len(items),
        found_count=found_count,
        duration_ms=total_duration_ms,
        status="completed",
        error_message=None,
    )
    runtime_store.increment_counter("batch_search_items_total", delta=len(items))
    runtime_store.increment_counter("batch_search_found_total", delta=found_count)

    return {
        "summary": {
            "run_id": run_id,
            "limit": MAX_BATCH_TITLES,
            "total_input": len(titles),
            "total_processed": len(items),
            "found_count": found_count,
            "not_found_count": len(items) - found_count,
            "max_candidates": payload.max_candidates,
            "duration_ms": total_duration_ms,
        },
        "items": items,
    }



