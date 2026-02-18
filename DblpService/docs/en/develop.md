# Development Guide

## 1. Backend Architecture

DblpService combines query APIs and build pipeline control in one FastAPI service.

Core layers:

1. **API layer** (`app.py`)
   - Bootstrap page and REST endpoints.
   - Coauthor pair query APIs for CoAuthors.
2. **Pipeline orchestration layer** (`PipelineManager` in `app.py`)
   - Start/stop/reset state machine.
   - Threaded execution and in-memory status snapshots.
3. **Build pipeline layer** (`dblp_builder/pipeline.py`)
   - Download DTD/XML.GZ.
   - Decompress XML.
   - Parse XML and rebuild SQLite + FTS indexes.

## 2. API Surface and Responsibilities

### Query APIs

- `GET /api/health`: schema readiness check.
- `GET /api/stats`: publication/author counters and data date.
- `GET /api/pc-members`: optional reviewer list.
- `POST /api/coauthors/pairs`: coauthor matrix + pair publication details.

### Build/control APIs

- `GET /api/config`: default build parameters.
- `GET /api/state`: pipeline runtime snapshot.
- `GET /api/files`: managed data files status.
- `POST /api/start`, `/api/stop`, `/api/reset`: pipeline lifecycle control.

## 3. Coauthor Query Path (`/api/coauthors/pairs`)

Execution flow:

1. Normalize/deduplicate left/right author entries.
2. Resolve candidate author IDs via exact match -> FTS -> LIKE fallback.
3. Join `pub_authors` twice to compute intersections.
4. Read publication metadata from `publications`.
5. Return matrix and per-pair publication lists.

Safety controls:

- Maximum authors per side (`MAX_ENTRIES_PER_SIDE`).
- Author resolve cap (`MAX_AUTHOR_RESOLVE`).
- Optional pair result limit and year filter.

## 4. Build Pipeline Internals (`dblp_builder/pipeline.py`)

Pipeline phases:

1. URL validation and trusted-host download.
2. XML decompression.
3. SQLite rebuild (optional cleanup of existing db/wal/shm).
4. XML iterparse with secure DTD resolver.
5. Batch insert into:
   - `publications`
   - `authors`
   - `pub_authors`
   - `title_fts`
   - `author_fts`

`PipelineManager` updates status, step, progress, and log buffers for frontend polling.

## 5. Data Model

Main DB tables:

- `publications(id, title, year, venue, pub_type, raw_xml)`
- `authors(id, name)`
- `pub_authors(pub_id, author_id)`
- `title_fts`, `author_fts` (FTS5 virtual tables)

SQLite tuning includes WAL, `busy_timeout`, and temp-store memory optimization.

## 6. Extensibility Notes

- Keep heavy build logic inside `dblp_builder/pipeline.py`; keep route handlers thin.
- Add new pipeline phases through progress callbacks so UI can observe them.
- If schema changes, update both `_ensure_fullmeta_schema()` checks and builder initialization.
- For new frontend controls, expose defaults in `/api/config` first, then bind in UI.
