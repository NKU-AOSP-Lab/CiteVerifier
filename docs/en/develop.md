# Development Guide

## 1. Architecture Overview

CiteVerifier has two execution surfaces:

1. **CLI verification pipeline** (`verifier.py`)
   - Runs strategy-chain verification (DBLP/cache/online/LLM reparse).
   - Exports results and statistics.
2. **Web lookup service** (`web_app.py`)
   - Serves single and batch title lookup.
   - Persists runtime metrics into `runtime.sqlite`.

Supporting modules:

- `unified_database.py`: verification cache DB (`scholar_results` / `search_results`).
- `runtime_store.py`: Web runtime telemetry and batch run records.
- `dblp_match.py`: local DBLP match functions (indexed path + brute-force fallback).

## 2. CLI Verification Flow (with cache)

### 2.1 Strategy Chain Assembly Rules

Actual behavior in `ScrapingDogVerifier._build_verification_chain()`:

- Default (`--enable-online` disabled):
  1. `DblpVerificationStrategy`
  2. `CacheVerificationStrategy`
- With `--enable-online` enabled, append:
  3. `APIVerificationStrategy`
  4. `GoogleFallbackStrategy`
  5. `LLMReparseStrategy`

If online clients fail to import/initialize, runtime degrades automatically to DBLP + cache only.

### 2.2 CLI Cache Read/Write Semantics

Cache DB is `scholar_results.db` (`UnifiedDatabase`):

- **Read path**: `CacheVerificationStrategy` calls `search_scholar_by_title(title)`.
  - SQL is `WHERE title LIKE '%{title}%' ORDER BY created_at DESC LIMIT 1`.
  - Hit result is still validated by the unified validator, not blindly accepted as VALID.
- **Write path**: only writes when final verification is `VALID` and `best_match` exists.
- **Dedup rule**: unique index `UNIQUE(title, authors, year)`; duplicate writes are ignored by default.

So CLI cache is a result cache (not request cache), and `LIKE`-based retrieval may introduce near-match risk.

## 3. Web Lookup Flow and Caching

### 3.1 Single Lookup (`POST /api/search/title`)

1. Normalize title (trim + collapse whitespace).
2. Verify DB file exists, otherwise `404`.
3. Search strategy:
   - use indexed search when word index exists;
   - fallback to brute-force otherwise.
4. If matched, enrich with publication metadata (`year/venue/pub_type`).
5. Record runtime counters and single-search event.

### 3.2 Batch Lookup (`POST /api/search/title/batch`)

1. Normalize + de-duplicate titles (`casefold`).
2. Empty list -> `400`; over `MAX_BATCH_TITLES=200` -> `400`.
3. Process each title in a request-local serial loop (`for`), writing one `batch_items` row per title.
4. Finalize `batch_runs` summary and counters.

### 3.3 Web In-Memory Cache Semantics

`web_app.py` keeps `_brute_cache: dict[db_path -> all_titles]`:

- used only for brute-force mode;
- first load is protected by `_brute_cache_lock`, then reused;
- no TTL, no size cap, no active eviction;
- cache is process-local and lost on process restart.

## 4. Concurrency Model and Overload Behavior (Wait / Reject / Degrade)

| Scenario | Constraint | Behavior when exceeded | Result |
|---|---|---|---|
| CLI batch concurrency | `asyncio.Semaphore(max_concurrent)`, default `--concurrent=10` | wait on semaphore | queued tasks continue, no reject |
| CLI `--concurrent` value | no hard cap in code | no automatic reject | can amplify IO pressure, limited by downstream systems |
| Web batch size | max `200` titles | immediate reject | `400 Batch size exceeds limit` |
| Web `max_candidates` | `1..500000` (Pydantic) | immediate reject | `422` |
| First brute-cache load | `_brute_cache_lock` mutex | wait for lock | continue after load |
| Per-request Web batch processing | serial `for` loop | no reject path, only longer processing | increased latency |
| Web runtime SQLite contention | `sqlite timeout=30s` + `WAL` | wait for lock first | timeout raises exception (typically `500`) |
| CLI online clients unavailable | import/init failure | automatic degradation | DBLP + cache mode continues |

Direct answer to the acceptance question:

- CLI over-concurrency means **wait** (not forbidden).
- Web batch-size overflow means **forbidden with 400**.
- SQLite lock contention means **wait first, then fail on timeout**.

## 5. Validation Constraints and Error Codes

### 5.1 Web API

- `POST /api/search/title`
  - `title`: minimum length 1; empty yields `422`.
  - `max_candidates`: `1..500000`; out-of-range yields `422`.
- `POST /api/search/title/batch`
  - normalized title list must contain at least 1 title, else `400`.
  - normalized list above 200 returns `400`.
- `GET /api/health`
  - when DB is missing, returns `{"status":"error"}` with HTTP 200.

### 5.2 Key CLI options

- `--dblp-db`: path to local DBLP sqlite.
- `--dblp-threshold`: DBLP title similarity threshold (default `0.9`).
- `--dblp-max-candidates`: DBLP candidate cap (default `100000`).
- `--disable-dblp`: skip DBLP pre-match.
- `--enable-online`: enable online strategy chain.

## 6. Web Runtime Data Model

`runtime_store.py` tables:

- `runtime_counters`
- `single_search_events`
- `batch_runs`
- `batch_items`
- `event_logs`

Recommended operational metrics:

- `single_search_requests / single_search_errors`
- `batch_search_requests`
- `batch_search_items_total / batch_search_found_total`
- `batch_runs.status` and `error_message`

## 7. Implementation Guidance (Cache + Concurrency)

- If cross-process cache is required, replace `_brute_cache` with an explicit shared cache and define TTL/capacity.
- If you add intra-request parallelism for Web batch, design DB connection and timeout policy first.
- For CLI online reliability, add global LLM/API rate limiter and backoff retry policies.
- Every new endpoint should document overload semantics explicitly: wait, reject, or degrade.