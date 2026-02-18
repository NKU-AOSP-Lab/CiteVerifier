# Configuration

## Core Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DBLP_DB_PATH` | `dblp.sqlite` | DBLP SQLite used by Web search |
| `CITEVERIFIER_DATA_DIR` | `${PROJECT_DIR}/data` | Runtime data folder |
| `CITEVERIFIER_RUNTIME_DB` | `${CITEVERIFIER_DATA_DIR}/runtime.sqlite` | Runtime SQLite path |

## Batch Limits

- `POST /api/search/title/batch` accepts at most `200` titles
- Web UI deduplicates entries and filters empty lines

## Runtime Data Tables

- `single_search_events`
- `batch_runs`
- `batch_items`
- `event_logs`
