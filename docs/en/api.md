# API Reference

## Web Page

### `GET /`

Returns the title search page.

## API

### `GET /api/health`

Returns service and DBLP data availability.

### `GET /api/runtime/stats`

Returns runtime counters and record counts.

### `POST /api/search/title`

Request body:

```json
{ "title": "Attention Is All You Need", "max_candidates": 100000 }
```

### `POST /api/search/title/batch`

Request body:

```json
{ "titles": ["Paper A", "Paper B"], "max_candidates": 100000 }
```

Response includes `summary` (batch-level stats) and `items` (per-title results).
