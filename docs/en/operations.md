# Operations

## Deployment Guidance

- Start `citeverifier-web` and bundled `citeverifier-dblp-service` in one compose stack
- Mount shared volume for DBLP SQLite data

## Upgrade Flow

1. Upgrade DblpService and verify build capability
2. Verify `/data/dblp.sqlite` is readable from Web container
3. Upgrade Web and validate single/batch endpoints

## Backup Guidance

- Back up `runtime.sqlite`
- Archive large `batch_runs` / `batch_items` history periodically

## Observability

- Health check: `GET /api/health`
- Batch success ratio: `found_count / total_processed`
- Error counters: `single_search_errors` and item-level `error_message`
