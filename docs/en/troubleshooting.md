# Troubleshooting

## `/api/health` Not `ok`

- Verify `DB_PATH` exists and is readable
- Verify SQLite schema is complete

## Pipeline Stuck or Failed

- Inspect `/api/state` and `/api/files`
- Verify XML/DTD URLs are reachable
- Check disk capacity and write permission

## Frontend CORS Failure

- Ensure `CORS_ORIGINS` includes the frontend origin
