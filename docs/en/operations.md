# Operations

## Capacity Planning

- Initial pipeline build can be long-running and disk-heavy
- Mount `DATA_DIR` to persistent storage

## Production Guidance

- Expose only required APIs and Bootstrap UI
- Put reverse proxy and access controls in front
- Schedule periodic rebuilds to refresh DBLP data

## Upgrade Procedure

1. Back up `dblp.sqlite`
2. Upgrade service image
3. Rebuild/verify `/api/health` and `/api/stats`
4. Switch frontend traffic after validation
