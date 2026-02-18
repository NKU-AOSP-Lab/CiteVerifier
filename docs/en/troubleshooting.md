# Troubleshooting

## `DBLP database not found`

- Verify `DBLP_DB_PATH` exists
- In Docker, verify volume is mounted to `/data`

## Batch API Returns 400

- Ensure title count does not exceed 200
- Ensure request does not send an empty list

## Page Loads but No Matches

- Verify DBLP database build is completed
- Increase `max_candidates` and retry
