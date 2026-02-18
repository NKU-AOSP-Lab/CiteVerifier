# API Reference

## Page Endpoints

- `GET /`
- `GET /bootstrap`

## Query Endpoints

- `GET /api/health`
- `GET /api/stats`
- `GET /api/pc-members`
- `POST /api/coauthors/pairs`

`/api/coauthors/pairs` request example:

```json
{
  "left": ["Geoffrey Hinton"],
  "right": ["Yoshua Bengio"],
  "exact_base_match": true
}
```

## Pipeline Control Endpoints

- `GET /api/config`
- `GET /api/state`
- `GET /api/files`
- `POST /api/start`
- `POST /api/stop`
- `POST /api/reset`
