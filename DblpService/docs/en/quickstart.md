# Quick Start

## Local Run

```bash
cd DblpService
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8091
```

Open:

- `http://localhost:8091/bootstrap`
- `http://localhost:8091/api/health`

## Docker Run

```bash
docker compose up -d --build
```

## Initial DB Build

1. Open `/bootstrap`
2. Use default DBLP XML/DTD URLs
3. Start pipeline and monitor logs/state
4. Verify `/api/stats` after completion
