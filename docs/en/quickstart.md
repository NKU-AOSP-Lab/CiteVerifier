# Quick Start

## Local Web Run

```bash
cd CiteVerifier
python -m pip install -r requirements.txt
python -m uvicorn web_app:app --host 0.0.0.0 --port 8092
```

Open: `http://localhost:8092`

## CLI Examples

```bash
python verifier.py --title "Attention Is All You Need" --dblp-db dblp.sqlite
python verifier.py --sample
```

## Standalone Docker Deployment

```bash
cd CiteVerifier
docker compose up -d --build
```

Default services:

- `citeverifier-web`: `8092`
- `citeverifier-dblp-service`: `8093` (Bootstrap page)
