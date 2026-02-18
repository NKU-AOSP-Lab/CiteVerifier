<div align="center" style="display:flex;justify-content:center;align-items:center;gap:8px;">
  <img src="./static/citeverifier-logo.svg" alt="CiteVerifier Logo" width="34" />
  <strong>CiteVerifier</strong>
</div>

<p align="center">DBLP-first citation verification toolkit with CLI and Web modes.</p>

<p align="center">[<a href="./README.md"><strong>EN</strong></a>] | [<a href="./README.zh-CN.md"><strong>CN</strong></a>]</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-1f7a8c" alt="version" />
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="python" />
  <img src="https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white" alt="fastapi" />
  <img src="https://img.shields.io/badge/docs-MkDocs-526CFE?logo=materialformkdocs&logoColor=white" alt="docs" />
</p>

## Overview

CiteVerifier verifies citation titles against DBLP with optional online enhancement strategies. It supports both CLI workflows and a lightweight Web interface.

## Core Capabilities

- Single-title and batch-title verification.
- Local DBLP cache and runtime telemetry storage.
- Optional online fallback chain for difficult cases.
- Shared backend integration with `DblpService`.

## Local Run

```bash
cd CiteVerifier
python -m pip install -r requirements.txt
python -m uvicorn web_app:app --host 0.0.0.0 --port 8092
```

CLI examples:

```bash
python verifier.py --title "Attention Is All You Need" --dblp-db dblp.sqlite
python verifier.py --input references.json --dblp-db dblp.sqlite
python verifier.py --sample
```

## Docker

```bash
cd CiteVerifier
docker compose up -d --build
```

Default services:

- `citeverifier-web`: `http://localhost:8092`
- `citeverifier-dblp-service`: `http://localhost:8093/bootstrap`

## Documentation

- English docs: https://citeverifier.readthedocs.io/en/latest/
- Docs source (in repo): `docs/en/`, `docs/zh/`
- Detailed runtime behavior (cache/concurrency/error handling) is documented in MkDocs.

Local preview:

```bash
cd CiteVerifier
python -m pip install -r docs/requirements.txt
mkdocs serve
```


