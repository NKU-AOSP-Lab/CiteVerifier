<div align="center" style="display:flex;justify-content:center;align-items:center;gap:8px;">
  <img src="./static/dblpservice-logo.svg" alt="DblpService Logo" width="34" />
  <strong>DblpService</strong>
</div>

<p align="center">DBLP build-and-query backend service.</p>

<p align="center">[<a href="./README.md"><strong>EN</strong></a>] | [<a href="./README.zh-CN.md"><strong>CN</strong></a>]</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-1f7a8c" alt="version" />
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="python" />
  <img src="https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white" alt="fastapi" />
  <img src="https://img.shields.io/badge/docs-MkDocs-526CFE?logo=materialformkdocs&logoColor=white" alt="docs" />
</p>

## Overview

DblpService is a reusable DBLP backend for data bootstrap, SQLite build, and query serving.
It can be integrated by CoAuthors, CiteVerifier, and other systems that need local DBLP data services.

## Core Capabilities

- DBLP source download and parsing pipeline.
- Bootstrap control console (`/bootstrap`).
- Query APIs (`/api/health`, `/api/stats`, `/api/coauthors/pairs`).
- Pipeline lifecycle APIs (`/api/start`, `/api/stop`, `/api/reset`, `/api/state`).

## Quick Start

```bash
cd DblpService
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8091
```

Open:

- `http://localhost:8091/bootstrap`
- `http://localhost:8091/`

## Key Environment Variables

- `DATA_DIR` (default: `./data`)
- `DB_PATH` (default: `${DATA_DIR}/dblp.sqlite`)
- `CORS_ORIGINS` (comma-separated origins)
- `DB_BUSY_TIMEOUT_MS` (default: `30000`)

## Documentation

- English docs: `docs/en/`
- Chinese docs: `docs/zh/`

Local preview:

```bash
cd DblpService
python -m pip install -r docs/requirements.txt
mkdocs serve
```


