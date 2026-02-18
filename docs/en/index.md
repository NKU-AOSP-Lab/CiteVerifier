<p align="center">
  <img src="../assets/logo.svg" alt="DblpService Logo" width="56" style="vertical-align:middle;" />
  <span style="font-size:1.8rem;font-weight:700;vertical-align:middle;margin-left:8px;">DblpService Documentation</span>
</p>

![Version](https://img.shields.io/badge/version-0.1.0-1f7a8c)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white)

Language: **English** (default) | [Chinese](../zh/)

DblpService is the DBLP build-and-query backend used by CoAuthors and CiteVerifier.

## Core Responsibilities

- DBLP source download and local build pipeline
- Bootstrap control console (`/bootstrap`)
- Query APIs for health, stats, and coauthor pairs
- Pipeline lifecycle control (`start / stop / reset / state`)

## Integration Scope

- Default service port: `8091`
- Direct backend for `CoAuthors`
- Shared/backup build-query backend for `CiteVerifier`

## Operational Characteristics

- SQLite-backed query service (`dblp.sqlite`)
- Fullmeta schema checks before query APIs
- Configurable lock timeout (`DB_BUSY_TIMEOUT_MS`)
- CORS control via `CORS_ORIGINS`

## Recommended Reader Path

1. [Quick Start](quickstart.md)
2. [Configuration](configuration.md)
3. [API Reference](api.md)
4. [Development Guide](develop.md)
5. [Operations](operations.md)
6. [Troubleshooting](troubleshooting.md)
7. [Changelog](changelog.md)


