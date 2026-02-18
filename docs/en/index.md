<p align="center">
  <img src="../assets/logo.svg" alt="CiteVerifier Logo" width="56" style="vertical-align:middle;" />
  <span style="font-size:1.8rem;font-weight:700;vertical-align:middle;margin-left:8px;">CiteVerifier Documentation</span>
</p>

CiteVerifier is a DBLP-first citation/title verification project with CLI and Web interfaces.

## Core Capability Set

- Single-title lookup against local DBLP SQLite
- Batch-title lookup with request limit control (`<=200`)
- Runtime telemetry and batch run persistence
- Optional online strategy chain for extended matching

## Service Relationship

- Web endpoint: `http://localhost:8092`
- Optional bundled builder backend: `CiteVerifier/DblpService` (`8093` mapping by default)
- Web service reads local DBLP SQLite (`/data/dblp.sqlite` in Docker mode)

## Architecture at a Glance

- CLI pipeline: `verifier.py`
- Web service: `web_app.py`
- Runtime metrics store: `runtime_store.py`
- DBLP matching utilities: `dblp_match.py`
- Unified cache DB: `scholar_results.db`

## Recommended Reader Path

1. [Quick Start](quickstart.md)
2. [Configuration](configuration.md)
3. [API Reference](api.md)
4. [Development Guide](develop.md)
5. [Operations](operations.md)
6. [Troubleshooting](troubleshooting.md)
7. [Changelog](changelog.md)

## Audience

- Engineers deploying DBLP-only lookup/verification services
- Researchers running batch verification pipelines
- Maintainers analyzing cache hit rates and runtime telemetry


