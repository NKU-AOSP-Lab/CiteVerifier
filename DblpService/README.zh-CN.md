<div align="center" style="display:flex;justify-content:center;align-items:center;gap:8px;">
  <img src="./static/dblpservice-logo.svg" alt="DblpService Logo" width="34" />
  <strong>DblpService</strong>
</div>

<p align="center">DBLP 数据构建与查询后端服务。</p>

<p align="center">[<a href="./README.md"><strong>EN</strong></a>] | [<a href="./README.zh-CN.md"><strong>CN</strong></a>]</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-1f7a8c" alt="version" />
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="python" />
  <img src="https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white" alt="fastapi" />
  <img src="https://img.shields.io/badge/docs-MkDocs-526CFE?logo=materialformkdocs&logoColor=white" alt="docs" />
</p>

## 项目概览

DblpService 是可复用的 DBLP 后端能力层，负责数据下载、解析、SQLite 构建和查询服务，既可独立部署，也可被 CoAuthors、CiteVerifier 等系统集成。

## 核心能力

- DBLP 源数据下载与解析流水线。
- Bootstrap 控制台（`/bootstrap`）。
- 查询接口（`/api/health`、`/api/stats`、`/api/coauthors/pairs`）。
- 流水线生命周期接口（`/api/start`、`/api/stop`、`/api/reset`、`/api/state`）。

## 快速开始

```bash
cd DblpService
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8091
```

访问：

- `http://localhost:8091/bootstrap`
- `http://localhost:8091/`

## 关键环境变量

- `DATA_DIR`（默认：`./data`）
- `DB_PATH`（默认：`${DATA_DIR}/dblp.sqlite`）
- `CORS_ORIGINS`（逗号分隔来源）
- `DB_BUSY_TIMEOUT_MS`（默认：`30000`）

## 文档

- 英文文档：`docs/en/`
- 中文文档：`docs/zh/`

本地预览：

```bash
cd DblpService
python -m pip install -r docs/requirements.txt
mkdocs serve
```


