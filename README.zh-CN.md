<div align="center" style="display:flex;justify-content:center;align-items:center;gap:8px;">
  <img src="./static/citeverifier-logo.svg" alt="CiteVerifier Logo" width="34" />
  <strong>CiteVerifier</strong>
</div>

<p align="center">以 DBLP 为核心的文献引用校验工具，支持 CLI 与 Web 两种模式。</p>

<p align="center">[<a href="./README.md"><strong>EN</strong></a>] | [<a href="./README.zh-CN.md"><strong>CN</strong></a>]</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-1f7a8c" alt="version" />
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="python" />
  <img src="https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white" alt="fastapi" />
  <img src="https://img.shields.io/badge/docs-MkDocs-526CFE?logo=materialformkdocs&logoColor=white" alt="docs" />
</p>

## 项目概览

CiteVerifier 用于文献题名核验，默认基于本地 DBLP 数据进行匹配，并可按需启用在线增强策略。项目同时提供命令行与 Web 接口，方便批处理和可视化验收。

## 核心能力

- 单条与批量题名校验。
- 本地 DBLP 数据缓存与运行时遥测。
- 可选在线兜底链路，提升复杂场景召回。
- 与 `DblpService` 共享后端数据能力。

## 本地运行

```bash
cd CiteVerifier
python -m pip install -r requirements.txt
python -m uvicorn web_app:app --host 0.0.0.0 --port 8092
```

CLI 示例：

```bash
python verifier.py --title "Attention Is All You Need" --dblp-db dblp.sqlite
python verifier.py --input references.json --dblp-db dblp.sqlite
python verifier.py --sample
```

## Docker 启动

```bash
cd CiteVerifier
docker compose up -d --build
```

默认服务：

- `citeverifier-web`: `http://localhost:8092`
- `citeverifier-dblp-service`: `http://localhost:8093/bootstrap`

## 文档

- 英文文档：`docs/en/`
- 中文文档：`docs/zh/`
- 运行时细节（缓存、并发约束、错误处理）请参考 MkDocs 文档页面。

本地预览：

```bash
cd CiteVerifier
python -m pip install -r docs/requirements.txt
mkdocs serve
```


