<p align="center">
  <img src="/en/latest/image/logo.svg" alt="CiteVerifier Logo" width="56" style="vertical-align:middle;" />
  <span style="font-size:1.8rem;font-weight:700;vertical-align:middle;margin-left:8px;">CiteVerifier 文档</span>
</p>

CiteVerifier 是一个以 DBLP 为优先的数据核验项目，提供 CLI 与 Web 两种使用方式。

## 核心能力

- 基于本地 DBLP SQLite 的单条标题检索
- 支持批量标题校验（单次请求上限 `<=200`）
- 记录运行时遥测与批处理结果
- 可选在线策略链，用于增强复杂场景匹配

## 服务关系

- Web 服务入口：`http://localhost:8092`
- 可选内置构建后端：`CiteVerifier/DblpService`（默认映射 `8093`）
- Docker 模式下，Web 服务读取本地 DBLP SQLite（`/data/dblp.sqlite`）

## 架构概览

- CLI 主流程：`verifier.py`
- Web 服务：`web_app.py`
- 运行时指标存储：`runtime_store.py`
- DBLP 匹配工具：`dblp_match.py`
- 统一缓存数据库：`scholar_results.db`

## 推荐阅读顺序

1. [快速开始](quickstart.md)
2. [配置说明](configuration.md)
3. [接口说明](api.md)
4. [开发文档](develop.md)
5. [运维手册](operations.md)
6. [故障排查](troubleshooting.md)
7. [变更记录](changelog.md)

## 适用对象

- 需要部署 DBLP 本地核验服务的工程师
- 运行批量引用核验流程的研究人员
- 关注缓存命中率与运行时指标的维护者


