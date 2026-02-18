# 开发文档

## 1. 后端架构

DblpService 是一个 FastAPI 后端，统一承载“建库控制 + 查询 API”。

核心分层：

1. **API 层**（`app.py`）
   - 提供 Bootstrap 页面和 REST 接口。
   - 对 CoAuthors 暴露共作查询能力。
2. **流水线调度层**（`app.py` 内 `PipelineManager`）
   - 维护 start/stop/reset 状态机。
   - 通过后台线程执行任务并暴露快照。
3. **建库执行层**（`dblp_builder/pipeline.py`）
   - 下载 DTD/XML.GZ。
   - 解压 XML。
   - 解析 XML 并重建 SQLite + FTS 索引。

## 2. API 职责划分

### 查询接口

- `GET /api/health`：数据库与 schema 可用性检查。
- `GET /api/stats`：论文/作者规模与数据日期。
- `GET /api/pc-members`：可选 PC 成员列表。
- `POST /api/coauthors/pairs`：共作矩阵与配对论文明细。

### 建库控制接口

- `GET /api/config`：默认建库参数。
- `GET /api/state`：任务运行状态快照。
- `GET /api/files`：数据文件存在性与大小。
- `POST /api/start`、`/api/stop`、`/api/reset`：任务生命周期控制。

## 3. 共作查询实现（`/api/coauthors/pairs`）

主流程：

1. 规范化并去重左右作者输入。
2. 作者 ID 解析：精确匹配 -> FTS -> LIKE 回退。
3. 通过 `pub_authors` 双重连接计算交集。
4. 从 `publications` 读取标题/年份/venue/type。
5. 输出矩阵与 pair 级论文列表。

约束控制：

- 每侧作者上限 `MAX_ENTRIES_PER_SIDE`。
- 作者解析上限 `MAX_AUTHOR_RESOLVE`。
- 支持按年份过滤和每对结果条数限制。

## 4. 建库流水线实现（`dblp_builder/pipeline.py`）

阶段顺序：

1. URL 校验与可信主机下载。
2. XML.GZ 解压。
3. 可选重建（清理 sqlite/wal/shm）。
4. 安全 DTD 解析并 iterparse 处理。
5. 批量写入：
   - `publications`
   - `authors`
   - `pub_authors`
   - `title_fts`
   - `author_fts`

`PipelineManager` 持续维护 `status/step/progress/logs`，前端轮询展示。

## 5. 数据模型

核心表：

- `publications(id, title, year, venue, pub_type, raw_xml)`
- `authors(id, name)`
- `pub_authors(pub_id, author_id)`
- `title_fts`、`author_fts`（FTS5）

SQLite 使用 WAL、`busy_timeout` 和内存临时存储优化并发与性能。

## 6. 扩展建议

- 建库重逻辑尽量集中在 `dblp_builder/pipeline.py`，路由层保持轻量。
- 新增流水线阶段时，务必通过 progress 回调暴露给 UI。
- 调整 schema 时同步更新 `_ensure_fullmeta_schema()` 与初始化建表逻辑。
- 新参数优先进入 `/api/config`，再由前端绑定控件，避免前后端漂移。
