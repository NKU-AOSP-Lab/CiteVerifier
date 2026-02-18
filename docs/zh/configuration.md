# 配置说明

## 核心环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DBLP_DB_PATH` | `dblp.sqlite` | Web 查询使用的 DBLP SQLite 文件 |
| `CITEVERIFIER_DATA_DIR` | `${PROJECT_DIR}/data` | 运行时数据目录 |
| `CITEVERIFIER_RUNTIME_DB` | `${CITEVERIFIER_DATA_DIR}/runtime.sqlite` | runtime SQLite 路径 |

## 批处理限制

- 接口 `POST /api/search/title/batch` 单次最多 `200` 条
- 前端会去重并校验空行

## 运行时数据

- `single_search_events`
- `batch_runs`
- `batch_items`
- `event_logs`
