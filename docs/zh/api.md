# 接口说明

## Web 页面

### `GET /`

返回标题检索页面。

## API

### `GET /api/health`

返回服务与 DBLP 数据可用性状态。

### `GET /api/runtime/stats`

返回 runtime 统计计数。

### `POST /api/search/title`

请求体：

```json
{ "title": "Attention Is All You Need", "max_candidates": 100000 }
```

### `POST /api/search/title/batch`

请求体：

```json
{ "titles": ["Paper A", "Paper B"], "max_candidates": 100000 }
```

返回 `summary`（批次概览）和 `items`（逐条结果）。
