# 接口说明

## 页面接口

- `GET /`
- `GET /bootstrap`

## 查询接口

- `GET /api/health`
- `GET /api/stats`
- `GET /api/pc-members`
- `POST /api/coauthors/pairs`

`/api/coauthors/pairs` 请求示例：

```json
{
  "left": ["Geoffrey Hinton"],
  "right": ["Yoshua Bengio"],
  "exact_base_match": true
}
```

## 构建控制接口

- `GET /api/config`
- `GET /api/state`
- `GET /api/files`
- `POST /api/start`
- `POST /api/stop`
- `POST /api/reset`
