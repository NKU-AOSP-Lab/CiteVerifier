# 故障排查

## 返回 `DBLP database not found`

- 检查 `DBLP_DB_PATH` 是否存在
- Docker 场景下检查卷是否挂载到 `/data`

## 批处理返回 400

- 检查标题数量是否超过 200
- 检查是否传入空列表

## 页面可开但检索无结果

- 确认本地 DBLP 数据是否完成构建
- 调整 `max_candidates` 并重试
