# 故障排查

## `/api/health` 非 `ok`

- 检查 `DB_PATH` 是否存在且可读
- 检查数据库 schema 是否完整

## 构建卡住或失败

- 查看 `/api/state` 与 `/api/files`
- 检查 XML/DTD 下载地址是否可达
- 检查磁盘空间与写权限

## 前端跨域失败

- 检查 `CORS_ORIGINS` 配置是否包含前端地址
