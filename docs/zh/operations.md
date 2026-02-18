# 运维手册

## 部署建议

- 推荐在同一个 compose 中同时启动 `citeverifier-web` 与 `citeverifier-dblp-service`
- 通过共享卷挂载 DBLP 数据库

## 升级流程

1. 升级并验证 DblpService 可建库
2. 确认 `/data/dblp.sqlite` 可被 Web 容器读取
3. 升级 Web 容器并验证单条/批处理接口

## 备份建议

- 备份 `runtime.sqlite`
- 大批量历史建议归档 `batch_runs` / `batch_items`

## 可观测性

- 健康检查：`GET /api/health`
- 批处理成功率：`found_count / total_processed`
- 错误统计：`single_search_errors` 与 `error_message`
