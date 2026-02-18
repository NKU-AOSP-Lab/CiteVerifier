# 配置说明

## 核心环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DATA_DIR` | `./data` | DBLP 数据目录 |
| `DB_PATH` | `${DATA_DIR}/dblp.sqlite` | 查询数据库路径 |
| `CORS_ORIGINS` | `http://localhost:8090` | 允许的前端来源 |
| `DBLP_XML_GZ_URL` | `https://dblp.org/xml/dblp.xml.gz` | DBLP 数据源 |
| `DBLP_DTD_URL` | `https://dblp.org/xml/dblp.dtd` | DTD 数据源 |
| `BATCH_SIZE` | `1000` | 建库批处理大小 |
| `PROGRESS_EVERY` | `10000` | 进度输出频率 |

## 数据文件

`DATA_DIR` 中常见文件：

- `dblp.xml.gz`
- `dblp.xml`
- `dblp.dtd`
- `dblp.sqlite`
