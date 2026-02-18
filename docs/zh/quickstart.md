# 快速开始

## Web 本地运行

```bash
cd CiteVerifier
python -m pip install -r requirements.txt
python -m uvicorn web_app:app --host 0.0.0.0 --port 8092
```

访问：`http://localhost:8092`

## CLI 示例

```bash
python verifier.py --title "Attention Is All You Need" --dblp-db dblp.sqlite
python verifier.py --sample
```

## Docker 独立部署

```bash
cd CiteVerifier
docker compose up -d --build
```

默认服务：

- `citeverifier-web`: `8092`
- `citeverifier-dblp-service`: `8093`（Bootstrap 页面）
