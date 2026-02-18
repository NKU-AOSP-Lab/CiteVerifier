# 快速开始

## 本地运行

```bash
cd DblpService
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8091
```

访问：

- `http://localhost:8091/bootstrap`
- `http://localhost:8091/api/health`

## Docker 运行

```bash
docker compose up -d --build
```

## 首次建库

1. 打开 `/bootstrap`
2. 使用默认 DBLP XML/DTD 下载地址
3. 点击启动构建并观察状态日志
4. 构建完成后验证 `/api/stats`
