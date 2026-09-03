# Docker Compose 运行说明（最小版）

## 前置条件

- Docker Desktop（含 Docker Compose v2）
- 已复制 `.env` 并填入真实 `DEEPSEEK_API_KEY`

## 启动

```bash
# 1. 准备环境变量（Windows 下可手动复制 .env.example 为 .env 并填入密钥）
cp .env.example .env

# 2. 构建并启动（Redis Stack + FastAPI）
docker compose up -d --build

# 3. 打开 API 文档 / 健康检查
#    http://localhost:8000/docs
#    http://localhost:8000/health
```

## 停止 / 重启 / 日志

```bash
docker compose down          # 停止并移除容器（保留数据卷）
docker compose up -d         # 重新启动（复用已有数据卷）
docker compose logs -f api   # 查看 API 日志
docker compose ps            # 查看服务状态
```

## 数据持久化

- `redis_data` 卷：Redis 知识库 / 会话数据（含向量索引）
- `hf_cache` 卷：HuggingFace 模型缓存（Embedding + Reranker，首次启动联网下载）

仅 `docker compose down` 不会删除数据；只有 `docker compose down -v` 才会显式删除以上命名卷。

## 与本机已有 Redis 冲突

若本机已运行 `enterprise-rag-redis` 容器占用 6379，请先：

```bash
docker stop enterprise-rag-redis
```

再执行 `docker compose up -d --build`。

## 本地开发方式仍可用

```bash
python -m venv .venv
.venv/Scripts/activate     # Windows
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload
```

本地开发方式与 Docker Compose 互不影响，二者均为可选运行方式。
