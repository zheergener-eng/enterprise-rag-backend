# 企业级 AI 知识库问答系统（RAG 后端原型）—— production-like 镜像。
#
# 说明：
# - 基础镜像 Python 3.12（与本地 .venv 的 3.12.7 保持一致）；
# - Embedding / Reranker 均以 device="cpu" 运行，故显式安装 CPU 版 torch，
#   避免 pip 默认拉取带 CUDA 的巨大 wheel，显著减小镜像体积；
# - 真实 .env 与 .venv 均被 .dockerignore 排除，绝不进入镜像；
# - 生产启动不启用 --reload。

FROM python:3.12-slim

# 不写 .pyc；stdout/stderr 不缓冲（日志实时可见）；pip 不落缓存
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装 CPU 版 torch，再装项目依赖（sentence-transformers 会复用已装的 CPU torch，
# 不会重新拉取 CUDA 版本）。
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# 仅拷贝运行所需的 app 代码（tests / evaluation / scripts / .env / .venv 由 .dockerignore 排除）
COPY app ./app

EXPOSE 8000

# 生产启动：监听所有网卡，不启用 --reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
