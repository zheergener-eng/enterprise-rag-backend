"""FastAPI 应用入口。

职责：创建应用实例、在 lifespan 中做启动/关闭清理、挂载版本化路由。
当前为骨架阶段：路由已就位，业务逻辑由 services 层逐步填充。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.error_handlers import register_exception_handlers
from app.api.v1 import chat, document


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：在此初始化/释放重资源。

    TODO(后续阶段):
      - 启动时懒加载 embedding 模型
      - 建立 Redis 连接池并创建向量索引
    """
    yield
    # 关闭时释放资源（暂留空）


app = FastAPI(
    title="Enterprise RAG 后端原型",
    description="企业级 AI 知识库问答系统（RAG）",
    version="0.1.0",
    lifespan=lifespan,
)

# 统一异常处理器：APIError / 校验失败(422) / Redis 不可用(503) / 未预期异常(500)
register_exception_handlers(app)

# 挂载版本化路由（前缀统一为 /api/v1）
app.include_router(document.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


@app.get("/health", tags=["meta"])
async def health() -> dict:
    """健康检查，便于确认服务已启动。"""
    return {"status": "ok"}
