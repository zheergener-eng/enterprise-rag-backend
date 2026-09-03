"""集中式配置。

所有可调参数均从环境变量 / .env 读取，避免在代码中硬编码密钥或地址。
用法：`from app.config import settings`
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# 加载项目根目录下的 .env（若存在）；已存在的环境变量优先于 .env。
load_dotenv()


def _get_int(name: str, default: int) -> int:
    """读取整型环境变量，未设置时返回默认值。"""
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def _get_float(name: str, default: float) -> float:
    """读取浮点环境变量，未设置时返回默认值；非法值抛带变量名的 ValueError。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a numeric value, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """应用全局配置（只读）。"""

    # --- DeepSeek（OpenAI 兼容协议）---
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str

    # --- Redis Stack ---
    redis_url: str
    redis_index_name: str
    chunk_prefix: str
    session_prefix: str

    # --- Embedding ---
    embedding_model: str
    embedding_dim: int

    # --- Reranker（Cross-Encoder 精排）---
    reranker_model: str

    # --- Relevance Gate（相关性门槛；阈值经 Evaluation v2 校准，见 evaluation/reports/evaluation_v2.md）---
    rerank_relevance_threshold: float

    # --- 切片 ---
    chunk_size: int
    chunk_overlap: int

    # --- 检索 / 会话 ---
    recall_top_n: int
    rerank_top_k: int
    session_ttl_seconds: int
    max_history_messages: int

    # --- 文件上传 ---
    max_upload_file_size_mb: int


@lru_cache
def get_settings() -> Settings:
    """返回进程级单例配置（惰性初始化并缓存）。"""
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        redis_index_name=os.getenv("REDIS_INDEX_NAME", "rag:index"),
        chunk_prefix=os.getenv("REDIS_CHUNK_PREFIX", "chunk:"),
        session_prefix=os.getenv("REDIS_SESSION_PREFIX", "session:"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        embedding_dim=_get_int("EMBEDDING_DIM", 512),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        rerank_relevance_threshold=_get_float("RERANK_RELEVANCE_THRESHOLD", 0.8),
        chunk_size=_get_int("CHUNK_SIZE", 600),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 100),
        recall_top_n=_get_int("RECALL_TOP_N", 10),
        rerank_top_k=_get_int("RERANK_TOP_K", 3),
        session_ttl_seconds=_get_int("SESSION_TTL_SECONDS", 7200),
        max_history_messages=_get_int("MAX_HISTORY_MESSAGES", 10),
        max_upload_file_size_mb=_get_int("MAX_UPLOAD_FILE_SIZE_MB", 5),
    )


# 便捷单例：`from app.config import settings`
settings = get_settings()
