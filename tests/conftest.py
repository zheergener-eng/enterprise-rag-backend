"""pytest 公共 fixtures。

此外集中处理「外部依赖（Redis Stack）未启动」与「代码测试失败」的区分：
- 真实 Redis 集成测试统一打 ``@pytest.mark.integration`` 标记；
- 收集阶段探测一次 Redis 是否可达，不可达时自动 skip 集成测试（skip reason 明确
  写明是外部依赖未启动），unit tests 照常运行，避免把环境缺失误报为代码失败。

当 Redis 可用时，集成测试正常执行；要只跑集成测试：``pytest -m integration``。
"""
from __future__ import annotations

import pytest

from app.config import get_settings


def _redis_reachable() -> bool:
    """探测配置的 Redis 是否可达（一次，短超时）。"""
    try:
        from redis import Redis

        Redis.from_url(
            get_settings().redis_url, socket_connect_timeout=1, socket_timeout=1
        ).ping()
        return True
    except Exception:
        return False


# 收集阶段只探测一次（避免每个测试都去连一次 Redis）
REDIS_AVAILABLE = _redis_reachable()


def pytest_configure(config):
    """注册 integration 标记，避免未注册标记触发 pytest warning。"""
    config.addinivalue_line(
        "markers",
        "integration: 需要真实 Redis Stack 运行（外部依赖）",
    )


def pytest_collection_modifyitems(config, items):
    """Redis 不可用时跳过集成测试，unit tests 照常运行。

    只 skip 打了 ``@pytest.mark.integration`` 的测试；普通 unit test（mock 实现）
    不依赖外部 Redis，仍会执行，从而把「外部依赖未启动」与「代码失败」分开。
    """
    if REDIS_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="Redis Stack 不可用（外部依赖未启动，非代码失败）")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)


@pytest.fixture
def settings():
    """返回应用配置。"""
    return get_settings()


@pytest.fixture(scope="session")
def real_embedder():
    """session 级：整个测试会话只下载/加载一次真实 embedding 模型。

    仅供集成测试使用（真实向量写入 / 检索）；unit tests 用假 embedder，
    不会触发本 fixture，因此 Redis 不可用跳过集成测试时也不会下载模型。
    """
    from app.services.embeddings import Embedder

    return Embedder(model_name=get_settings().embedding_model)
