"""集中式配置（app.config）单元测试。

覆盖 Relevance Gate 阈值相关行为：
- 默认阈值 0.8（Evaluation v2 选定）；
- 环境变量 RERANK_RELEVANCE_THRESHOLD 覆盖默认值；
- 非法数值抛 ValueError。
"""
from __future__ import annotations

import pytest

from app import config


def test_default_rerank_threshold_is_0_8(monkeypatch):
    """未设置环境变量时，默认阈值为 0.8。"""
    monkeypatch.delenv("RERANK_RELEVANCE_THRESHOLD", raising=False)
    config.get_settings.cache_clear()
    assert config.get_settings().rerank_relevance_threshold == 0.8


def test_env_override_rerank_threshold(monkeypatch):
    """RERANK_RELEVANCE_THRESHOLD=0.6 覆盖默认值 0.8。"""
    monkeypatch.setenv("RERANK_RELEVANCE_THRESHOLD", "0.6")
    config.get_settings.cache_clear()
    assert config.get_settings().rerank_relevance_threshold == 0.6


def test_invalid_rerank_threshold_raises(monkeypatch):
    """非法数值（非数字）→ ValueError（带变量名）。"""
    monkeypatch.setenv("RERANK_RELEVANCE_THRESHOLD", "abc")
    config.get_settings.cache_clear()
    with pytest.raises(ValueError):
        config.get_settings()


def test_default_redis_index_name(monkeypatch):
    """未设置环境变量时，生产默认索引为 rag:index。"""
    monkeypatch.delenv("REDIS_INDEX_NAME", raising=False)
    config.get_settings.cache_clear()
    assert config.get_settings().redis_index_name == "rag:index"


def test_default_chunk_prefix(monkeypatch):
    """未设置环境变量时，生产默认 chunk prefix 为 chunk:。"""
    monkeypatch.delenv("REDIS_CHUNK_PREFIX", raising=False)
    config.get_settings.cache_clear()
    assert config.get_settings().chunk_prefix == "chunk:"


def test_default_session_prefix(monkeypatch):
    """未设置环境变量时，生产默认 session prefix 为 session:。"""
    monkeypatch.delenv("REDIS_SESSION_PREFIX", raising=False)
    config.get_settings.cache_clear()
    assert config.get_settings().session_prefix == "session:"


def test_env_override_chunk_prefix(monkeypatch):
    """REDIS_CHUNK_PREFIX 可覆盖默认 chunk prefix。"""
    monkeypatch.setenv("REDIS_CHUNK_PREFIX", "test:chunk:")
    config.get_settings.cache_clear()
    assert config.get_settings().chunk_prefix == "test:chunk:"


def test_env_override_session_prefix(monkeypatch):
    """REDIS_SESSION_PREFIX 可覆盖默认 session prefix（测试 / Evaluation 隔离）。"""
    monkeypatch.setenv("REDIS_SESSION_PREFIX", "test:session:")
    config.get_settings.cache_clear()
    assert config.get_settings().session_prefix == "test:session:"
