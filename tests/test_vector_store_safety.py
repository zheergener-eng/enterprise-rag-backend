"""VectorStore 安全性单元测试（不依赖真实 Redis）。

覆盖 Stage 4 引入的三类安全加固：
1. 向量维度应用层提前校验（写入 / 检索），避免维度错误拖到 Redis 才暴露；
2. document_id 含 glob 特殊字符时，delete / get 的 SCAN 匹配模式被转义，
   不会误匹配其它文档；
3. 真实 Redis 集成测试使用的 index / prefix 与生产默认隔离，绝不含 rag:index / chunk:。
"""
from __future__ import annotations

import pytest

from app.services.vector_store import VectorDimensionError, VectorStore, _escape_glob


# ---------------------------------------------------------------------------
# 1. 向量维度提前校验（在触碰 Redis 之前即失败）
# ---------------------------------------------------------------------------

def test_add_document_wrong_dim_raises():
    """写入维度不符：在 delete_document / Redis 命令前抛出 VectorDimensionError。"""
    store = VectorStore(redis_url="redis://localhost:6379", index_name="t", dim=512, prefix="test:chunk:")
    with pytest.raises(VectorDimensionError):
        store.add_document("doc", ["text"], [[0.1] * 10])  # 10 != 512


def test_add_document_any_wrong_vector_raises():
    """批量向量中任一维度不符即整体拒绝（校验发生在写前，避免半写状态）。"""
    store = VectorStore(redis_url="redis://localhost:6379", index_name="t", dim=512, prefix="test:chunk:")
    vectors = [[0.1] * 512, [0.1] * 128, [0.1] * 512]
    with pytest.raises(VectorDimensionError) as exc_info:
        store.add_document("doc", ["a", "b", "c"], vectors)
    assert "index 1" in str(exc_info.value)


def test_search_wrong_dim_raises():
    """查询向量维度不符：抛出 VectorDimensionError（不发起 KNN 查询）。"""
    store = VectorStore(redis_url="redis://localhost:6379", index_name="t", dim=512, prefix="test:chunk:")
    with pytest.raises(VectorDimensionError):
        store.search([0.1] * 10, top_k=3)


# ---------------------------------------------------------------------------
# 2. document_id 含 glob 特殊字符时的 SCAN 模式转义
# ---------------------------------------------------------------------------

def test_escape_glob_special_chars():
    """glob 特殊字符（* ? [ ] \\）被逐字符转义为字面量。"""
    assert _escape_glob("doc*star?[0]") == "doc\\*star\\?\\[0\\]"
    assert _escape_glob("a\\b") == "a\\\\b"
    assert _escape_glob("normal_id") == "normal_id"
    assert _escape_glob("") == ""


class _FakeScanClient:
    """捕获 scan 的 match 参数，返回空结果（不真正访问 Redis）。"""

    def __init__(self):
        self.scan_patterns: list[str] = []

    def scan(self, cursor, match=None, count=100):
        self.scan_patterns.append(match)
        return 0, []

    def delete(self, *keys):
        return len(keys)

    def hgetall(self, key):
        return {}


def _install_fake_redis(monkeypatch) -> _FakeScanClient:
    client = _FakeScanClient()

    class _FakeRedis:
        @classmethod
        def from_url(cls, url, **kwargs):
            return client

    monkeypatch.setattr("app.services.vector_store.Redis", _FakeRedis)
    return client


def test_delete_document_escapes_glob_pattern(monkeypatch):
    """含通配符的 document_id 在 delete 时被转义，SCAN 只匹配该文档字面 key。"""
    client = _install_fake_redis(monkeypatch)
    store = VectorStore(redis_url="redis://x", index_name="t", dim=512, prefix="test:chunk:")
    store.delete_document("doc*star")
    assert client.scan_patterns == ["test:chunk:doc\\*star:*"]


def test_get_document_chunks_escapes_glob_pattern(monkeypatch):
    """含通配符的 document_id 在 get 时同样被转义。"""
    client = _install_fake_redis(monkeypatch)
    store = VectorStore(redis_url="redis://x", index_name="t", dim=512, prefix="test:chunk:")
    store.get_document_chunks("doc?[x]")
    assert client.scan_patterns == ["test:chunk:doc\\?\\[x\\]:*"]


# ---------------------------------------------------------------------------
# 3. 集成测试 index / prefix 与生产默认隔离
# ---------------------------------------------------------------------------

def test_test_indexes_isolated_from_production():
    """真实 Redis 集成测试使用独立命名空间，绝不含生产默认 rag:index / chunk:。"""
    from tests.test_retrieval import RETRIEVAL_INDEX, RETRIEVAL_PREFIX
    from tests.test_vector_store import TEST_INDEX, TEST_PREFIX

    assert TEST_INDEX != "rag:index"
    assert RETRIEVAL_INDEX != "rag:index"
    assert TEST_PREFIX != "chunk:"
    assert RETRIEVAL_PREFIX != "chunk:"
