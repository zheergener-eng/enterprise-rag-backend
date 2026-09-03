"""Redis Vector Store 集成测试（需要真实 Redis Stack 运行）。

与 test_document_api.py（mock 单元测试）区分，本文件使用真实 Redis，
验证连接、索引、写入、读取、维度、完整链路与重复导入策略。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.services.chunking import split_document
from app.services.vector_store import VectorStore


TEST_INDEX = "rag:test:index"
TEST_PREFIX = "testchunk:"
DIM = settings.embedding_dim  # 512

# 本文件全部依赖真实 Redis Stack，统一打 integration 标记（Redis 不可用时自动 skip）
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def store():
    """module 级：整个测试模块共用一个真实 Redis 向量存储（独立索引/前缀）。"""
    s = VectorStore(
        redis_url=settings.redis_url,
        index_name=TEST_INDEX,
        dim=DIM,
        prefix=TEST_PREFIX,
    )
    s.drop_index()  # 清理可能残留的旧索引
    s.ensure_index()
    yield s
    s.drop_index()  # 测试结束清理


def test_redis_connection():
    """Redis 连接正常。"""
    from redis import Redis

    r = Redis.from_url(settings.redis_url)
    assert r.ping() is True


def test_index_creation_idempotent(store):
    """Vector Index 可以创建，且重复创建幂等（不报错）。"""
    store.ensure_index()
    store.ensure_index()
    store.ensure_index()
    assert store.index_exists() is True


def test_add_and_read_chunk(store):
    """chunk + metadata + embedding 可以写入并读取。"""
    vec = np.random.default_rng(0).random(DIM).astype(np.float32).tolist()
    n = store.add_document("doc-test-1", ["这是一段测试文本。"], [vec], title="测试文档")
    assert n == 1

    chunks = store.get_document_chunks("doc-test-1")
    assert len(chunks) == 1
    c = chunks[0]
    assert c["document_id"] == "doc-test-1"
    assert c["chunk_id"] == "doc-test-1:0"
    assert c["index"] == 0
    assert c["text"] == "这是一段测试文本。"
    assert c["title"] == "测试文档"


def test_embedding_dimension_correct(store):
    """写入/读取的 embedding 维度正确（float32，DIM 维）。"""
    vec = np.random.default_rng(1).random(DIM).astype(np.float32).tolist()
    store.add_document("doc-dim", ["维度测试"], [vec])
    c = store.get_document_chunks("doc-dim")[0]
    assert c["embedding"].shape == (DIM,)
    assert c["embedding"].dtype == np.float32


def test_full_pipeline_chunk_embed_store(store, real_embedder):
    """完整链路：chunking → embedding → Redis（真实组件）。"""
    content = "这是完整链路测试的文档内容。\n" * 50
    chunks = split_document(
        content, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    chunk_texts = [c.text for c in chunks]
    vectors = real_embedder.embed_documents(chunk_texts)

    n = store.add_document("doc-pipeline", chunk_texts, vectors, title="链路测试")
    assert n == len(chunk_texts)

    stored = store.get_document_chunks("doc-pipeline")
    assert len(stored) == len(chunk_texts)
    assert stored[0]["embedding"].shape == (DIM,)


def test_reimport_no_duplicates(store, real_embedder):
    """同一 document_id 重复导入不会无限产生重复数据。"""
    content = "重复导入测试内容。\n" * 30
    chunks = split_document(
        content, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    chunk_texts = [c.text for c in chunks]
    vectors = real_embedder.embed_documents(chunk_texts)

    n1 = store.add_document("doc-reimport", chunk_texts, vectors, title="重复导入")
    n2 = store.add_document("doc-reimport", chunk_texts, vectors, title="重复导入")

    assert n1 == n2
    stored = store.get_document_chunks("doc-reimport")
    assert len(stored) == n2  # 等于最后一次写入数量，而非累加


def test_reimport_overwrites_changed_content(store, real_embedder):
    """重复导入时，旧 chunk 被新 chunk 覆盖（内容变化时数据量随之更新）。"""
    doc_id = "doc-overwrite"
    # 第一次：3 个 chunk
    v1 = real_embedder.embed_documents(["旧内容片段。"] * 3)
    store.add_document(doc_id, ["旧内容片段。"] * 3, v1, title="覆盖测试")
    assert len(store.get_document_chunks(doc_id)) == 3
    # 第二次：5 个 chunk（覆盖后应为 5，而非 3 + 5）
    v2 = real_embedder.embed_documents(["新内容片段。"] * 5)
    store.add_document(doc_id, ["新内容片段。"] * 5, v2, title="覆盖测试")
    assert len(store.get_document_chunks(doc_id)) == 5
