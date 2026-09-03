"""POST /api/v1/document/import 接口测试（mock 单元测试）。

采用假 embedder / 假 vector store 验证接口行为，不依赖真实 Redis 与真实模型；
真实 Redis 集成见 test_vector_store.py。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


def _import(client: TestClient, document_id: str = "doc-001", content: str = ""):
    return client.post(
        "/api/v1/document/import",
        json={"document_id": document_id, "content": content},
    )


class FakeEmbedder:
    """假向量化：返回固定维度（512）的全零向量。"""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed_documents(self, chunks: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in chunks]


class FakeStore:
    """假向量存储：记录写入调用，不做真实持久化。"""

    def __init__(self):
        self.calls: list[dict] = []

    def ensure_index(self) -> None:
        pass

    def add_document(self, document_id, chunks, vectors, title="", filename="") -> int:
        self.calls.append(
            {
                "document_id": document_id,
                "chunks": chunks,
                "vectors": vectors,
                "title": title,
                "filename": filename,
            }
        )
        return len(chunks)


@pytest.fixture
def fake_services(monkeypatch):
    """将共享导入服务中的 get_embedder / get_vector_store 替换为假实现。"""
    embedder = FakeEmbedder()
    store = FakeStore()
    monkeypatch.setattr(
        "app.services.document_import.get_embedder", lambda: embedder
    )
    monkeypatch.setattr(
        "app.services.document_import.get_vector_store", lambda: store
    )
    return embedder, store


def test_import_empty_content_rejected():
    """空 content：应被拒绝（Pydantic 校验 422，不触发后续逻辑）。"""
    with TestClient(app) as client:
        resp_empty = _import(client, content="")
        resp_blank = _import(client, content="   \n\t  ")
    assert resp_empty.status_code == 422
    assert resp_blank.status_code == 422


def test_import_no_chunks_rejected():
    """仅含标题、无正文：切分为空，返回 400。"""
    content = "# 只有标题\n## 没有正文"
    with TestClient(app) as client:
        resp = _import(client, content=content)
    assert resp.status_code == 400


def test_import_full_pipeline(fake_services):
    """正常文档：应完成 chunking → embedding → store 并返回结果。"""
    _, store = fake_services
    content = "这是接口测试的一段较长的文本内容。\n" * 200

    with TestClient(app) as client:
        resp = _import(client, document_id="doc-001", content=content)

    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == "doc-001"
    assert data["status"] == "ok"
    assert data["chunk_count"] > 1

    # 响应不应包含完整 embedding 向量
    assert "embedding" not in data
    assert "vectors" not in data

    # 验证 store 被正确调用一次，且 chunks/vectors 数量一致
    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["document_id"] == "doc-001"
    assert len(call["chunks"]) == data["chunk_count"]
    assert len(call["vectors"]) == data["chunk_count"]

    # 预览结构正确
    assert len(data["chunks"]) == data["chunk_count"]
    for i, chunk in enumerate(data["chunks"]):
        assert chunk["index"] == i
        assert chunk["length"] > 0
        assert 0 < len(chunk["preview"]) <= 100
