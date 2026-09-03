"""POST /api/v1/document/upload 接口测试（mock 单元测试）。

覆盖文件上传的校验路径与复用导入链路：
- .md / .txt 正常上传；
- 空文件 / 纯空白文件 / 空 filename / 不支持扩展名 / 超大文件 / 非 UTF-8 内容
  均返回明确 4xx，且不触发真实模型与 Redis。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app


def _upload(
    client: TestClient,
    filename: str,
    content: bytes,
    content_type: str = "application/octet-stream",
    title: str | None = None,
):
    data = {} if title is None else {"title": title}
    return client.post(
        "/api/v1/document/upload",
        files={"file": (filename, content, content_type)},
        data=data,
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


# 一份能切出多个 chunk 的 Markdown（3 个不同标题 → 3 段 → 不合并）
_MARKDOWN = (
    "# 备份策略\n\n星河项目数据库全量备份保留 14 天。\n\n"
    "# 恢复策略\n\n增量备份保留 7 天，恢复按变更窗口执行。\n\n"
    "# 审批流程\n\n备份操作需在生产变更窗口内审批后执行。\n\n"
).encode("utf-8")

_TXT = ("这是一段纯文本知识库内容。星河项目数据库备份文件保留 14 天。\n" * 10).encode("utf-8")


# ---- A. 正常 Markdown 上传 ----
def test_upload_markdown_success(fake_services):
    _, store = fake_services
    with TestClient(app) as client:
        resp = _upload(client, "sample.md", _MARKDOWN, "text/markdown")

    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "sample.md"
    assert data["title"] == "sample"  # 未显式传 title → 用 filename stem
    assert data["chunk_count"] > 0
    assert data["status"] == "ok"

    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["filename"] == "sample.md"
    assert call["title"] == "sample"
    assert len(call["chunks"]) == data["chunk_count"]
    assert len(call["vectors"]) == data["chunk_count"]
    # document_id 为服务生成的 UUID hex（32 位）
    assert len(call["document_id"]) == 32


# ---- 显式 title 覆盖 ----
def test_upload_explicit_title(fake_services):
    _, store = fake_services
    with TestClient(app) as client:
        resp = _upload(client, "sample.md", _MARKDOWN, "text/markdown", title="数据库备份规范")

    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "数据库备份规范"
    assert store.calls[0]["title"] == "数据库备份规范"


# ---- B. 正常 TXT 上传 ----
def test_upload_txt_success(fake_services):
    _, store = fake_services
    with TestClient(app) as client:
        resp = _upload(client, "sample.txt", _TXT, "text/plain")

    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "sample.txt"
    assert data["chunk_count"] > 0
    assert store.calls[0]["filename"] == "sample.txt"


# ---- C. 空文件（0 bytes）----
def test_upload_empty_file_rejected(fake_services):
    _, store = fake_services
    with TestClient(app) as client:
        resp = _upload(client, "empty.md", b"", "text/markdown")

    assert resp.status_code == 400
    assert store.calls == []  # 未进入导入链路


# ---- D. 纯空白文件 ----
def test_upload_blank_file_rejected(fake_services):
    _, store = fake_services
    with TestClient(app) as client:
        resp = _upload(client, "blank.md", "   \n\n".encode("utf-8"), "text/markdown")

    assert resp.status_code == 400
    assert store.calls == []


# ---- 空 filename（纯空白）----
def test_upload_missing_filename_rejected(fake_services):
    _, store = fake_services
    # 纯空白 filename：可被解析为 UploadFile，但应被路由的 filename 校验拒绝
    with TestClient(app) as client:
        resp = _upload(client, "   ", _TXT, "text/plain")

    assert resp.status_code == 400
    assert store.calls == []


# ---- E. 不支持扩展名 ----
@pytest.mark.parametrize("filename", ["sample.pdf", "sample.docx", "sample.csv", "photo.png"])
def test_upload_unsupported_extension_rejected(fake_services, filename):
    _, store = fake_services
    with TestClient(app) as client:
        resp = _upload(client, filename, b"not really", "application/octet-stream")

    assert resp.status_code == 415
    assert store.calls == []


# ---- F. 超大文件（> MAX_UPLOAD_FILE_SIZE）----
def test_upload_oversize_file_rejected(fake_services, monkeypatch):
    _, store = fake_services
    # 将路由读取的配置上限压到 0 MB，使任意非空内容触发 413
    monkeypatch.setattr(
        "app.api.v1.document.settings",
        SimpleNamespace(max_upload_file_size_mb=0),
    )
    with TestClient(app) as client:
        resp = _upload(client, "big.md", _TXT, "text/markdown")

    assert resp.status_code == 413
    assert store.calls == []


# ---- G. 非 UTF-8 / 无法解码内容 ----
def test_upload_non_utf8_rejected(fake_services):
    _, store = fake_services
    invalid = b"\x80\x81\x82\xff\xfe\x00invalid"
    with TestClient(app) as client:
        resp = _upload(client, "bad.md", invalid, "text/markdown")

    assert resp.status_code == 400
    assert store.calls == []
