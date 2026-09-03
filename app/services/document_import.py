"""文档导入统一业务入口（JSON content 与文件上传共用）。

将「切分 → 向量化 → Redis 写入」抽成一个公共函数，避免 JSON Import 与
File Upload 各自维护一套导入逻辑。两条 API 路由只负责各自的请求 / 文件校验，
最终都调用本模块的 `import_document_content`。
"""
from __future__ import annotations

from app.config import settings
from app.models.document import ChunkPreview, DocumentImportResponse
from app.services.chunking import split_document
from app.services.embeddings import get_embedder
from app.services.vector_store import get_vector_store

# 预览截断长度（字符）
_PREVIEW_LEN = 100


class EmptyDocumentError(Exception):
    """文档内容切分后无片段（如仅有标题、无正文）。"""


def import_document_content(
    content: str,
    document_id: str,
    title: str = "",
    filename: str = "",
) -> DocumentImportResponse:
    """统一导入链路：切分 → 向量化 → Redis 持久化。

    Args:
        content: 文档正文（Markdown 或纯文本；调用方需保证非空）。
        document_id: 文档唯一标识（JSON import 由调用方提供；upload 由服务生成 UUID）。
        title: 文档标题（作为元数据写入向量库；为空则写入空字符串）。
        filename: 原始文件名（作为元数据写入向量库；JSON import 可为空）。

    Returns:
        DocumentImportResponse（不含 embedding 向量）。

    Raises:
        EmptyDocumentError: 内容切分后无片段。
    """
    chunks = split_document(
        content, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    if not chunks:
        raise EmptyDocumentError(
            "document content produced no chunks (e.g. only headings, no body)"
        )

    chunk_texts = [c.text for c in chunks]

    # 向量化（与查询侧同一模型）
    vectors = get_embedder().embed_documents(chunk_texts)

    # 持久化到 Redis（覆盖式写入，保证重复导入不产生重复 chunk）
    store = get_vector_store()
    store.ensure_index()
    store.add_document(
        document_id,
        chunk_texts,
        vectors,
        title=title,
        filename=filename,
    )

    # 返回导入结果（不含完整 embedding 向量）
    previews = [
        ChunkPreview(index=c.index, preview=c.text[:_PREVIEW_LEN], length=len(c.text))
        for c in chunks
    ]
    return DocumentImportResponse(
        document_id=document_id,
        filename=filename,
        title=title,
        chunk_count=len(chunks),
        status="ok",
        chunks=previews,
    )
