"""文档导入接口。

POST /api/v1/document/import —— 提交文档 JSON（content）→ 切分 → 向量化 → 持久化。
POST /api/v1/document/upload —— 上传 .md/.txt 文件 → 校验 → 读取 → 复用同一导入链路。
"""
from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, File, Form, UploadFile

from app.api.errors import APIError, ErrorCode
from app.config import settings
from app.models.document import DocumentImportRequest, DocumentImportResponse
from app.services.document_import import EmptyDocumentError, import_document_content

router = APIRouter(tags=["document"])

# 支持上传的扩展名（小写、含点）
_ALLOWED_EXTENSIONS = {".md", ".txt"}


@router.post("/document/import", response_model=DocumentImportResponse)
async def import_document(payload: DocumentImportRequest) -> DocumentImportResponse:
    """提交文档 JSON 内容并完成：切分 → 向量化 → Redis 持久化。"""
    try:
        return import_document_content(
            payload.content,
            document_id=payload.document_id,
            title=payload.title or "",
        )
    except EmptyDocumentError as exc:
        raise APIError(400, ErrorCode.EMPTY_DOCUMENT, str(exc)) from exc


@router.post("/document/upload", response_model=DocumentImportResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(""),
) -> DocumentImportResponse:
    """上传 .md / .txt 文件并导入知识库。

    流程：filename 校验 → 扩展名校验 → 读取原始字节 → 大小校验 → UTF-8 解码
    （兼容 BOM）→ 空内容校验 → 生成 UUID document_id → 调用
    ``import_document_content``（复用 document/import 的切分 / 向量化 / 写入链路）。

    不将上传文件持久化到服务器磁盘，仅在内存读取文本后交给导入服务。
    """
    # 1. filename 非空
    filename = (file.filename or "").strip()
    if not filename:
        raise APIError(400, ErrorCode.MISSING_FILENAME, "Filename is required.")

    # 2. 扩展名校验（仅 .md / .txt；MIME 不可靠，以扩展名为准）
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise APIError(
            415,
            ErrorCode.UNSUPPORTED_FILE_TYPE,
            "Only .md and .txt files are supported.",
        )

    # 3. 读取原始字节（内存中，不写磁盘）
    raw = await file.read()

    # 4. 大小校验
    max_bytes = settings.max_upload_file_size_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise APIError(
            413,
            ErrorCode.FILE_TOO_LARGE,
            f"File exceeds maximum upload size of {settings.max_upload_file_size_mb} MB.",
        )

    # 5. UTF-8 解码（utf-8-sig 兼容 BOM）
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise APIError(
            400, ErrorCode.INVALID_ENCODING, "File content must be valid UTF-8 text."
        ) from exc

    # 6. 空 / 纯空白内容拒绝
    if not text.strip():
        raise APIError(400, ErrorCode.EMPTY_FILE, "File content is empty.")

    # 7. 生成 document_id（UUID）；title 缺省用 filename stem
    document_id = uuid.uuid4().hex
    resolved_title = title.strip() or os.path.splitext(filename)[0]

    try:
        return import_document_content(
            text,
            document_id=document_id,
            title=resolved_title,
            filename=filename,
        )
    except EmptyDocumentError as exc:
        raise APIError(400, ErrorCode.EMPTY_DOCUMENT, str(exc)) from exc
