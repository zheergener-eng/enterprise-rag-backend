"""统一错误码与 API 异常。

提供稳定的错误码常量与携带 ``code`` 的 HTTP 异常，供各路由抛错时使用；
统一错误响应体渲染在 error_handlers.py 中完成。
"""
from __future__ import annotations

from fastapi import HTTPException


class ErrorCode:
    """稳定错误码（供客户端程序判断，避免散落魔法字符串）。

    - VALIDATION_ERROR: Pydantic / FastAPI 请求体校验失败（422）。
    - EMPTY_DOCUMENT: 文档内容切分后无片段（400）。
    - MISSING_FILENAME: 上传缺少文件名（400）。
    - EMPTY_FILE: 上传文件内容为空（400）。
    - INVALID_ENCODING: 上传内容非 UTF-8（400）。
    - UNSUPPORTED_FILE_TYPE: 不支持的文件扩展名（415）。
    - FILE_TOO_LARGE: 上传超过大小上限（413）。
    - SERVICE_UNAVAILABLE: Redis 等依赖不可用（503）。
    - INTERNAL_ERROR: 未预期内部异常（500）。
    - LLM_ERROR: LLM 调用失败（仅 SSE error event 使用）。
    """

    VALIDATION_ERROR = "VALIDATION_ERROR"
    EMPTY_DOCUMENT = "EMPTY_DOCUMENT"
    MISSING_FILENAME = "MISSING_FILENAME"
    EMPTY_FILE = "EMPTY_FILE"
    INVALID_ENCODING = "INVALID_ENCODING"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    LLM_ERROR = "LLM_ERROR"


class APIError(HTTPException):
    """携带稳定 ``code`` 的 HTTP 异常。

    相比直接使用 HTTPException，多了 ``code`` 用于渲染统一错误体中的 code 字段。
    用法：``raise APIError(415, ErrorCode.UNSUPPORTED_FILE_TYPE, "Only .md and .txt files are supported.")``。
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.code = code
        self.message = message
