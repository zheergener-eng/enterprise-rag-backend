"""统一异常处理器。

把「业务 APIError / Pydantic 校验失败 / Redis 不可用 / 未预期异常」映射为
统一错误响应体 ``{"error": {"code": ..., "message": ...}}``，并保证：
- 客户端不看到 traceback、连接串、密钥、内部路径或模型原始响应；
- 服务端通过 logger 记录 error code / endpoint / exception type / 必要上下文。

流式（SSE）错误不在此处理（HTTP 响应头已发出，只能走 SSE error event），
见 app/api/v1/chat.py 的 _sse_stream。
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from app.api.errors import APIError, ErrorCode
from app.models.errors import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


def _json_error(status_code: int, code: str, message: str) -> JSONResponse:
    """构造统一错误 JSONResponse。"""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=ErrorDetail(code=code, message=message)).model_dump(),
    )


def _first_validation_message(errors: list[dict]) -> str:
    """从 Pydantic 校验错误中提取首条人类可读描述（带字段定位，不含输入原文）。"""
    if not errors:
        return "Invalid request body"
    err = errors[0]
    loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
    msg = err.get("msg", "Invalid request body")
    return f"{loc}: {msg}" if loc else msg


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """业务 APIError → 统一错误体（4xx / 5xx）。"""
    return _json_error(exc.status_code, exc.code, exc.message)


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Pydantic 请求校验失败 → 422 统一错误体（code=VALIDATION_ERROR）。

    详细校验信息只记录服务端日志（仅消息文本，不含输入的原始值），
    客户端只拿到稳定 message。
    """
    errors = exc.errors()
    logger.warning(
        "Request validation failed: %s",
        [e.get("msg") for e in errors],
    )
    return _json_error(422, ErrorCode.VALIDATION_ERROR, _first_validation_message(errors))


async def redis_error_handler(request: Request, exc: RedisError) -> JSONResponse:
    """Redis 等依赖不可用 → 503（记录 exception type，不泄露连接串 / traceback）。"""
    logger.error(
        "Redis dependency unavailable on %s %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return _json_error(
        503,
        ErrorCode.SERVICE_UNAVAILABLE,
        "Redis is temporarily unavailable.",
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """未预期内部异常 → 500（记录完整异常到服务端日志，客户端只看到稳定消息）。"""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _json_error(500, ErrorCode.INTERNAL_ERROR, "Internal server error.")


def register_exception_handlers(app: FastAPI) -> None:
    """注册所有统一异常处理器。"""
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(RedisError, redis_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
