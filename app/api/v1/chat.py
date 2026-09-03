"""对话相关接口。

POST   /api/v1/chat/completions   —— 发起对话（SSE 流式返回）
GET    /api/v1/chat/history/{id}  —— 查看会话上下文记录
DELETE /api/v1/chat/session/{id}  —— 重置/清理会话

chat/completions 只负责：请求校验 → 调用 services.rag 的多轮流式 RAG → 封装 SSE →
返回 StreamingResponse。不在此层实现 Retrieval / Query Rewrite / DeepSeek / Session 写回。
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from redis.exceptions import RedisError

from app.api.errors import ErrorCode
from app.api.v1.sse import encode_sse
from app.models.chat import (
    ChatCompletionRequest,
    ChatHistoryResponse,
    ChatMessage,
    ClearSessionResponse,
)
from app.services.llm import LLMError
from app.services.rag import answer_with_session_stream
from app.services.session import get_session_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# SSE error event 使用的稳定、非泄露消息（不含 traceback / 异常原文 / 连接串 / 密钥）。
_LLM_ERROR_MESSAGE = "Failed to generate response."
_SERVICE_UNAVAILABLE_MESSAGE = "Redis is temporarily unavailable."
_INTERNAL_ERROR_MESSAGE = "Internal server error."


def _sse_stream(
    session_id: str,
    message: str,
    recall_top_n: int | None,
    rerank_top_k: int | None,
) -> Iterator[str]:
    """把多轮流式 RAG 的结果封装为 SSE event 序列（delta / done / error）。

    - 每个文本增量 → ``event: delta``；
    - 完整回答成功结束 → ``event: done``（只发一次）；
    - 流开始后失败 → ``event: error``（LLMError→LLM_ERROR；RedisError→SERVICE_UNAVAILABLE；
      其它未预期异常→INTERNAL_ERROR），随后结束流，不发 done。

    失败语义：HTTP 响应头一旦发出便无法改成 5xx，因此本函数内的任何失败都只能通过
    SSE error event 通知客户端，且 message 使用稳定非泄露文本（真实异常只落服务端日志）。
    流开始前的失败仅来自请求体校验（Pydantic 422，见 models/chat.py），在进入本函数前已返回。

    Session 写回由 services.rag 的 answer_with_session_stream 内部完成，
    本函数只做 SSE 封装，不重复写 Session，也不实现检索 / Rerank 逻辑。
    """
    try:
        for chunk in answer_with_session_stream(
            session_id, message, recall_top_n=recall_top_n, rerank_top_k=rerank_top_k
        ):
            yield encode_sse("delta", {"content": chunk})
    except LLMError:
        logger.exception("LLM error during streaming (session=%s)", session_id)
        yield encode_sse("error", {"code": ErrorCode.LLM_ERROR, "message": _LLM_ERROR_MESSAGE})
        return
    except RedisError as exc:
        logger.error(
            "Redis unavailable during streaming (session=%s): %s",
            session_id,
            type(exc).__name__,
        )
        yield encode_sse(
            "error", {"code": ErrorCode.SERVICE_UNAVAILABLE, "message": _SERVICE_UNAVAILABLE_MESSAGE}
        )
        return
    except Exception:
        logger.exception("Unexpected error during streaming (session=%s)", session_id)
        yield encode_sse(
            "error", {"code": ErrorCode.INTERNAL_ERROR, "message": _INTERNAL_ERROR_MESSAGE}
        )
        return
    # 仅当流正常结束才发送 done
    yield encode_sse("done", {"session_id": session_id})


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest) -> StreamingResponse:
    """发起对话，以 SSE 流式返回回答。

    请求体校验（session_id / message 非空、recall_top_n / rerank_top_k 范围及
    rerank_top_k <= recall_top_n）由 Pydantic 在进入本函数前完成，校验失败返回 422，
    不会建立 SSE。业务逻辑全部复用 services.rag 的多轮流式 RAG（两阶段检索）。
    """
    return StreamingResponse(
        _sse_stream(
            payload.session_id,
            payload.message,
            payload.recall_top_n,
            payload.rerank_top_k,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/history/{session_id}", response_model=ChatHistoryResponse)
async def chat_history(session_id: str) -> ChatHistoryResponse:
    """查看指定会话的完整上下文记录（按时间顺序）。

    会话不存在时返回空 messages 列表（status 200）。
    """
    messages = [
        ChatMessage(role=m["role"], content=m["content"], ts=m.get("ts"))
        for m in get_session_store().get_history(session_id)
    ]
    return ChatHistoryResponse(session_id=session_id, messages=messages)


@router.delete("/chat/session/{session_id}", response_model=ClearSessionResponse)
async def clear_session(session_id: str) -> ClearSessionResponse:
    """重置/清理指定会话（幂等）。

    - 会话存在：删除其历史，cleared = true；
    - 会话不存在：无操作，cleared = false（仍返回 200，不视为错误）。
    """
    deleted = get_session_store().clear_session(session_id)
    return ClearSessionResponse(session_id=session_id, cleared=bool(deleted))
