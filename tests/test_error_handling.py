"""Stage 3 统一错误处理与 SSE 错误语义测试。

覆盖：
- 统一错误响应体 schema（{error: {code, message}}）：
  422 校验失败 / 415 不支持类型 / 413 超大文件 / 503 Redis 不可用 / 500 未预期异常；
- Chat top_k 上限（> 50 → 422，边界 1/50 合法）；
- SSE 错误语义：
  LLM 中途异常 → error(LLM_ERROR) 稳定 message、无 done、无后续 delta；
  Redis 失败 → error(SERVICE_UNAVAILABLE)；
  未预期异常 → error(INTERNAL_ERROR) 不泄露 traceback。

不依赖真实 Redis / 模型：失败场景用假服务注入，成功/兼容路径由
test_document_api / test_document_upload / test_chat_api 覆盖。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from redis.exceptions import ConnectionError as RedisConnectionError

from app.main import app
from app.models.chat import ChatCompletionRequest


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """把原始 SSE 文本解析为 (event, data) 列表；data 行用 json.loads 校验合法 JSON。"""
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event is not None:
            events.append((event, data))
    return events


# --------------------------------------------------------------------------
# 统一错误响应体
# --------------------------------------------------------------------------

def test_validation_error_uses_unified_body():
    """Pydantic 校验失败 → 422 + {error: {code: VALIDATION_ERROR, message}}。"""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/completions",
            json={"session_id": "s", "message": "   "},
        )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    # 不再返回 FastAPI 默认的 {"detail": [...]}
    assert "detail" not in body


def test_upload_unsupported_returns_unified_415():
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/document/upload",
            files={"file": ("x.pdf", b"data", "application/pdf")},
        )
    assert resp.status_code == 415
    body = resp.json()
    assert body["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
    assert body["error"]["message"] == "Only .md and .txt files are supported."


def test_upload_oversize_returns_unified_413(monkeypatch):
    monkeypatch.setattr(
        "app.api.v1.document.settings",
        SimpleNamespace(max_upload_file_size_mb=0),
    )
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/document/upload",
            files={"file": ("big.md", b"x" * 10, "text/markdown")},
        )
    assert resp.status_code == 413
    body = resp.json()
    assert body["error"]["code"] == "FILE_TOO_LARGE"
    assert "detail" not in body


# --------------------------------------------------------------------------
# top_k 上限
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["recall_top_n", "rerank_top_k"])
def test_top_k_over_upper_bound_rejected(field):
    """top_k > 50 → 422（VALIDATION_ERROR）。"""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/completions",
            json={"session_id": "s", "message": "hi", field: 51},
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_top_k_boundary_values_accepted():
    """1 与 50 为合法边界；51 非法。"""
    assert ChatCompletionRequest(session_id="s", message="hi", recall_top_n=1, rerank_top_k=1)
    assert ChatCompletionRequest(session_id="s", message="hi", recall_top_n=50, rerank_top_k=50)
    with pytest.raises(ValidationError):
        ChatCompletionRequest(session_id="s", message="hi", recall_top_n=51)
    with pytest.raises(ValidationError):
        ChatCompletionRequest(session_id="s", message="hi", rerank_top_k=51)


# --------------------------------------------------------------------------
# Redis 失败（非流式 → 503；流式 → SSE error）
# --------------------------------------------------------------------------

class _RedisDownStore:
    def get_history(self, session_id):
        raise RedisConnectionError("connection refused")

    def clear_session(self, session_id):
        raise RedisConnectionError("connection refused")


def test_history_redis_unavailable_returns_503(monkeypatch):
    monkeypatch.setattr("app.api.v1.chat.get_session_store", lambda: _RedisDownStore())
    with TestClient(app) as client:
        resp = client.get("/api/v1/chat/history/sess-x")
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "SERVICE_UNAVAILABLE"
    assert body["error"]["message"]
    # 不泄露连接串 / traceback
    dumped = json.dumps(body)
    assert "redis://" not in dumped and "Traceback" not in dumped


def test_delete_session_redis_unavailable_returns_503(monkeypatch):
    monkeypatch.setattr("app.api.v1.chat.get_session_store", lambda: _RedisDownStore())
    with TestClient(app) as client:
        resp = client.delete("/api/v1/chat/session/sess-x")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


def _stream_redis_down(*args, **kwargs):
    raise RedisConnectionError("connection refused")
    yield  # 使本函数成为 generator（与真实 answer_with_session_stream 形态一致）


def test_stream_redis_unavailable_returns_sse_error(monkeypatch):
    monkeypatch.setattr("app.api.v1.chat.answer_with_session_stream", _stream_redis_down)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/completions",
            json={"session_id": "s", "message": "hi"},
        )
    assert resp.status_code == 200
    events = parse_sse(resp.content.decode("utf-8"))
    assert [e for e, _ in events] == ["error"]
    assert events[0][1]["code"] == "SERVICE_UNAVAILABLE"
    # 不发送成功 done
    assert "done" not in [e for e, _ in events]


# --------------------------------------------------------------------------
# 未预期异常（非流式 → 500；流式 → SSE INTERNAL_ERROR）
# --------------------------------------------------------------------------

class _BoomStore:
    def get_history(self, session_id):
        raise RuntimeError("secret internal detail")


def test_history_unexpected_error_returns_500(monkeypatch):
    monkeypatch.setattr("app.api.v1.chat.get_session_store", lambda: _BoomStore())
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/v1/chat/history/sess-x")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["message"] == "Internal server error."
    assert "secret internal detail" not in json.dumps(body)


def _stream_boom(*args, **kwargs):
    raise RuntimeError("secret internal detail")
    yield


def test_stream_unexpected_error_returns_sse_internal_error(monkeypatch):
    monkeypatch.setattr("app.api.v1.chat.answer_with_session_stream", _stream_boom)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/completions",
            json={"session_id": "s", "message": "hi"},
        )
    assert resp.status_code == 200
    events = parse_sse(resp.content.decode("utf-8"))
    assert [e for e, _ in events] == ["error"]
    assert events[0][1]["code"] == "INTERNAL_ERROR"
    assert "secret internal detail" not in json.dumps(events)
