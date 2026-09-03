"""POST /chat/completions 接口测试（API 契约 + SSE 协议 + Session 行为）。

覆盖：
- 请求校验（session_id / message 非空、top_k 范围、缺字段）→ 422；
- SSE 协议（delta 顺序、data 合法 JSON、单 done、delta 拼接 == 完整 answer、content-type）；
- 流中途失败（delta 后 error、无 done）；
- route 不重复写 Session；
- 端到端 Session 行为（成功写回完整 user/assistant、失败不写 partial、DELETE 清空、隔离）。

SSE 结构测试用假 answer_with_session_stream；端到端 Session 测试用真实
answer_with_session_stream + 假 rewrite/retrieve/stream LLM + 内存 SessionStore。
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.llm import LLMError
from app.services.query_rewrite import RewriteResult
from app.services.reranker import RerankedChunk


# --------------------------------------------------------------------------
# SSE 解析辅助
# --------------------------------------------------------------------------

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


def _post(client: TestClient, payload: dict):
    """POST /chat/completions，返回 (resp, 解析后的 SSE events)。"""
    resp = client.post("/api/v1/chat/completions", json=payload)
    return resp, parse_sse(resp.content.decode("utf-8"))


# --------------------------------------------------------------------------
# 假 answer_with_session_stream（仅用于 SSE 结构 / 职责边界测试）
# --------------------------------------------------------------------------

class FakeStreamingRAG:
    """生成器：yield 预设 chunks，可选中途抛 LLMError，记录调用。"""

    def __init__(self, chunks, error=None):
        self.chunks = list(chunks)
        self.error = error
        self.calls: list[tuple] = []

    def __call__(self, session_id, message, recall_top_n=None, rerank_top_k=None):
        self.calls.append((session_id, message, recall_top_n, rerank_top_k))
        for c in self.chunks:
            yield c
        if self.error is not None:
            raise self.error


@pytest.fixture
def fake_rag_stream(monkeypatch):
    rag = FakeStreamingRAG(["星河", "项目", "每天凌晨 2:30 备份。"])
    monkeypatch.setattr("app.api.v1.chat.answer_with_session_stream", rag)
    return rag


# --------------------------------------------------------------------------
# A. 请求校验（流式开始前错误 → HTTP 4xx，不建立 SSE）
# --------------------------------------------------------------------------

def test_empty_message_rejected():
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/completions",
            json={"session_id": "sess-1", "message": "   "},
        )
    assert resp.status_code == 422


def test_empty_session_id_rejected():
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/completions",
            json={"session_id": "", "message": "你好"},
        )
    assert resp.status_code == 422


@pytest.mark.parametrize("recall_top_n", [0, -1, -100])
def test_recall_top_n_nonpositive_rejected(recall_top_n):
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/completions",
            json={"session_id": "sess-1", "message": "你好", "recall_top_n": recall_top_n},
        )
    assert resp.status_code == 422


@pytest.mark.parametrize("rerank_top_k", [0, -1, -100])
def test_rerank_top_k_nonpositive_rejected(rerank_top_k):
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/completions",
            json={"session_id": "sess-1", "message": "你好", "rerank_top_k": rerank_top_k},
        )
    assert resp.status_code == 422


def test_rerank_top_k_gt_recall_rejected():
    """rerank_top_k > recall_top_n → 422。"""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/completions",
            json={"session_id": "sess-1", "message": "你好", "recall_top_n": 3, "rerank_top_k": 5},
        )
    assert resp.status_code == 422


def test_missing_fields_rejected():
    with TestClient(app) as client:
        resp = client.post("/api/v1/chat/completions", json={})
    assert resp.status_code == 422


# --------------------------------------------------------------------------
# B. SSE 协议
# --------------------------------------------------------------------------

def test_sse_delta_events_in_order_and_content_type(fake_rag_stream):
    """正常：content-type 为 text/event-stream，delta 顺序正确，末尾单 done。"""
    with TestClient(app) as client:
        resp, events = _post(client, {"session_id": "sess-1", "message": "问题"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    names = [e for e, _ in events]
    assert names == ["delta", "delta", "delta", "done"]
    deltas = [d["content"] for e, d in events if e == "delta"]
    assert deltas == ["星河", "项目", "每天凌晨 2:30 备份。"]
    # 顶层参数透传给 service
    assert fake_rag_stream.calls == [("sess-1", "问题", None, None)]


def test_every_data_is_valid_json(fake_rag_stream):
    """每个 event 的 data 都是合法 JSON（parse_sse 内 json.loads 已保证，此处显式断言结构）。"""
    with TestClient(app) as client:
        resp, events = _post(client, {"session_id": "sess-1", "message": "问题"})
    for event, data in events:
        assert isinstance(event, str) and event
        assert isinstance(data, dict)
        if event == "delta":
            assert "content" in data and isinstance(data["content"], str)


def test_done_sent_exactly_once(fake_rag_stream):
    """正常结束：包含且只包含一个 done event，data 含 session_id。"""
    with TestClient(app) as client:
        _, events = _post(client, {"session_id": "sess-1", "message": "问题"})
    done = [d for e, d in events if e == "done"]
    assert len(done) == 1
    assert done[0] == {"session_id": "sess-1"}


def test_delta_concat_equals_full_answer(fake_rag_stream):
    """delta 内容拼接后等于完整 assistant answer。"""
    with TestClient(app) as client:
        _, events = _post(client, {"session_id": "sess-1", "message": "问题"})
    full = "".join(d["content"] for e, d in events if e == "delta")
    assert full == "星河项目每天凌晨 2:30 备份。"


def test_recall_rerank_forwarded_to_service(fake_rag_stream):
    """显式传入 recall_top_n / rerank_top_k 时应透传给 service。"""
    with TestClient(app) as client:
        resp, _ = _post(
            client,
            {
                "session_id": "sess-1",
                "message": "问题",
                "recall_top_n": 10,
                "rerank_top_k": 3,
            },
        )
    assert resp.status_code == 200
    assert fake_rag_stream.calls == [("sess-1", "问题", 10, 3)]


# --------------------------------------------------------------------------
# C. 流中途失败
# --------------------------------------------------------------------------

def test_mid_stream_failure_sends_error_not_done(monkeypatch):
    """流中途失败：已有 delta 保留，后续 error event，不发送 done，message 稳定非泄露。"""
    rag = FakeStreamingRAG(
        ["星河", "项目"],
        error=LLMError("DeepSeek stream failed mid-generation: connection reset"),
    )
    monkeypatch.setattr("app.api.v1.chat.answer_with_session_stream", rag)

    with TestClient(app) as client:
        resp, events = _post(client, {"session_id": "sess-1", "message": "问题"})

    assert resp.status_code == 200
    names = [e for e, _ in events]
    assert names == ["delta", "delta", "error"]  # 有 delta，最后 error，无 done
    err = events[-1][1]
    assert err["code"] == "LLM_ERROR"
    # message 稳定，不泄露原始异常细节（如 "connection reset"）
    assert err["message"] == "Failed to generate response."


# --------------------------------------------------------------------------
# D. route 职责边界：不重复写 Session
# --------------------------------------------------------------------------

def test_completions_does_not_touch_session_store(monkeypatch):
    """completions route 不访问 SessionStore（写回完全交给 service 层）。"""
    rag = FakeStreamingRAG(["a", "b"])
    monkeypatch.setattr("app.api.v1.chat.answer_with_session_stream", rag)
    accessed = []

    def _forbidden():
        accessed.append(True)
        raise AssertionError("completions route must not access session store")

    monkeypatch.setattr("app.api.v1.chat.get_session_store", _forbidden)

    with TestClient(app) as client:
        resp, events = _post(client, {"session_id": "sess-1", "message": "问题"})
    assert resp.status_code == 200
    assert [e for e, _ in events][-1] == "done"
    assert accessed == []


# --------------------------------------------------------------------------
# 端到端 Session 行为（真实 answer_with_session_stream + 假依赖 + 内存 store）
# --------------------------------------------------------------------------

CHUNKS = [
    RerankedChunk(
        document_id="doc-star",
        chunk_id="doc-star:0",
        index=0,
        text="星河项目数据库每天凌晨 2:30 执行备份。",
        title="运维手册",
        vector_distance=0.1,
        vector_similarity=0.9,
        rerank_score=0.9,
        recall_rank=1,
        final_rank=1,
    )
]


class MemorySessionStore:
    def __init__(self):
        self.messages: dict[str, list[dict]] = {}

    def get_recent_history(self, session_id, max_messages=None):
        msgs = self.messages.get(session_id, [])
        if max_messages is not None:
            return list(msgs[-max_messages:])
        return list(msgs)

    def get_history(self, session_id):
        return list(self.messages.get(session_id, []))

    def add_message(self, session_id, role, content):
        self.messages.setdefault(session_id, []).append({"role": role, "content": content})

    def clear_session(self, session_id):
        if session_id in self.messages:
            del self.messages[session_id]
            return 1
        return 0


class FakeRewrite:
    def __call__(self, history, question):
        return RewriteResult(query=question, used_llm=False, fallback=False)


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = list(chunks) if chunks is not None else list(CHUNKS)

    def __call__(self, query, recall_top_n=None, rerank_top_k=None):
        return list(self.chunks)


class FakeStreamLLM:
    def __init__(self, chunks, error=None):
        self.chunks = list(chunks)
        self.error = error

    def stream_generate(self, prompt):
        for c in self.chunks:
            yield c
        if self.error is not None:
            raise self.error


@pytest.fixture
def full_stack(monkeypatch):
    store = MemorySessionStore()
    llm = FakeStreamLLM(["根据知识库内容，星河项目数据库", "每天凌晨 2:30 执行备份。"])
    monkeypatch.setattr("app.services.rag.get_session_store", lambda: store)
    monkeypatch.setattr("app.services.rag.rewrite", FakeRewrite())
    monkeypatch.setattr("app.services.rag.retrieve_with_rerank", FakeRetriever())
    monkeypatch.setattr("app.services.rag.get_llm_client", lambda: llm)
    monkeypatch.setattr("app.api.v1.chat.get_session_store", lambda: store)
    return store, llm


def test_success_then_get_history(full_stack):
    """成功完成后 GET history 可见完整 user / assistant，assistant 为完整 answer。"""
    with TestClient(app) as client:
        resp, events = _post(client, {"session_id": "sess-1", "message": "问题"})
        assert [e for e, _ in events][-1] == "done"

        full = "".join(d["content"] for e, d in events if e == "delta")
        hist = client.get("/api/v1/chat/history/sess-1").json()

    assert hist["session_id"] == "sess-1"
    assert [m["role"] for m in hist["messages"]] == ["user", "assistant"]
    assert hist["messages"][0]["content"] == "问题"
    assert hist["messages"][1]["content"] == full  # 完整 answer，非逐 chunk


def test_failure_does_not_save_partial(full_stack):
    """流中途失败：GET history 为空（不保存 partial assistant）。"""
    _, llm = full_stack
    llm.chunks = ["部分回答"]
    llm.error = LLMError("mid-generation boom")

    with TestClient(app) as client:
        resp, events = _post(client, {"session_id": "sess-2", "message": "问题"})
        names = [e for e, _ in events]
        assert "error" in names and "done" not in names
        hist = client.get("/api/v1/chat/history/sess-2").json()

    assert hist["messages"] == []


def test_delete_then_empty_history(full_stack):
    """DELETE 后 GET history 为空；再次 DELETE 幂等返回 cleared=false。"""
    with TestClient(app) as client:
        _post(client, {"session_id": "sess-3", "message": "问题"})
        del_resp = client.delete("/api/v1/chat/session/sess-3")
        assert del_resp.status_code == 200
        assert del_resp.json()["cleared"] is True

        hist = client.get("/api/v1/chat/history/sess-3").json()
        assert hist["messages"] == []

        del2 = client.delete("/api/v1/chat/session/sess-3")
        assert del2.json()["cleared"] is False


def test_sessions_isolated(full_stack):
    """不同 session 不串话。"""
    with TestClient(app) as client:
        _post(client, {"session_id": "sess-a", "message": "问题A"})
        _post(client, {"session_id": "sess-b", "message": "问题B"})
        ha = client.get("/api/v1/chat/history/sess-a").json()
        hb = client.get("/api/v1/chat/history/sess-b").json()

    assert [m["content"] for m in ha["messages"] if m["role"] == "user"] == ["问题A"]
    assert [m["content"] for m in hb["messages"] if m["role"] == "user"] == ["问题B"]
