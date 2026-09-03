"""多轮 RAG 流式（answer_with_session_stream）单元测试。

mock SessionStore / Query Rewrite / Retrieval / LLM，验证流式编排：
复用前置（history / rewrite / retrieval / prompt 构造）、流式 yield、full_answer 累积、
正常结束后写回完整 answer、中途失败不写回 partial、session 隔离。
"""
from __future__ import annotations

import pytest

from app.services.llm import LLMError
from app.services.query_rewrite import RewriteResult
from app.services.rag import answer_with_session_stream
from app.services.reranker import RerankedChunk


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
    ),
    RerankedChunk(
        document_id="doc-star",
        chunk_id="doc-star:1",
        index=1,
        text="备份文件保留 14 天。",
        title="运维手册",
        vector_distance=0.2,
        vector_similarity=0.8,
        rerank_score=0.5,
        recall_rank=2,
        final_rank=2,
    ),
]


class FakeSessionStore:
    def __init__(self, history=None):
        self.history = list(history or [])
        self.recent_calls: list[tuple] = []
        self.writes: list[tuple] = []  # (session_id, role, content)

    def get_recent_history(self, session_id, max_messages=None):
        self.recent_calls.append((session_id, max_messages))
        return list(self.history)

    def add_message(self, session_id, role, content):
        self.writes.append((session_id, role, content))


class FakeRewriter:
    def __init__(self, query="星河项目的数据库备份保留多久？", used_llm=True, fallback=False):
        self.query = query
        self.used_llm = used_llm
        self.fallback = fallback
        self.calls: list[tuple] = []

    def __call__(self, history, question):
        self.calls.append((list(history), question))
        return RewriteResult(query=self.query, used_llm=self.used_llm, fallback=self.fallback)


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = list(chunks) if chunks is not None else list(CHUNKS)
        self.calls: list[tuple] = []

    def __call__(self, query, recall_top_n=None, rerank_top_k=None):
        self.calls.append((query, recall_top_n, rerank_top_k))
        return list(self.chunks)


class FakeStreamingLLM:
    """记录 stream_generate 的 prompt，yield 预设 chunks，可选中途抛异常。"""

    def __init__(self, chunks=None, error=None):
        self.chunks = list(chunks) if chunks is not None else ["备份文件", "保留 14 天。"]
        self.error = error  # 迭代完 chunks 后抛出
        self.prompts: list[str] = []

    def stream_generate(self, prompt):
        self.prompts.append(prompt)
        for c in self.chunks:
            yield c
        if self.error is not None:
            raise self.error


@pytest.fixture
def services(monkeypatch):
    session = FakeSessionStore()
    retriever = FakeRetriever()
    llm = FakeStreamingLLM()
    rewriter = FakeRewriter()
    monkeypatch.setattr("app.services.rag.get_session_store", lambda: session)
    monkeypatch.setattr("app.services.rag.retrieve_with_rerank", retriever)
    monkeypatch.setattr("app.services.rag.get_llm_client", lambda: llm)
    monkeypatch.setattr("app.services.rag.rewrite", rewriter)
    return session, retriever, llm, rewriter


def _collect(gen) -> list[str]:
    return list(gen)


# --------------------------------------------------------------------------
# 正常流式 / 写回
# --------------------------------------------------------------------------

def test_first_turn_streaming_rag(services):
    """第一轮正常流式：yield 全部 chunk，正常结束后写回完整 answer。"""
    session, retriever, llm, rewriter = services
    rewriter.query = "星河项目每天什么时候进行数据库备份？"
    rewriter.used_llm = False
    llm.chunks = ["星河项目", "数据库每天", "凌晨 2:30", "执行备份。"]

    q = "星河项目每天什么时候进行数据库备份？"
    chunks = _collect(answer_with_session_stream("sess-1", q))

    assert chunks == ["星河项目", "数据库每天", "凌晨 2:30", "执行备份。"]
    full = "星河项目数据库每天凌晨 2:30执行备份。"
    assert session.writes == [
        ("sess-1", "user", q),
        ("sess-1", "assistant", full),
    ]


def test_second_turn_runs_query_rewrite(services):
    """多轮问题仍执行 Query Rewrite（rewriter 收到 history + 原始问题）。"""
    session, retriever, llm, rewriter = services
    session.history = [
        {"role": "user", "content": "星河项目什么时候进行数据库备份？"},
        {"role": "assistant", "content": "每天凌晨 2:30。"},
    ]
    rewriter.query = "星河项目的数据库备份保留多久？"
    q = "那保留多久？"
    _collect(answer_with_session_stream("sess-1", q))

    assert rewriter.calls == [(session.history, q)]


def test_retrieval_uses_rewritten_query(services):
    """Retrieval 使用 rewritten query，而非原始问题。"""
    session, retriever, llm, rewriter = services
    session.history = [{"role": "user", "content": "星河项目什么时候备份？"}]
    rewriter.query = "星河项目的数据库备份保留多久？"
    q = "那保留多久？"
    _collect(answer_with_session_stream("sess-1", q))

    assert retriever.calls[0][0] == "星河项目的数据库备份保留多久？"


def test_prompt_keeps_original_question(services):
    """最终 Prompt 仍使用原始问题，而非 rewritten query。"""
    session, retriever, llm, rewriter = services
    session.history = [
        {"role": "user", "content": "星河项目什么时候进行数据库备份？"},
        {"role": "assistant", "content": "每天凌晨 2:30。"},
    ]
    rewriter.query = "星河项目的数据库备份保留多久？"
    q = "那保留多久？"
    _collect(answer_with_session_stream("sess-1", q))

    prompt = llm.prompts[0]
    assert "那保留多久？" in prompt                       # 原始问题在 prompt
    assert "星河项目什么时候进行数据库备份？" in prompt      # 历史也在
    assert "星河项目的数据库备份保留多久？" not in prompt     # rewritten query 不进 prompt


def test_session_saves_single_full_answer(services):
    """正常结束：assistant 只写一条完整消息，不逐 chunk 保存。"""
    session, retriever, llm, rewriter = services
    rewriter.query = "q"
    rewriter.used_llm = False
    llm.chunks = ["a", "b", "c"]
    _collect(answer_with_session_stream("sess-1", "问题"))

    assistant_writes = [c for _, role, c in session.writes if role == "assistant"]
    assert assistant_writes == ["abc"]  # 仅一条，拼接完整


# --------------------------------------------------------------------------
# 失败 / 空检索 / 隔离
# --------------------------------------------------------------------------

def test_mid_stream_failure_writes_nothing(services):
    """流中途失败：不写回 partial assistant answer，异常向上传播。"""
    session, retriever, llm, rewriter = services
    rewriter.query = "q"
    rewriter.used_llm = False
    llm.chunks = ["根据知识库内容，星河项目"]
    llm.error = LLMError("DeepSeek stream failed mid-generation")

    with pytest.raises(LLMError):
        _collect(answer_with_session_stream("sess-1", "问题"))

    assert session.writes == []  # 不写 partial，也不写 user（与非流式失败策略一致）


def test_empty_retrieval_no_llm_yields_cannot_answer(services):
    """检索为空：不调用 LLM，单次 yield 统一“无法确定”，并写回。"""
    session, retriever, llm, rewriter = services
    rewriter.query = "q"
    rewriter.used_llm = False
    retriever.chunks = []

    chunks = _collect(answer_with_session_stream("sess-1", "问题"))
    assert chunks == ["根据当前知识库无法确定。"]
    assert llm.prompts == []  # 未调用 stream_generate
    assert [role for _, role, _ in session.writes] == ["user", "assistant"]


def test_insufficient_relevance_streaming_no_llm(services):
    """检索非空但 Relevance Gate 判定 insufficient：不调用流式 LLM，
    单次 yield 统一 no-answer，写回 user + assistant(no-answer)。"""
    session, retriever, llm, rewriter = services
    rewriter.query = "q"
    rewriter.used_llm = False
    retriever.chunks = [
        RerankedChunk(
            document_id="doc-star",
            chunk_id="doc-star:0",
            index=0,
            text="无关内容",
            title="运维手册",
            vector_distance=0.9,
            vector_similarity=0.1,
            rerank_score=0.1,
            recall_rank=1,
            final_rank=1,
        )
    ]

    q = "明天天气怎么样？"
    chunks = _collect(answer_with_session_stream("sess-1", q))

    assert chunks == ["根据当前知识库无法确定。"]
    assert llm.prompts == []  # 未调用 stream_generate
    assert session.writes == [
        ("sess-1", "user", q),
        ("sess-1", "assistant", "根据当前知识库无法确定。"),
    ]


def test_sessions_isolated(services):
    """不同 session 各自读写，互不串扰。"""
    session, retriever, llm, rewriter = services
    rewriter.query = "q"
    rewriter.used_llm = False
    _collect(answer_with_session_stream("sess-a", "问题A"))
    _collect(answer_with_session_stream("sess-b", "问题B"))

    assert [sid for sid, _ in session.recent_calls] == ["sess-a", "sess-b"]
    assert [sid for sid, _, _ in session.writes] == ["sess-a", "sess-a", "sess-b", "sess-b"]


def test_empty_question_rejected():
    """空问题 / 纯空白问题被拒绝（ValueError，迭代时触发）。"""
    with pytest.raises(ValueError):
        _collect(answer_with_session_stream("sess-1", "   "))
