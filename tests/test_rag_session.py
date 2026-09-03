"""多轮 RAG（answer_with_session）单元测试。

mock SessionStore / Query Rewrite / Retrieval / LLM，验证多轮编排逻辑：
history 读取、rewrite 状态透传、rewritten query 用于检索、原始问题进入最终 Prompt、
写回顺序与内容、不同 session 隔离、fallback、空检索、LLM 失败、max_history 委托。
"""
from __future__ import annotations

import pytest

from app.services.llm import LLMError
from app.services.query_rewrite import RewriteResult
from app.services.rag import answer_with_session, build_prompt_with_history
from app.services.reranker import RerankedChunk


# 样例检索 chunk（两阶段后的 RerankedChunk）
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
    """记录 get_recent_history / get_history / add_message 调用，返回预设历史。"""

    def __init__(self, history=None):
        self.history = list(history or [])
        self.recent_calls: list[tuple] = []
        self.full_calls: list[str] = []
        self.writes: list[tuple] = []  # (session_id, role, content)

    def get_recent_history(self, session_id, max_messages=None):
        self.recent_calls.append((session_id, max_messages))
        return list(self.history)

    def get_history(self, session_id):
        self.full_calls.append(session_id)
        return list(self.history)

    def add_message(self, session_id, role, content):
        self.writes.append((session_id, role, content))


class FakeRewriter:
    """记录 (history, question) 调用，返回预设 RewriteResult。"""

    def __init__(self, query="星河项目的数据库备份保留多久？", used_llm=True, fallback=False):
        self.query = query
        self.used_llm = used_llm
        self.fallback = fallback
        self.calls: list[tuple] = []

    def __call__(self, history, question):
        self.calls.append((list(history), question))
        return RewriteResult(query=self.query, used_llm=self.used_llm, fallback=self.fallback)


class FakeRetriever:
    """记录 (query, recall_top_n, rerank_top_k) 调用，返回预设 chunk 列表。"""

    def __init__(self, chunks=None):
        self.chunks = list(chunks) if chunks is not None else list(CHUNKS)
        self.calls: list[tuple] = []

    def __call__(self, query, recall_top_n=None, rerank_top_k=None):
        self.calls.append((query, recall_top_n, rerank_top_k))
        return list(self.chunks)


class FakeLLM:
    """记录 generate 的 prompt，返回预设回答或抛出预设异常。"""

    def __init__(self, answer="备份文件保留 14 天。", error=None):
        self.answer = answer
        self.error = error
        self.calls: list[str] = []

    def generate(self, prompt):
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return self.answer


@pytest.fixture
def services(monkeypatch):
    """将 answer_with_session 依赖的四个服务替换为假实现。"""
    session = FakeSessionStore()
    retriever = FakeRetriever()
    llm = FakeLLM()
    rewriter = FakeRewriter()
    monkeypatch.setattr("app.services.rag.get_session_store", lambda: session)
    monkeypatch.setattr("app.services.rag.retrieve_with_rerank", retriever)
    monkeypatch.setattr("app.services.rag.get_llm_client", lambda: llm)
    monkeypatch.setattr("app.services.rag.rewrite", rewriter)
    return session, retriever, llm, rewriter


# --------------------------------------------------------------------------
# 第一轮 / 指代
# --------------------------------------------------------------------------

def test_first_turn_empty_history_uses_original(services):
    """新 session 第一轮：history 为空，rewrite 返回原问题，检索用原问题。"""
    session, retriever, llm, rewriter = services
    rewriter.query = "Redis 如何创建向量索引？"
    rewriter.used_llm = False
    q = "Redis 如何创建向量索引？"
    r = answer_with_session("sess-1", q)

    assert rewriter.calls == [([], q)]           # rewrite 收到空历史
    assert retriever.calls[0][0] == q            # 检索用原问题
    assert r.answer == llm.answer
    assert r.answered is True
    assert session.writes == [("sess-1", "user", q), ("sess-1", "assistant", llm.answer)]


def test_second_turn_referential_uses_rewritten_and_keeps_original(services):
    """第二轮指代：检索用 rewritten query，最终 Prompt 仍保留原始问题。"""
    session, retriever, llm, rewriter = services
    session.history = [
        {"role": "user", "content": "星河项目什么时候进行数据库备份？"},
        {"role": "assistant", "content": "每天凌晨 2:30。"},
    ]
    rewriter.query = "星河项目的数据库备份保留多久？"
    rewriter.used_llm = True
    q = "那保留多久？"
    r = answer_with_session("sess-1", q)

    # 检索使用 rewritten query，而非原始问题
    assert retriever.calls[0][0] == "星河项目的数据库备份保留多久？"

    # 最终 Prompt 三部分清晰，且 Current Question 是原始问题
    prompt = llm.calls[0]
    assert "【对话历史】" in prompt
    assert "【知识库内容】" in prompt
    assert "【当前问题】" in prompt
    assert "那保留多久？" in prompt
    assert "星河项目什么时候进行数据库备份？" in prompt
    assert "每天凌晨 2:30。" in prompt

    # 结果字段
    assert r.original_question == "那保留多久？"
    assert r.rewritten_query == "星河项目的数据库备份保留多久？"
    assert r.rewrite_used_llm is True
    assert r.rewrite_fallback is False
    assert r.answer == llm.answer


# --------------------------------------------------------------------------
# 写回
# --------------------------------------------------------------------------

def test_session_write_order(services):
    """两轮写回顺序：user1/assistant1/user2/assistant2。"""
    session, retriever, llm, rewriter = services

    # 第一轮（空历史）
    rewriter.query = "星河项目每天什么时候进行数据库备份？"
    rewriter.used_llm = False
    llm.answer = "凌晨 2:30。"
    answer_with_session("sess-1", "星河项目每天什么时候进行数据库备份？")

    # 第二轮（有历史）
    session.history = [
        {"role": "user", "content": "星河项目每天什么时候进行数据库备份？"},
        {"role": "assistant", "content": "凌晨 2:30。"},
    ]
    rewriter.query = "星河项目的数据库备份保留多久？"
    rewriter.used_llm = True
    llm.answer = "备份文件保留 14 天。"
    answer_with_session("sess-1", "那保留多久？")

    assert [r for _, r, _ in session.writes] == ["user", "assistant", "user", "assistant"]
    assert [c for _, _, c in session.writes] == [
        "星河项目每天什么时候进行数据库备份？",
        "凌晨 2:30。",
        "那保留多久？",
        "备份文件保留 14 天。",
    ]


def test_write_back_uses_original_question(services):
    """写入 Session 的 user 是原始问题，而非 rewritten query。"""
    session, retriever, llm, rewriter = services
    session.history = [
        {"role": "user", "content": "星河项目什么时候进行数据库备份？"},
        {"role": "assistant", "content": "每天凌晨 2:30。"},
    ]
    rewriter.query = "星河项目的数据库备份保留多久？"  # 与原始问题不同
    answer_with_session("sess-1", "那保留多久？")

    user_writes = [c for _, role, c in session.writes if role == "user"]
    assert user_writes == ["那保留多久？"]
    assert "星河项目的数据库备份保留多久？" not in user_writes


# --------------------------------------------------------------------------
# 隔离 / 委托
# --------------------------------------------------------------------------

def test_sessions_isolated(services):
    """不同 session 各自用各自的 id 读取与写回。"""
    session, retriever, llm, rewriter = services
    rewriter.query = "q"
    answer_with_session("sess-a", "问题A")
    answer_with_session("sess-b", "问题B")

    assert [sid for sid, _ in session.recent_calls] == ["sess-a", "sess-b"]
    assert [sid for sid, _, _ in session.writes] == ["sess-a", "sess-a", "sess-b", "sess-b"]


def test_uses_recent_history_not_full(services):
    """多轮 RAG 通过 get_recent_history 读历史（受 max 限制），而非完整 get_history。"""
    session, retriever, llm, rewriter = services
    session.history = [{"role": "user", "content": "历史"}]
    answer_with_session("sess-1", "问题")
    assert session.recent_calls == [("sess-1", None)]  # 未显式传 max，用 store 默认（settings）
    assert session.full_calls == []                    # 未读取完整历史


# --------------------------------------------------------------------------
# fallback / 空检索 / LLM 失败
# --------------------------------------------------------------------------

def test_rewrite_fallback_continues(services):
    """Query Rewrite fallback：检索继续执行，fallback 状态保留。"""
    session, retriever, llm, rewriter = services
    session.history = [{"role": "user", "content": "历史"}]
    rewriter.query = "那保留多久？"  # fallback 后 query == 原问题
    rewriter.used_llm = True
    rewriter.fallback = True
    q = "那保留多久？"
    r = answer_with_session("sess-1", q)

    assert retriever.calls[0][0] == "那保留多久？"  # 继续用 fallback query 检索
    assert r.answered is True
    assert r.rewrite_fallback is True
    assert r.rewrite_used_llm is True


def test_empty_retrieval_no_hallucination(services):
    """检索为空：不调用 LLM，返回统一“无法确定”，不产生知识库外回答。"""
    session, retriever, llm, rewriter = services
    retriever.chunks = []
    r = answer_with_session("sess-1", "问题")

    assert r.answered is False
    assert "无法确定" in r.answer
    assert r.relevance is not None
    assert r.relevance.is_relevant is False
    assert llm.calls == []  # 未调用 LLM


def test_insufficient_relevance_writes_no_answer(services):
    """检索非空但 Relevance Gate 判定 insufficient：不调用 LLM，
    写回 user + 统一 no-answer，sources 为空。"""
    session, retriever, llm, rewriter = services
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
    q = "公司食堂几点开门？"
    r = answer_with_session("sess-1", q)

    assert r.answered is False
    assert r.relevance is not None
    assert r.relevance.is_relevant is False
    assert r.sources == []
    assert llm.calls == []  # 未调用 LLM
    assert session.writes == [
        ("sess-1", "user", q),
        ("sess-1", "assistant", "根据当前知识库无法确定。"),
    ]


def test_llm_failure_writes_nothing(services):
    """LLM 调用失败：异常向上传播，不写入任何消息（不产生虚假 answer）。"""
    session, retriever, llm, rewriter = services
    llm.error = LLMError("DeepSeek API call failed")
    with pytest.raises(LLMError):
        answer_with_session("sess-1", "问题")
    assert session.writes == []


# --------------------------------------------------------------------------
# prompt 构造 / 空问题
# --------------------------------------------------------------------------

def test_build_prompt_with_history_sections():
    """多轮 prompt 三部分清晰、顺序正确，含历史约束规则。"""
    history = [
        {"role": "user", "content": "星河项目什么时候进行数据库备份？"},
        {"role": "assistant", "content": "每天凌晨 2:30。"},
    ]
    prompt = build_prompt_with_history(history, "【context】", "那保留多久？")
    assert "【对话历史】" in prompt
    assert "【知识库内容】" in prompt
    assert "【当前问题】" in prompt
    # 用 rfind：三个标记也出现在系统指令里，真正的分节标题是最后一次出现
    assert prompt.rfind("【对话历史】") < prompt.rfind("【知识库内容】") < prompt.rfind("【当前问题】")
    assert "星河项目什么时候进行数据库备份？" in prompt
    assert "那保留多久？" in prompt
    assert "可靠知识来源" in prompt  # 历史约束规则


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_empty_question_rejected(bad):
    """空问题 / 纯空白问题应被拒绝（ValueError）。"""
    with pytest.raises(ValueError):
        answer_with_session("sess-1", bad)
