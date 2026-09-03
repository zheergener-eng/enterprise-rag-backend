"""单轮 RAG service 单元测试。

分别 mock Retriever 与 LLM，验证 RAG 编排逻辑（不消耗真实 API Token、不依赖 Redis）：
context 组装、prompt 构造、检索为空、空问题、answer 传递。
"""
from __future__ import annotations

import pytest

from app.services.rag import answer_question, build_context, build_prompt
from app.services.reranker import RerankedChunk


# 样例 chunk（模拟 retrieve_with_rerank 的返回，即两阶段后的 RerankedChunk）
CHUNKS = [
    RerankedChunk(
        document_id="doc-star",
        chunk_id="doc-star:0",
        index=0,
        text="星河项目的数据库备份时间为每天凌晨 2:30，备份文件保留 14 天。",
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
        text="数据库采用每日增量备份策略，全量备份每周一次。",
        title="",
        vector_distance=0.2,
        vector_similarity=0.8,
        rerank_score=0.5,
        recall_rank=2,
        final_rank=2,
    ),
]


class FakeRetriever:
    """记录调用，返回预设 chunk 列表（两阶段检索结果）。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls: list[tuple] = []

    def __call__(self, query, recall_top_n=None, rerank_top_k=None):
        self.calls.append((query, recall_top_n, rerank_top_k))
        return self.chunks


class FakeLLM:
    """记录 generate 的 prompt，返回预设回答。"""

    def __init__(self, answer="这是回答"):
        self.answer = answer
        self.calls: list[str] = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.answer


@pytest.fixture
def rag_services(monkeypatch):
    """mock rag 模块依赖的 retrieve_with_rerank 与 get_llm_client。"""
    retriever = FakeRetriever(CHUNKS)
    llm = FakeLLM("凌晨 2:30")
    monkeypatch.setattr("app.services.rag.retrieve_with_rerank", retriever)
    monkeypatch.setattr("app.services.rag.get_llm_client", lambda: llm)
    return retriever, llm


# --------------------------------------------------------------------------
# context / prompt 构造
# --------------------------------------------------------------------------

def test_build_context_organizes_chunks():
    """context 应包含标题/内容，且不含向量细节。"""
    ctx = build_context(CHUNKS)
    assert "[Document 1]" in ctx
    assert "标题：运维手册" in ctx
    assert "内容：星河项目的数据库备份时间为每天凌晨 2:30" in ctx
    assert "[Document 2]" in ctx
    assert "标题：（无标题）" in ctx  # 空 title 兜底
    # 不包含向量细节
    assert "embedding" not in ctx
    assert "distance" not in ctx


def test_build_prompt_contains_instruction_context_question():
    """prompt 应包含系统规则、知识库 context 与当前问题。"""
    ctx = build_context(CHUNKS)
    prompt = build_prompt("星河项目什么时候备份？", ctx)
    assert "知识库问答助手" in prompt           # 系统角色
    assert "不得编造" in prompt                  # 回答规则
    assert "星河项目的数据库备份时间为每天凌晨 2:30" in prompt  # context
    assert "星河项目什么时候备份？" in prompt      # question


# --------------------------------------------------------------------------
# answer_question 编排
# --------------------------------------------------------------------------

def test_question_passed_to_retrieve(rag_services):
    """question 正确传入 retrieve_with_rerank，默认不传 recall/rerank。"""
    retriever, _ = rag_services
    answer_question("  星河项目什么时候备份？  ")
    assert retriever.calls[0][0] == "星河项目什么时候备份？"  # 去空白后
    assert retriever.calls[0][1] is None                       # recall_top_n 默认
    assert retriever.calls[0][2] is None                       # rerank_top_k 默认


def test_recall_and_rerank_passed_to_retrieve(rag_services):
    """显式 recall_top_n / rerank_top_k 正确传递。"""
    retriever, _ = rag_services
    answer_question("问题", recall_top_n=10, rerank_top_k=3)
    assert retriever.calls[0][1] == 10
    assert retriever.calls[0][2] == 3


def test_llm_receives_built_prompt(rag_services):
    """LLM 收到构造后的 RAG prompt（含 context 与 question）。"""
    _, llm = rag_services
    answer_question("星河项目什么时候备份？")
    prompt = llm.calls[0]
    assert "星河项目的数据库备份时间为每天凌晨 2:30" in prompt
    assert "星河项目什么时候备份？" in prompt
    assert "知识库内容" in prompt


def test_answer_and_sources_returned(rag_services):
    """answer / sources / retrieval_count / answered 正确返回。"""
    result = answer_question("星河项目什么时候备份？")
    assert result.answer == "凌晨 2:30"
    assert result.retrieval_count == 2
    assert result.answered is True
    assert result.relevance is not None
    assert result.relevance.is_relevant is True
    assert len(result.sources) == 2
    assert result.sources[0]["document_id"] == "doc-star"
    assert result.sources[0]["text"] == CHUNKS[0].text
    assert "embedding" not in result.sources[0]
    assert "distance" not in result.sources[0]


def test_empty_retrieval_skips_llm(monkeypatch):
    """检索为空：不调用 LLM，返回明确的“无法回答”结果。"""
    retriever = FakeRetriever([])
    llm = FakeLLM()
    monkeypatch.setattr("app.services.rag.retrieve_with_rerank", retriever)
    monkeypatch.setattr("app.services.rag.get_llm_client", lambda: llm)

    result = answer_question("星河项目负责人是谁？")

    assert result.answered is False
    assert result.retrieval_count == 0
    assert result.sources == []
    assert "无法确定" in result.answer
    assert result.relevance is not None
    assert result.relevance.is_relevant is False
    assert llm.calls == []  # 未调用 LLM


def test_low_relevance_skips_llm(monkeypatch):
    """检索非空但 top-1 分数低于阈值：Relevance Gate 判定 insufficient，
    不调用 LLM，返回统一 no-answer，sources 为空（不把低相关 chunks 当依据）。"""
    low_chunks = [
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
    retriever = FakeRetriever(low_chunks)
    llm = FakeLLM("不该被调用")
    monkeypatch.setattr("app.services.rag.retrieve_with_rerank", retriever)
    monkeypatch.setattr("app.services.rag.get_llm_client", lambda: llm)

    result = answer_question("问题")

    assert result.answered is False
    assert result.retrieval_count == 1          # 检索到了 chunk，但被 gate 拦下
    assert result.sources == []                  # 不暴露低相关 sources
    assert "无法确定" in result.answer
    assert result.relevance is not None
    assert result.relevance.is_relevant is False
    assert llm.calls == []                       # 未调用 LLM


def test_empty_question_rejected():
    """空问题 / 纯空白问题应被拒绝。"""
    for bad in ["", "   ", "\n\t "]:
        with pytest.raises(ValueError):
            answer_question(bad)
