"""Rerank Service 测试（mock，不下载真实模型）。

覆盖：
- 空候选 → 空列表；
- 单候选正常返回；
- 多候选按 rerank_score 降序；
- metadata 不丢失 + vector distance/similarity 与 rerank_score 分字段保留；
- top_k 生效；
- top_k > 候选数返回全部；
- top_k <= 0 拒绝；
- 模型只初始化一次；
- query 空拒绝。

所有打分逻辑都用假 Reranker（monkeypatch get_reranker / CrossEncoder），
不触发真实 bge-reranker-base 下载。
"""
from __future__ import annotations

import pytest

from app.services.reranker import (
    RerankedChunk,
    RetrievedChunk,
    Reranker,
    rerank,
)


def _candidate(
    document_id="doc-1",
    chunk_id="doc-1:0",
    index=0,
    text="内容",
    title="标题",
    distance=0.1,
    similarity=0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=document_id,
        chunk_id=chunk_id,
        index=index,
        text=text,
        title=title,
        distance=distance,
        similarity=similarity,
    )


class FakeReranker:
    """记录 (query, chunk) pair、返回预设分数。"""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls: list[list[tuple[str, str]]] = []

    def score_pairs(self, pairs):
        self.calls.append(pairs)
        return self.scores


@pytest.fixture
def fake_reranker(monkeypatch):
    """把 get_reranker 替换为假实现，避免加载真实模型。"""
    model = FakeReranker([0.2, 0.9, 0.5])
    monkeypatch.setattr("app.services.reranker.get_reranker", lambda: model)
    return model


CANDIDATES = [
    _candidate(document_id="doc-a", chunk_id="doc-a:0", index=0, text="A 文本", title="A"),
    _candidate(document_id="doc-b", chunk_id="doc-b:0", index=0, text="B 文本", title="B"),
    _candidate(document_id="doc-c", chunk_id="doc-c:0", index=0, text="C 文本", title="C"),
]


def test_empty_candidates_returns_empty(fake_reranker):
    """候选为空 → 返回空列表，且不调用打分模型。"""
    assert rerank("query", [], top_k=3) == []
    assert fake_reranker.calls == []


def test_single_candidate(fake_reranker):
    """单候选正常返回：列表长度 1，metadata 保留，含 rerank_score。"""
    c = _candidate(document_id="doc-x", chunk_id="doc-x:0", index=0, text="X", title="X")
    fake_reranker.scores = [0.7]
    out = rerank("query", [c], top_k=3)
    assert len(out) == 1
    r = out[0]
    assert r.document_id == "doc-x"
    assert r.chunk_id == "doc-x:0"
    assert r.index == 0
    assert r.text == "X"
    assert r.title == "X"
    assert r.rerank_score == 0.7


def test_sorted_descending(fake_reranker):
    """多候选按 rerank_score 降序排列。"""
    out = rerank("query", CANDIDATES, top_k=3)
    scores = [r.rerank_score for r in out]
    assert scores == sorted(scores, reverse=True)
    # 分数 [0.2, 0.9, 0.5] → 降序 [0.9, 0.5, 0.2]
    assert [r.document_id for r in out] == ["doc-b", "doc-c", "doc-a"]


def test_recall_and_final_rank_recorded(fake_reranker):
    """recall_rank 保留原召回位次，final_rank 为 rerank 降序后的位次。"""
    out = rerank("query", CANDIDATES, top_k=3)
    ranks = {r.document_id: (r.recall_rank, r.final_rank) for r in out}
    assert ranks == {
        "doc-b": (2, 1),
        "doc-c": (3, 2),
        "doc-a": (1, 3),
    }


def test_metadata_preserved(fake_reranker):
    """metadata 不丢失：document_id/chunk_id/index/text/title 全保留，
    且 vector_distance/vector_similarity 与 rerank_score 分字段保留。"""
    c = _candidate(
        document_id="doc-m", chunk_id="doc-m:3", index=3, text="M 内容", title="M 标题",
        distance=0.12, similarity=0.88,
    )
    fake_reranker.scores = [0.4]
    r = rerank("query", [c], top_k=3)[0]

    assert r.document_id == "doc-m"
    assert r.chunk_id == "doc-m:3"
    assert r.index == 3
    assert r.text == "M 内容"
    assert r.title == "M 标题"
    assert r.vector_distance == 0.12
    assert r.vector_similarity == 0.88
    assert r.rerank_score == 0.4


def test_top_k_takes_effect(fake_reranker):
    """top_k 截断生效。"""
    out = rerank("query", CANDIDATES, top_k=2)
    assert [r.document_id for r in out] == ["doc-b", "doc-c"]


def test_top_k_greater_than_candidates(fake_reranker):
    """top_k 大于候选数 → 返回全部（仍按分数降序）。"""
    out = rerank("query", CANDIDATES, top_k=10)
    assert len(out) == 3
    assert [r.document_id for r in out] == ["doc-b", "doc-c", "doc-a"]


@pytest.mark.parametrize("top_k", [0, -1, -100])
def test_top_k_nonpositive_rejected(top_k):
    """top_k <= 0 → 明确拒绝（ValueError）。"""
    with pytest.raises(ValueError):
        rerank("query", CANDIDATES, top_k=top_k)


def test_blank_query_rejected():
    """query 空 / 仅空白 → ValueError。"""
    for q in ["", "   ", "\n\t "]:
        with pytest.raises(ValueError):
            rerank(q, CANDIDATES, top_k=3)


def test_pairs_use_query_and_text(fake_reranker):
    """构造的 pair 为 (query, chunk.text)。"""
    rerank("数据库备份保留多久？", CANDIDATES, top_k=3)
    expected = [("数据库备份保留多久？", c.text) for c in CANDIDATES]
    assert fake_reranker.calls == [expected]


def test_model_initialized_once(monkeypatch):
    """Reranker 懒加载：多次 score_pairs 只实例化一次 CrossEncoder。"""
    calls: list[tuple] = []

    class FakeCrossEncoder:
        def __init__(self, model_name, device="cpu"):
            calls.append((model_name, device))

        def predict(self, pairs):
            return [0.5 for _ in pairs]

    monkeypatch.setattr("sentence_transformers.CrossEncoder", FakeCrossEncoder)

    r = Reranker(model_name="BAAI/bge-reranker-base")
    r.score_pairs([("q", "a")])
    r.score_pairs([("q", "b")])

    assert len(calls) == 1
    assert calls[0] == ("BAAI/bge-reranker-base", "cpu")


def test_retrieved_chunk_from_dict():
    """RetrievedChunk.from_dict 桥接现有 dict 召回结果（字段映射正确）。"""
    c = RetrievedChunk.from_dict(
        {
            "document_id": "doc-d",
            "chunk_id": "doc-d:1",
            "index": 1,
            "text": "D 内容",
            "title": "D 标题",
            "distance": 0.25,
            "similarity": 0.75,
        }
    )
    assert c.document_id == "doc-d"
    assert c.chunk_id == "doc-d:1"
    assert c.index == 1
    assert c.text == "D 内容"
    assert c.title == "D 标题"
    assert c.distance == 0.25
    assert c.similarity == 0.75
