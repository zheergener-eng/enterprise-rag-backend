"""Relevance Gate（evaluate_relevance）单元测试。

验证结构化决策：
- 候选为空 → insufficient（不调用 LLM / 模型）；
- top-1 分数低于阈值 → insufficient；
- top-1 分数高于阈值 → relevant；
- top-1 分数 == 阈值 → relevant（边界闭区间，`>=` 判定）；
- 阈值缺省来自 settings（配置化）；
- 非法阈值（超出 [0, 1]）有明确处理（ValueError）。

全部用构造的 RerankedChunk，不触发真实模型 / Redis。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.relevance import (
    BELOW_THRESHOLD,
    MEETS_THRESHOLD,
    NO_CANDIDATES,
    RelevanceResult,
    evaluate_relevance,
)
from app.services.reranker import RerankedChunk


def _chunk(score: float, document_id: str = "doc") -> RerankedChunk:
    """构造一条 RerankedChunk（仅 rerank_score 参与判定，其余字段占位）。"""
    return RerankedChunk(
        document_id=document_id,
        chunk_id=f"{document_id}:0",
        index=0,
        text="内容",
        title="标题",
        vector_distance=0.1,
        vector_similarity=0.9,
        rerank_score=score,
        recall_rank=1,
        final_rank=1,
    )


def test_empty_candidates_insufficient():
    """候选为空 → insufficient，top_score=None，reason=no_candidates，且不调用模型。"""
    result = evaluate_relevance([], threshold=0.5)
    assert isinstance(result, RelevanceResult)
    assert result.is_relevant is False
    assert result.top_score is None
    assert result.threshold == 0.5
    assert result.reason == NO_CANDIDATES


def test_below_threshold_insufficient():
    """top-1 分数明显低于阈值 → insufficient。"""
    result = evaluate_relevance([_chunk(0.3)], threshold=0.5)
    assert result.is_relevant is False
    assert result.top_score == 0.3
    assert result.threshold == 0.5
    assert result.reason == BELOW_THRESHOLD


def test_above_threshold_relevant():
    """top-1 分数明显高于阈值 → relevant。"""
    result = evaluate_relevance([_chunk(0.9)], threshold=0.5)
    assert result.is_relevant is True
    assert result.top_score == 0.9
    assert result.reason == MEETS_THRESHOLD


def test_equal_threshold_boundary_relevant():
    """top-1 分数 == 阈值 → relevant（边界明确：`>=` 判定）。"""
    result = evaluate_relevance([_chunk(0.5)], threshold=0.5)
    assert result.is_relevant is True
    assert result.reason == MEETS_THRESHOLD


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.79, False),  # 略低于默认阈值 0.8 → reject
        (0.80, True),   # 恰好等于 0.8 → pass（`>=` 闭区间）
        (0.81, True),   # 略高于 0.8 → pass
    ],
)
def test_default_threshold_boundary_0_8(score, expected):
    """默认阈值 0.8 的边界行为：0.79 拒绝，0.80 / 0.81 通过。"""
    result = evaluate_relevance([_chunk(score)], threshold=0.8)
    assert result.is_relevant is expected


def test_top_one_used_not_second():
    """判定只依据 top-1（已降序）的最高分，忽略后续低分候选。"""
    chunks = [_chunk(0.9, "top"), _chunk(0.1, "low")]
    result = evaluate_relevance(chunks, threshold=0.5)
    assert result.is_relevant is True
    assert result.top_score == 0.9


def test_threshold_read_from_settings(monkeypatch):
    """未显式传 threshold 时，从 settings 读取（配置化，不硬编码）。"""
    monkeypatch.setattr(
        "app.services.relevance.settings",
        SimpleNamespace(rerank_relevance_threshold=0.8),
    )
    result = evaluate_relevance([_chunk(0.75)])
    assert result.threshold == 0.8
    assert result.is_relevant is False  # 0.75 < 0.8

    result_hi = evaluate_relevance([_chunk(0.85)])
    assert result_hi.threshold == 0.8
    assert result_hi.is_relevant is True


@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0, -100.0])
def test_invalid_threshold_rejected(bad):
    """非法阈值（超出 [0, 1]）→ 明确 ValueError。"""
    with pytest.raises(ValueError):
        evaluate_relevance([_chunk(0.5)], threshold=bad)


def test_threshold_bounds_inclusive():
    """阈值边界 0 与 1 均合法（不抛错）。"""
    assert evaluate_relevance([_chunk(0.0)], threshold=0.0).is_relevant is True
    assert evaluate_relevance([_chunk(1.0)], threshold=1.0).is_relevant is True
    assert evaluate_relevance([_chunk(0.99)], threshold=1.0).is_relevant is False
