"""Relevance Gate（相关性门槛）服务。

位于两阶段检索（向量召回 + Rerank）之后、RAG 回答之前，判断「当前检索结果
是否足以作为知识库依据作答」。

架构位置（本阶段接入 RAG 主链路）：

    Query → Recall Top-N → Rerank Top-K → Relevance Gate → relevant / insufficient
                                                              ↓              ↓
                                                            RAG         no-answer

职责单一：给定 Rerank 后的候选（已按 rerank_score 降序），返回结构化决策
「relevant（足以作答）/ insufficient（不足以作答）」。

关键约定：
- 只依据 top-1 候选的 rerank_score 与配置阈值比较，不做任何启发式、不调用 LLM；
- 阈值来自 settings.rerank_relevance_threshold（配置化），本模块不硬编码具体数值；
- rerank_score 是 Cross-Encoder sigmoid 输出，仅表示 query-chunk 语义相关性，
  **不是概率**，不能解释为「置信度」；默认阈值 0.8 经 Evaluation v2 校准，
  为 benchmark-dependent 的工程折中值（见 evaluation/reports/evaluation_v2.md 的
  「Relevance Threshold Selection」章节）；
- 候选为空 → 直接 insufficient（reason="no_candidates"），不调用 LLM 判断；
- 本模块不改写 Query、不检索、不生成回答，也不返回低相关 chunks 作为依据。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.services.reranker import RerankedChunk


@dataclass(frozen=True)
class RelevanceResult:
    """Relevance Gate 的结构化决策结果。

    Attributes:
        is_relevant: 是否足以基于知识库作答。True=relevant（进入 RAG），
            False=insufficient（走 no-answer 路径）。
        top_score: top-1 候选的 rerank_score；候选为空时为 None。
        threshold: 本次判定使用的阈值（来自配置或调用方显式传入）。
        reason: 决策原因（稳定字符串，供 logging / evaluation 使用）：
            - ``no_candidates``：候选为空，知识库无任何召回；
            - ``top_score_meets_threshold``：top-1 分数 >= 阈值；
            - ``top_score_below_threshold``：top-1 分数 < 阈值。
    """

    is_relevant: bool
    top_score: float | None
    threshold: float
    reason: str


# 决策原因（稳定字符串，避免散落魔法字符串）
NO_CANDIDATES = "no_candidates"
MEETS_THRESHOLD = "top_score_meets_threshold"
BELOW_THRESHOLD = "top_score_below_threshold"


def evaluate_relevance(
    chunks: list[RerankedChunk],
    threshold: float | None = None,
) -> RelevanceResult:
    """对 Rerank 结果做相关性门槛判定。

    Args:
        chunks: 两阶段检索后的候选（须已按 rerank_score 降序，通常来自
            ``retrieve_with_rerank``）。为空表示知识库无任何召回。
        threshold: 判定阈值（[0, 1] 闭区间）；缺省使用
            ``settings.rerank_relevance_threshold``。

    Returns:
        RelevanceResult。候选为空时 top_score=None、is_relevant=False。

    Raises:
        ValueError: threshold 不在 [0, 1] 闭区间（sigmoid 输出的合法取值范围）。
    """
    if threshold is None:
        threshold = settings.rerank_relevance_threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1], got {threshold!r}")

    if not chunks:
        return RelevanceResult(False, None, threshold, NO_CANDIDATES)

    # chunks 已按 rerank_score 降序，top-1 即最高相关候选
    top_score = chunks[0].rerank_score
    if top_score >= threshold:
        return RelevanceResult(True, top_score, threshold, MEETS_THRESHOLD)
    return RelevanceResult(False, top_score, threshold, BELOW_THRESHOLD)
