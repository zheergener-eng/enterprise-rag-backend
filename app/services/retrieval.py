"""语义检索服务（含两阶段检索：向量召回 + Rerank 精排）。

两条清晰链路：

1. Baseline KNN（单阶段）：``retrieve(question)``
       question → embed_query(512 维) → VectorStore.search(KNN, COSINE) → Top-N chunks

2. 两阶段（Recall + Rerank）：``retrieve_with_rerank(query)``
       query → embed_query → Redis Recall Top-N → Cross-Encoder Rerank Top-K → final chunks

下游 RAG 阶段统一走两阶段链路；``retrieve`` 保留供 Baseline / Evaluation 对比。
"""
from __future__ import annotations

from app.config import settings
from app.services.embeddings import get_embedder
from app.services.reranker import RerankedChunk, RetrievedChunk, rerank
from app.services.vector_store import get_vector_store


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """对问题做单阶段 KNN 语义检索（Baseline），返回按相关性降序的 chunk 列表。

    Args:
        question: 用户问题（去除首尾空白后不得为空）。
        top_k: 返回最大条数（即向量召回数量）；缺省使用 settings.recall_top_n。

    Returns:
        结果列表，每项含 document_id / chunk_id / index / text / title /
        distance（COSINE 距离，越小越相关）/ similarity（越大越相关）。

    Raises:
        ValueError: question 为空或仅含空白（无意义检索）。
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty or blank")

    k = top_k if top_k is not None else settings.recall_top_n

    store = get_vector_store()
    store.ensure_index()  # 幂等：确保向量索引已就绪

    query_vector = get_embedder().embed_query(question)
    return store.search(query_vector, top_k=k)


def retrieve_with_rerank(
    query: str,
    recall_top_n: int | None = None,
    rerank_top_k: int | None = None,
) -> list[RerankedChunk]:
    """两阶段检索：向量召回 Top-N → Cross-Encoder Rerank Top-K。

    逻辑：
        query → embed_query → Redis search(recall_top_n) → 召回候选
        → Reranker.rerank(query, candidates, rerank_top_k) → 最终 RerankedChunk 列表

    Args:
        query: 检索 query（去除首尾空白后不得为空）。多轮场景由上层传入 rewritten query，
            因此向量召回与 Rerank 均使用同一 query。
        recall_top_n: Redis 向量召回数量；缺省 settings.recall_top_n（默认 10）。
        rerank_top_k: Rerank 后最终保留数量；缺省 settings.rerank_top_k（默认 3）。

    Returns:
        按 rerank_score 降序的 RerankedChunk 列表，长度 <= rerank_top_k。
        召回为空时返回空列表。

    Raises:
        ValueError: query 为空或仅空白；recall_top_n / rerank_top_k < 1；
            rerank_top_k > recall_top_n。
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty or blank")

    n = recall_top_n if recall_top_n is not None else settings.recall_top_n
    k = rerank_top_k if rerank_top_k is not None else settings.rerank_top_k

    if n < 1:
        raise ValueError("recall_top_n must be >= 1")
    if k < 1:
        raise ValueError("rerank_top_k must be >= 1")
    if k > n:
        raise ValueError("rerank_top_k must be <= recall_top_n")

    # 1. 向量召回（复用单阶段 retrieve：ensure_index + embed_query + KNN）
    recalled = retrieve(query, top_k=n)
    if not recalled:
        return []

    # 2. Cross-Encoder 精排
    candidates = [RetrievedChunk.from_dict(d) for d in recalled]
    return rerank(query, candidates, top_k=k)
