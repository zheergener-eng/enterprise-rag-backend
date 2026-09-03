"""Rerank（重排 / 精排）服务。

在 Semantic Retrieval 的 KNN 召回结果之后，用 Cross-Encoder 对 (query, chunk) 做
逐对语义相关性打分，按 rerank_score 降序重排，截取 Top-K。

架构位置（本阶段独立实现，尚未接入 RAG 主链路 / API）：

    Query → Embedding → Redis KNN 召回 Top-N → Rerank 重排 Top-K → RAG

本模块只负责「已召回候选里谁更相关」，不做：
- Relevance Gate / Score Threshold（判断结果是否值得回答）——留待下一阶段；
- 重新计算 chunk embedding（复用召回结果里的向量距离）；
- 检索、Query Rewrite、Session、Prompt、Streaming、API。

指标区分（避免混淆）：
- vector distance / similarity：来自 Redis COSINE 检索，衡量「向量空间接近程度」；
- rerank_score：来自 Cross-Encoder，衡量「query 与 chunk 逐字语义相关性」。
二者是不同指标。输出同时保留 vector_distance / vector_similarity / rerank_score，
供后续 Evaluation 比较两阶段排序变化。

模型：默认 BAAI/bge-reranker-base（CrossEncoder），CPU 运行，懒加载、
进程内只初始化一次，首次联网下载，后续走本地缓存。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import settings


@dataclass(frozen=True)
class RetrievedChunk:
    """来自 Semantic Retrieval 的召回候选（Rerank 的输入）。

    Attributes:
        document_id: 文档标识。
        chunk_id: 全局唯一 chunk 标识（如 ``{document_id}:{index}``）。
        index: 文档内片段序号。
        text: 片段正文（Rerank 用它与 query 构造 (query, text) pair）。
        title: 片段所属文档标题（可空）。
        distance: 向量 COSINE 距离（Redis 返回，越小越相关）。
        similarity: 余弦相似度 ``1 - distance``（越大越相关）。
    """

    document_id: str
    chunk_id: str
    index: int
    text: str
    title: str = ""
    distance: float = 0.0
    similarity: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievedChunk":
        """从 retrieval / vector_store 返回的 dict 构造（桥接现有 dict 召回结果）。

        现有 Semantic Retrieval 返回 dict，字段为 document_id / chunk_id / index /
        text / title / distance / similarity。这里显式做字段映射，避免在业务代码里
        散落重复的取值逻辑。
        """
        return cls(
            document_id=data["document_id"],
            chunk_id=data["chunk_id"],
            index=int(data["index"]),
            text=data["text"],
            title=data.get("title") or "",
            distance=float(data.get("distance", 0.0)),
            similarity=float(data.get("similarity", 0.0)),
        )


@dataclass(frozen=True)
class RerankedChunk:
    """Rerank 后的候选（Rerank 的输出）。

    保留原始 metadata，并新增 rerank_score；向量距离 / 相似度重命名为
    vector_distance / vector_similarity，与 rerank_score 明确区分开。
    同时保留 recall_rank（第一阶段向量召回的位次）与 final_rank（Rerank 后的位次），
    供后续 Evaluation 比较两阶段排序变化。

    Attributes:
        document_id / chunk_id / index / text / title: 原始候选 metadata，原样保留。
        vector_distance: 保留自输入的 distance（COSINE 距离，越小越相关）。
        vector_similarity: 保留自输入的 similarity（余弦相似度，越大越相关）。
        rerank_score: Cross-Encoder 打分（越大越相关）。
        recall_rank: 该候选在向量召回阶段的位次（1 起）。
        final_rank: 该候选在 Rerank 后的位次（1 起，按 rerank_score 降序）。
    """

    document_id: str
    chunk_id: str
    index: int
    text: str
    title: str
    vector_distance: float
    vector_similarity: float
    rerank_score: float
    recall_rank: int = 1
    final_rank: int = 1


class Reranker:
    """Cross-Encoder 精排封装（懒加载 + 进程内复用同一模型实例）。"""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None  # 懒加载：首次使用时才真正加载模型

    def _load_model(self):
        """加载 Cross-Encoder（仅首次调用时执行，后续复用同一实例）。"""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """对 (query, chunk_text) 列表逐对打分，返回等长的 float 列表。

        Args:
            pairs: ``(query, chunk_text)`` 列表，由调用方统一使用同一 query 构造。

        Returns:
            与 pairs 等长的相关性分数列表（越大越相关）。
        """
        model = self._load_model()
        scores = model.predict(pairs)
        return [float(s) for s in scores]


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RerankedChunk]:
    """对召回候选做精排：按 rerank_score 降序重排，截取 Top-K。

    Args:
        query: 用户问题 / 检索 query（去除首尾空白后不得为空）。
        candidates: Semantic Retrieval 召回候选（RetrievedChunk 列表）。
        top_k: 重排后返回的最大条数，必须 >= 1。

    Returns:
        按 rerank_score 从高到低排列的 Top-K 结果。候选为空时返回空列表；
        top_k 大于候选数时返回全部。

    Raises:
        ValueError: query 为空或仅空白；top_k <= 0。
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty or blank")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    if not candidates:
        return []

    pairs = [(query, c.text) for c in candidates]
    scores = get_reranker().score_pairs(pairs)

    # 保留召回位次（recall_rank），按 rerank_score 降序后赋予最终位次（final_rank）
    scored = [
        (score, recall_pos, c)
        for recall_pos, (c, score) in enumerate(zip(candidates, scores), start=1)
    ]
    scored.sort(key=lambda t: t[0], reverse=True)

    ranked = [
        RerankedChunk(
            document_id=c.document_id,
            chunk_id=c.chunk_id,
            index=c.index,
            text=c.text,
            title=c.title,
            vector_distance=c.distance,
            vector_similarity=c.similarity,
            rerank_score=score,
            recall_rank=recall_pos,
            final_rank=final_pos,
        )
        for final_pos, (score, recall_pos, c) in enumerate(scored, start=1)
    ]
    return ranked[:top_k]


@lru_cache
def get_reranker() -> Reranker:
    """返回进程级单例 Reranker（模型只初始化一次）。"""
    return Reranker(model_name=settings.reranker_model)


# 便捷单例：`from app.services.reranker import reranker`
reranker = get_reranker()
