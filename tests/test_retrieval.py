"""Semantic Retrieval 阶段测试。

包含两类：
- 基于假 embedder / 假 store 的单元测试（快，验证 retrieve 服务逻辑）
- 基于真实 Redis + 真实 bge 模型的集成测试（KNN 排序、空库、Top-K 边界、语义检索链）
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.services.retrieval import retrieve, retrieve_with_rerank
from app.services.vector_store import VectorStore


DIM = settings.embedding_dim  # 512
RETRIEVAL_INDEX = "rag:test:retrieval"
RETRIEVAL_PREFIX = "retchunk:"


# --------------------------------------------------------------------------
# 假实现（单元测试用，不依赖真实 Redis / 模型）
# --------------------------------------------------------------------------

class FakeEmbedder:
    """记录 embed_query 调用，返回固定向量。"""

    def __init__(self, vector: list[float] | None = None):
        self.vector = vector or [0.1, 0.2, 0.3]
        self.query_calls: list[str] = []

    def embed_query(self, question: str) -> list[float]:
        self.query_calls.append(question)
        return self.vector


class FakeStore:
    """记录 search 调用，返回预设结果。"""

    def __init__(self, results: list[dict] | None = None):
        self.results = results or []
        self.search_calls: list[tuple] = []
        self.ensure_index_calls = 0

    def ensure_index(self) -> None:
        self.ensure_index_calls += 1

    def search(self, query_vector, top_k):
        self.search_calls.append((query_vector, top_k))
        return self.results


@pytest.fixture
def fake_services(monkeypatch):
    """将 retrieve 依赖的 get_embedder / get_vector_store 替换为假实现。"""
    embedder = FakeEmbedder()
    store = FakeStore()
    monkeypatch.setattr("app.services.retrieval.get_embedder", lambda: embedder)
    monkeypatch.setattr("app.services.retrieval.get_vector_store", lambda: store)
    return embedder, store


# --------------------------------------------------------------------------
# 单元测试：retrieve 服务逻辑
# --------------------------------------------------------------------------

def test_retrieve_empty_question_rejected(fake_services):
    """空问题 / 纯空白问题应被拒绝（ValueError）。"""
    for bad in ["", "   ", "\n\t  "]:
        with pytest.raises(ValueError):
            retrieve(bad)


def test_retrieve_uses_default_top_k(fake_services):
    """未显式传入 top_k 时，使用 settings.recall_top_n（召回数量）。"""
    _, store = fake_services
    retrieve("这是一个问题")
    assert store.search_calls[0][1] == settings.recall_top_n


def test_retrieve_passes_custom_top_k(fake_services):
    """显式传入 top_k 时，以传入值为准。"""
    _, store = fake_services
    retrieve("问题", top_k=7)
    assert store.search_calls[0][1] == 7


def test_retrieve_embeds_then_searches(fake_services):
    """调用链：先 embed_query(去空白后的问题)，再 search(向量, top_k)。"""
    embedder, store = fake_services
    retrieve("  这是问题  ")
    # 问题去除首尾空白后传给 embed_query
    assert embedder.query_calls == ["这是问题"]
    # search 使用 embed_query 返回的向量
    assert store.search_calls[0][0] == embedder.vector


def test_retrieve_returns_store_results(fake_services):
    """结果应原样返回 store.search 的输出。"""
    expected = [{"document_id": "d1", "text": "命中"}]
    fake_services[1].results = expected
    assert retrieve("问题") == expected


# --------------------------------------------------------------------------
# 两阶段检索（retrieve_with_rerank）单元测试：mock store + reranker
# --------------------------------------------------------------------------

def _recall_dict(doc_id: str, text: str, distance: float) -> dict:
    """构造一条 Redis 召回结果（字段与 vector_store.search 一致）。"""
    return {
        "document_id": doc_id,
        "chunk_id": f"{doc_id}:0",
        "index": 0,
        "text": text,
        "title": "t",
        "distance": distance,
        "similarity": 1.0 - distance,
    }


class FakeRerankerModel:
    """记录 (query, text) pair 调用，返回预设分数。"""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls: list[list[tuple[str, str]]] = []

    def score_pairs(self, pairs):
        self.calls.append(pairs)
        return self.scores


@pytest.fixture
def fake_rerank_services(monkeypatch):
    """mock 两阶段检索依赖：embedder + store + reranker 模型。"""
    embedder = FakeEmbedder()
    store = FakeStore()
    reranker = FakeRerankerModel([0.1, 0.9, 0.5])
    monkeypatch.setattr("app.services.retrieval.get_embedder", lambda: embedder)
    monkeypatch.setattr("app.services.retrieval.get_vector_store", lambda: store)
    monkeypatch.setattr("app.services.reranker.get_reranker", lambda: reranker)
    return embedder, store, reranker


def test_retrieve_with_rerank_recall_count(fake_rerank_services):
    """Redis 初始召回数量 = recall_top_n（传给 store.search 的 top_k）。"""
    _, store, _ = fake_rerank_services
    store.results = [_recall_dict("d1", "a", 0.1)]
    retrieve_with_rerank("问题", recall_top_n=7, rerank_top_k=3)
    assert store.search_calls[0][1] == 7


def test_retrieve_with_rerank_uses_defaults(fake_rerank_services):
    """未显式传值时，使用 settings.recall_top_n。"""
    _, store, _ = fake_rerank_services
    store.results = [_recall_dict("d1", "a", 0.1)]
    retrieve_with_rerank("问题")
    assert store.search_calls[0][1] == settings.recall_top_n


def test_retrieve_with_rerank_final_count_le_rerank_top_k(fake_rerank_services):
    """最终数量 <= rerank_top_k。"""
    _, store, _ = fake_rerank_services
    store.results = [
        _recall_dict("a", "A", 0.1),
        _recall_dict("b", "B", 0.2),
        _recall_dict("c", "C", 0.3),
    ]
    out = retrieve_with_rerank("问题", recall_top_n=5, rerank_top_k=2)
    assert len(out) == 2


@pytest.mark.parametrize("n,k", [(3, 5), (1, 2), (5, 6)])
def test_retrieve_with_rerank_rerank_gt_recall_rejected(fake_rerank_services, n, k):
    """rerank_top_k > recall_top_n → ValueError。"""
    with pytest.raises(ValueError):
        retrieve_with_rerank("问题", recall_top_n=n, rerank_top_k=k)


def test_retrieve_with_rerank_fewer_candidates_than_top_k(fake_rerank_services):
    """候选少于 rerank_top_k 时返回全部。"""
    _, store, _ = fake_rerank_services
    store.results = [_recall_dict("a", "A", 0.1), _recall_dict("b", "B", 0.2)]
    out = retrieve_with_rerank("问题", recall_top_n=10, rerank_top_k=5)
    assert len(out) == 2


def test_retrieve_with_rerank_preserves_vector_and_rerank_scores(fake_rerank_services):
    """vector_distance / vector_similarity / rerank_score 均保留。"""
    _, store, _ = fake_rerank_services
    store.results = [
        _recall_dict("a", "A", 0.1),
        _recall_dict("b", "B", 0.2),
        _recall_dict("c", "C", 0.3),
    ]
    out = retrieve_with_rerank("问题", recall_top_n=5, rerank_top_k=3)
    assert len(out) == 3
    for r in out:
        assert r.vector_distance is not None
        assert r.vector_similarity is not None
        assert r.rerank_score is not None
    # 向量距离与相似度关系仍一致
    assert all(abs(r.vector_similarity - (1 - r.vector_distance)) < 1e-6 for r in out)


def test_retrieve_with_rerank_sorted_by_score_with_ranks(fake_rerank_services):
    """按 rerank_score 降序，final_rank 正确，recall_rank 保留原召回位次。"""
    _, store, reranker = fake_rerank_services
    reranker.scores = [0.1, 0.9, 0.5]  # A=0.1, B=0.9, C=0.5
    store.results = [
        _recall_dict("a", "A", 0.1),
        _recall_dict("b", "B", 0.2),
        _recall_dict("c", "C", 0.3),
    ]
    out = retrieve_with_rerank("问题", recall_top_n=5, rerank_top_k=3)

    assert [r.document_id for r in out] == ["b", "c", "a"]  # 按 score 降序
    assert [r.final_rank for r in out] == [1, 2, 3]
    assert [r.rerank_score for r in out] == [0.9, 0.5, 0.1]
    # recall_rank 保留第一阶段位次（与重排后的顺序无关）
    assert {r.document_id: r.recall_rank for r in out} == {"b": 2, "c": 3, "a": 1}


def test_retrieve_with_rerank_uses_query_for_rerank(fake_rerank_services):
    """向量召回与 Rerank 均使用同一（去空白后的）query。"""
    _, store, reranker = fake_rerank_services
    store.results = [_recall_dict("a", "A", 0.1)]
    retrieve_with_rerank("  数据库备份保留多久？  ", recall_top_n=5, rerank_top_k=3)
    assert reranker.calls[0] == [("数据库备份保留多久？", "A")]


def test_retrieve_with_rerank_empty_recall(fake_rerank_services):
    """召回为空 → 返回空列表，且不调用 reranker。"""
    _, store, reranker = fake_rerank_services
    store.results = []
    assert retrieve_with_rerank("问题") == []
    assert reranker.calls == []


def test_retrieve_with_rerank_blank_query_rejected(fake_rerank_services):
    """query 空 / 仅空白 → ValueError。"""
    with pytest.raises(ValueError):
        retrieve_with_rerank("   ")


# --------------------------------------------------------------------------
# 真实集成测试（需要真实 Redis Stack 运行）
# --------------------------------------------------------------------------

@pytest.fixture
def retr_store():
    """function 级：真实 Redis 向量存储（独立索引 / 前缀）。

    每个测试前后 drop_index，保证各测试之间数据隔离，不受执行顺序影响。
    """
    s = VectorStore(
        redis_url=settings.redis_url,
        index_name=RETRIEVAL_INDEX,
        dim=DIM,
        prefix=RETRIEVAL_PREFIX,
    )
    s.drop_index()
    s.ensure_index()
    yield s
    s.drop_index()


def _unit_vec(*ones: int) -> np.ndarray:
    """构造 512 维单位向量，指定位置置 1。"""
    v = np.zeros(DIM, dtype=np.float32)
    for i in ones:
        v[i] = 1.0
    return v


@pytest.mark.integration
def test_search_returns_top_k_ordered(retr_store):
    """人工向量：KNN 按 COSINE 距离升序返回，metadata 正确对应。"""
    v0 = _unit_vec(0)                              # 与查询同向 → 距离 0
    v1 = _unit_vec(1)                              # 与查询正交 → 距离 1
    v2 = np.zeros(DIM, dtype=np.float32)
    v2[0] = v2[1] = 0.70710678                     # 归一化对角 → 距离约 0.293

    retr_store.add_document("knn-0", ["苹果是水果"], [v0.tolist()], title="t0")
    retr_store.add_document("knn-1", ["香蕉"], [v1.tolist()], title="t1")
    retr_store.add_document("knn-2", ["混合向量"], [v2.tolist()], title="t2")

    q = _unit_vec(0)
    results = retr_store.search(q.tolist(), top_k=3)

    assert len(results) == 3
    assert results[0]["document_id"] == "knn-0"

    # 按距离升序（相关性降序）
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-4)

    # metadata 正确对应
    top = results[0]
    assert top["chunk_id"] == "knn-0:0"
    assert top["index"] == 0
    assert top["text"] == "苹果是水果"
    assert top["title"] == "t0"

    # similarity = 1 - distance（语义清晰，不把距离误当相似度）
    for r in results:
        assert r["similarity"] == pytest.approx(1.0 - r["distance"], abs=1e-6)


@pytest.mark.integration
def test_search_top_k_greater_than_count(retr_store):
    """top_k 大于实际 chunk 数时，返回全部而非报错。"""
    v = _unit_vec(0)
    retr_store.add_document("gtcount-a", ["只有两条"], [v.tolist()])
    retr_store.add_document("gtcount-b", ["数据"], [v.tolist()])
    results = retr_store.search(v.tolist(), top_k=5)
    assert len(results) == 2


@pytest.mark.integration
def test_search_empty_kb():
    """空知识库：返回空列表，不报错。"""
    s = VectorStore(
        redis_url=settings.redis_url,
        index_name="rag:test:empty",
        dim=DIM,
        prefix="emptychunk:",
    )
    s.drop_index()
    s.ensure_index()
    try:
        q = _unit_vec(0)
        assert s.search(q.tolist(), top_k=3) == []
    finally:
        s.drop_index()


@pytest.mark.integration
def test_search_semantic_retrieval_chain(retr_store, real_embedder):
    """真实语义检索链：语义相关（措辞不同）的 chunk 应排在无关 chunk 之前。"""
    docs = {
        "semantic-py": "pip 是 Python 官方推荐的包安装工具，可以从 PyPI 下载依赖。",
        "semantic-fruit": "苹果富含维生素 C，有助于增强免疫力。",
        "semantic-river": "长江是中国最长的河流，流经多个省份。",
    }
    for doc_id, text in docs.items():
        vec = real_embedder.embed_documents([text])
        retr_store.add_document(doc_id, [text], vec, title=doc_id)

    # 问题与 "semantic-py" 语义相关，但措辞不同（无 pip/PyPI/包 等相同关键词）
    question = "怎么给 Python 项目装第三方依赖？"
    qv = real_embedder.embed_query(question)

    results = retr_store.search(qv, top_k=3)

    assert len(results) == 3
    assert results[0]["document_id"] == "semantic-py"
    assert results[0]["text"] == docs["semantic-py"]
    # 相关性降序：距离严格非递减
    ds = [r["distance"] for r in results]
    assert ds == sorted(ds)
