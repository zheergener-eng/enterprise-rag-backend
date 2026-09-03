"""Evaluation 专用知识库与隔离 VectorStore。

关键约定（对应已登记技术债「verify_*.py 可能清默认 rag:index」）：
Evaluation 必须使用独立的 Redis index / key prefix，绝不 drop 生产默认 ``rag:index``。
运行 evaluation 后只清理自己的 eval 数据，不动任何生产数据。

本模块只负责「构造隔离 store + 填充评测知识库 + 提供两阶段检索辅助」，
复用 app 的 Embedder / Reranker（均与 Redis 无关、可安全共享），
但不复用 app 的 ``get_vector_store()`` 单例（其指向生产默认 index）。
"""
from __future__ import annotations

import time

from app.config import settings
from app.services.embeddings import get_embedder
from app.services.reranker import RerankedChunk, RetrievedChunk, rerank
from app.services.vector_store import VectorStore

# 隔离命名空间：与生产默认 index / prefix 不同，避免互相污染
EVAL_INDEX_NAME = "eval:rag:index"
EVAL_PREFIX = "eval:chunk:"

# 评测知识库：每个 (doc_id, title, text) 是一个独立 chunk（一个明确事实）。
# 用于支撑 >= 15 条 answerable 样本；hard_negative 与 irrelevant 样本的问题
# 都不在此知识库中（保证「相关但无答案」与「完全无关」两类负样本成立）。
KNOWLEDGE: list[tuple[str, str, str]] = [
    ("backup-time", "星河项目运维手册", "星河项目数据库每天凌晨 2:30 执行备份。"),
    ("backup-retention", "星河项目运维手册", "星河项目数据库备份文件保留 14 天。"),
    ("backup-strategy", "星河项目运维手册", "数据库采用每日增量备份策略，全量备份每周一次。"),
    ("redis-deploy", "Redis 部署手册", "Redis Stack 采用单节点部署，监听默认端口 6379。"),
    ("redis-dir", "Redis 部署手册", "Redis 数据目录挂载在本地磁盘 /data/redis。"),
    ("server-deployment", "服务器运维手册", "生产服务器部署在华东机房，共 8 台节点。"),
    ("server-fan", "服务器运维手册", "服务器 CPU 温度过高时应检查风扇。"),
    ("server-memory", "服务器运维手册", "服务器内存使用率超过 90% 时应重启相关服务。"),
    ("auth-sso", "权限与认证手册", "内部系统统一使用 SSO 单点登录，账号由 IT 部门分配。"),
    ("network-egress", "网络运维手册", "生产环境出口 IP 为 203.0.113.10，白名单在此配置。"),
    ("monitor", "监控运维手册", "监控系统使用 Prometheus 采集指标，告警通过企业微信推送。"),
    ("log-retention", "日志运维手册", "应用日志保留 30 天，之后自动归档到对象存储。"),
    ("db-version", "数据库运维手册", "生产数据库使用 MySQL 8.0 版本。"),
    ("backup-drill", "星河项目运维手册", "每季度进行一次备份恢复演练，记录在运维台账。"),
    ("lb", "网络运维手册", "前端使用 Nginx 负载均衡，共 2 台入口节点。"),
]


def build_store() -> VectorStore:
    """构建并填充隔离的 Evaluation VectorStore（drop+ensure 只作用于 eval index）。"""
    store = VectorStore(
        redis_url=settings.redis_url,
        index_name=EVAL_INDEX_NAME,
        dim=settings.embedding_dim,
        prefix=EVAL_PREFIX,
    )
    store.drop_index()
    store.ensure_index()
    embedder = get_embedder()
    for doc_id, title, text in KNOWLEDGE:
        vec = embedder.embed_documents([text])
        store.add_document(doc_id, [text], vec, title=title)
    return store


def cleanup_store(store: VectorStore) -> None:
    """只清理 eval 自己的数据（drop eval index），不动生产默认 index。"""
    store.drop_index()


def eval_recall(
    query: str, store: VectorStore, recall_top_n: int | None = None
) -> tuple[list[dict], float]:
    """对隔离 store 做单阶段向量召回，返回 (recall 结果 list[dict], 耗时秒)。"""
    n = recall_top_n if recall_top_n is not None else settings.recall_top_n
    qv = get_embedder().embed_query(query)
    t0 = time.perf_counter()
    recalled = store.search(qv, top_k=n)
    return recalled, time.perf_counter() - t0


def eval_rerank(
    query: str, candidates: list[RetrievedChunk], rerank_top_k: int | None = None
) -> tuple[list[RerankedChunk], float]:
    """对召回候选做 Rerank，返回 (RerankedChunk 列表, 耗时秒)。"""
    k = rerank_top_k if rerank_top_k is not None else settings.rerank_top_k
    t0 = time.perf_counter()
    reranked = rerank(query, candidates, top_k=k)
    return reranked, time.perf_counter() - t0


def eval_retrieve(
    query: str,
    store: VectorStore,
    recall_top_n: int | None = None,
    rerank_top_k: int | None = None,
) -> tuple[list[dict], list[RerankedChunk], float, float]:
    """对隔离 store 做完整两阶段检索。

    Returns:
        (recalled: list[dict], reranked: list[RerankedChunk],
         recall_latency: float, rerank_latency: float)
    """
    recalled, recall_latency = eval_recall(query, store, recall_top_n)
    candidates = [RetrievedChunk.from_dict(d) for d in recalled]
    reranked, rerank_latency = eval_rerank(query, candidates, rerank_top_k)
    return recalled, reranked, recall_latency, rerank_latency
