"""真实 Relevance Gate 验证脚本。

验证目标：
  1. 观察真实 score 分布：对一组 relevant / irrelevant query 记录
     top-1 chunk、vector_similarity、rerank_score、gate 决策（score observation table）。
  2. Case A（同 session 指代）：Q1「什么时候备份？」→ 2:30；Q2「那保留多久？」
     → Query Rewrite 恢复指代 → relevant → 正常回答「14 天」。
  3. Case B（全新 session 模糊问题）：直接问「那保留多久？」无历史，
     rewrite 返回原问题，观察真实 gate 决策（不强行断言，如实记录）。

注意：本脚本只观察 score distribution，不宣称阈值最优；阈值是 provisional，
后续 Evaluation / Calibration 阶段根据正负样本重新确定。

前置条件：Redis Stack 已运行；.env 已配置 DEEPSEEK_API_KEY；
          已下载 bge-small-zh-v1.5 与 bge-reranker-base。

用法：python -m scripts.verify_relevance_gate
"""
from __future__ import annotations

import io

from app.config import settings
from app.services.embeddings import get_embedder
from app.services.rag import answer_with_session
from app.services.relevance import evaluate_relevance
from app.services.retrieval import retrieve_with_rerank
from app.services.session import get_session_store
from app.services.vector_store import VectorStore

# 验证脚本使用独立 index / prefix，绝不触碰生产默认 rag:index / chunk:。
VERIFY_INDEX = "verify:rag:index"
VERIFY_PREFIX = "verify:chunk:"


def _build_store() -> VectorStore:
    return VectorStore(
        redis_url=settings.redis_url,
        index_name=VERIFY_INDEX,
        dim=settings.embedding_dim,
        prefix=VERIFY_PREFIX,
    )

# 构造知识库：备份相关（含「近逐字复述、但不给答案」的干扰 chunk）+ 服务器/Redis 部署。
DOCS = {
    "backup-time": ("星河项目运维手册", "星河项目数据库每天凌晨 2:30 执行备份。"),
    "backup-retention": ("星河项目运维手册", "星河项目数据库备份文件保留 14 天。"),
    "backup-strategy": ("星河项目运维手册", "数据库采用每日增量备份策略，全量备份每周一次。"),
    "backup-duration-distractor": (
        "星河项目运维手册",
        "关于数据库备份文件保留多久，请参考运维手册中的备份策略章节。",
    ),
    "server-fan": ("服务器运维手册", "服务器 CPU 温度过高时应检查风扇。"),
    "server-deployment": ("服务器运维手册", "生产服务器部署在华东机房，共 8 台节点。"),
    "redis-deploy": ("Redis 部署手册", "Redis Stack 采用单节点部署，数据目录挂载本地磁盘，监听默认端口 6379。"),
}

RETENTION_ID = "backup-retention"

# 观察样本：relevant（知识库有答案）与 irrelevant / insufficient（知识库无答案）。
RELEVANT_QUERIES = [
    "数据库每天什么时候备份？",
    "数据库备份保留多久？",
    "Redis 如何部署？",
    "生产服务器部署在哪个机房？",
]

IRRELEVANT_QUERIES = [
    "公司食堂几点开门？",
    "明天天气怎么样？",
    "如何煮红烧肉？",
]


def _seed() -> None:
    embedder = get_embedder()
    store = _build_store()
    store.drop_index()
    store.ensure_index()
    for doc_id, (title, text) in DOCS.items():
        vec = embedder.embed_documents([text])
        store.add_document(doc_id, [text], vec, title=title)


def _observe(query: str, recall_top_n: int, rerank_top_k: int) -> dict:
    """对单个 query 做两阶段检索 + Relevance Gate，返回一行观察记录。"""
    chunks = retrieve_with_rerank(query, recall_top_n=recall_top_n, rerank_top_k=rerank_top_k)
    decision = evaluate_relevance(chunks)
    top = chunks[0] if chunks else None
    return {
        "query": query,
        "top_chunk": top.document_id if top else None,
        "vector_similarity": round(top.vector_similarity, 4) if top else None,
        "rerank_score": round(top.rerank_score, 4) if top else None,
        "is_relevant": decision.is_relevant,
        "reason": decision.reason,
        "threshold": decision.threshold,
    }


def main() -> None:
    summary: dict[str, object] = {}
    recall_top_n = 10
    rerank_top_k = 3
    threshold = settings.rerank_relevance_threshold

    _seed()
    # 预热 cross-encoder（embedder 已在 _seed 加载），避免首次懒加载影响观察。
    retrieve_with_rerank("预热", recall_top_n=3, rerank_top_k=3)

    print("=" * 70)
    print("[Relevance Gate] 真实 score observation table")
    print(f"threshold = {threshold}  (provisional)  recall_top_n={recall_top_n}  rerank_top_k={rerank_top_k}\n")

    rows: list[dict] = []
    print("Relevant queries：")
    for q in RELEVANT_QUERIES:
        row = _observe(q, recall_top_n, rerank_top_k)
        rows.append(row)
        print(
            f"  {row['is_relevant']!s:<5} rerank={row['rerank_score']!s:<8} "
            f"vec_sim={row['vector_similarity']!s:<8} {row['top_chunk']:<24} {q}"
        )

    print("\nIrrelevant / insufficient queries：")
    for q in IRRELEVANT_QUERIES:
        row = _observe(q, recall_top_n, rerank_top_k)
        rows.append(row)
        print(
            f"  {row['is_relevant']!s:<5} rerank={row['rerank_score']!s:<8} "
            f"vec_sim={row['vector_similarity']!s:<8} {row['top_chunk']:<24} {q}"
        )
    summary["threshold"] = threshold
    summary["observation_table"] = rows

    # ======================================================================
    # Case A：同 session 指代（应 relevant → 正常回答 14 天）
    # ======================================================================
    print("\n" + "=" * 70)
    print("[Case A] 同 session：Q1 什么时候备份 → Q2 那保留多久")
    sid_a = "verify:gate-caseA"
    try:
        turn1 = "星河项目每天什么时候进行数据库备份？"
        r1 = answer_with_session(sid_a, turn1)
        assert "2:30" in r1.answer
        print(f"  Q1: {turn1}")
        print(f"  A1: {r1.answer}")

        turn2 = "那保留多久？"
        r2 = answer_with_session(sid_a, turn2)
        print(f"  Q2: {turn2}")
        print(f"    rewritten_query={r2.rewritten_query!r}  used_llm={r2.rewrite_used_llm}")
        print(f"    relevance: is_relevant={r2.relevance.is_relevant} "
              f"top_score={r2.relevance.top_score} reason={r2.relevance.reason}")
        print(f"    answer: {r2.answer}")
        assert r2.relevance.is_relevant is True, "同 session 指代应判定 relevant"
        assert "14" in r2.answer, "同 session 指代应回答 14 天"
        summary["caseA"] = {
            "turn2_rewritten_query": r2.rewritten_query,
            "turn2_rewrite_used_llm": r2.rewrite_used_llm,
            "turn2_is_relevant": r2.relevance.is_relevant,
            "turn2_top_score": r2.relevance.top_score,
            "turn2_reason": r2.relevance.reason,
            "turn2_answer": r2.answer,
        }
        print("  [通过] Case A：指代恢复 → relevant → 回答 14 天。")
    finally:
        get_session_store().clear_session(sid_a)

    # ======================================================================
    # Case B：全新 session 模糊问题（如实观察，不强行断言）
    # ======================================================================
    print("\n" + "=" * 70)
    print("[Case B] 全新 session：直接问「那保留多久？」（无历史）")
    sid_b = "verify:gate-caseB"
    try:
        q = "那保留多久？"
        r = answer_with_session(sid_b, q)
        print(f"  query: {q}")
        print(f"    rewritten_query={r.rewritten_query!r}  used_llm={r.rewrite_used_llm}")
        print(f"    relevance: is_relevant={r.relevance.is_relevant} "
              f"top_score={r.relevance.top_score} reason={r.relevance.reason}")
        print(f"    answer: {r.answer}")
        summary["caseB"] = {
            "rewritten_query": r.rewritten_query,
            "rewrite_used_llm": r.rewrite_used_llm,
            "is_relevant": r.relevance.is_relevant,
            "top_score": r.relevance.top_score,
            "reason": r.relevance.reason,
            "answer": r.answer,
        }
        # 只断言「rewrite 未调用 LLM（无历史短路）」，不强制 gate 结果：
        # 若 gate 无法稳定区分该场景，属已知技术债，由 Evaluation/Calibration 解决。
        assert r.rewrite_used_llm is False, "无历史时 rewrite 应短路返回原问题"
        print("  [观察] Case B 结果已记录（rewrite 未调用 LLM，gate 决策见上）。")
    finally:
        get_session_store().clear_session(sid_b)
        _build_store().drop_index()
        with io.open("scripts/_verify_relevance_gate.txt", "w", encoding="utf-8") as f:
            f.write(f"threshold = {threshold!r}\n")
            f.write(f"recall_top_n = {recall_top_n!r}\n")
            f.write(f"rerank_top_k = {rerank_top_k!r}\n")
            f.write("observation_table = [\n")
            for row in rows:
                f.write(f"    {row!r},\n")
            f.write("]\n")
            f.write(f"caseA = {summary['caseA']!r}\n")
            f.write(f"caseB = {summary['caseB']!r}\n")


if __name__ == "__main__":
    main()
