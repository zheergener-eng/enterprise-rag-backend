"""真实两阶段检索（向量召回 + Rerank）验证脚本。

验证：
  1. 单轮两阶段：query「数据库备份文件保留多久？」下，Redis Recall Top-N 与
     Rerank Top-K 的排序对比，最终 RAG 使用「保留 14 天」chunk 排在首位。
  2. 多轮同 session 两轮：第一轮「什么时候备份」→ 2:30；第二轮「那保留多久？」→ 14 天，
     验证 rewrite / recall / rerank 链路正确、最终回答 14 天、Session 写回不变。
  3. 性能：recall_top_n / rerank_top_k / Redis recall 耗时 / rerank 耗时 / 总 pipeline 耗时。

前置条件：Redis Stack 已运行；.env 已配置 DEEPSEEK_API_KEY；
          已下载 bge-small-zh-v1.5 与 bge-reranker-base。

用法：python -m scripts.verify_two_stage
"""
from __future__ import annotations

import io
import time

from app.services.embeddings import get_embedder
from app.services.query_rewrite import rewrite
from app.services.rag import answer_with_session
from app.services.reranker import RetrievedChunk, rerank
from app.services.retrieval import retrieve, retrieve_with_rerank
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

# 构造知识库：5 个短 chunk，其中「保留 14 天」是唯一真正回答「多久」的 chunk；
# 另有「近逐字复述 query、但不给答案」的干扰 chunk（含「数据库备份文件保留多久」原文），
# 用于在向量召回阶段把正确答案顶下去，从而验证 Rerank 是否能把正确答案拉回首位。
DOCS = {
    "backup-time": ("星河项目运维手册", "星河项目数据库每天凌晨 2:30 执行备份。"),
    "backup-retention": ("星河项目运维手册", "星河项目数据库备份文件保留 14 天。"),
    "backup-strategy": ("星河项目运维手册", "数据库采用每日增量备份策略，全量备份每周一次。"),
    "backup-duration-distractor": (
        "星河项目运维手册",
        "关于数据库备份文件保留多久，请参考运维手册中的备份策略章节。",
    ),
    "server-fan": ("服务器运维手册", "服务器 CPU 温度过高时应检查风扇。"),
}

RETENTION_ID = "backup-retention"


def _seed() -> None:
    embedder = get_embedder()
    store = _build_store()
    store.drop_index()
    store.ensure_index()
    for doc_id, (title, text) in DOCS.items():
        vec = embedder.embed_documents([text])
        store.add_document(doc_id, [text], vec, title=title)


def main() -> None:
    summary: dict[str, object] = {}
    recall_top_n = 10
    rerank_top_k = 3

    _seed()

    # 预热 cross-encoder（embedder 已在 _seed 中加载），避免首次懒加载污染计时。
    retrieve_with_rerank("预热", recall_top_n=3, rerank_top_k=3)

    # ======================================================================
    # 1) 单轮两阶段：Recall 顺序 vs Rerank 顺序
    # ======================================================================
    q1 = "数据库备份文件保留多久？"
    print("=" * 70)
    print(f"[单轮两阶段] query: {q1}")
    print(f"recall_top_n={recall_top_n}  rerank_top_k={rerank_top_k}")

    # --- Redis 召回（Baseline KNN）---
    t0 = time.perf_counter()
    recalled = retrieve(q1, top_k=recall_top_n)
    recall_elapsed = time.perf_counter() - t0

    # --- Rerank（单独计时）---
    candidates = [RetrievedChunk.from_dict(d) for d in recalled]
    t1 = time.perf_counter()
    reranked_only = rerank(q1, candidates, top_k=rerank_top_k)
    rerank_elapsed = time.perf_counter() - t1

    # --- 总 pipeline（复用正式入口）---
    t2 = time.perf_counter()
    final = retrieve_with_rerank(q1, recall_top_n=recall_top_n, rerank_top_k=rerank_top_k)
    total_elapsed = time.perf_counter() - t2

    print("\nRedis Recall Top-N（按 vector_similarity 降序，即 KNN 原顺序）：")
    for i, d in enumerate(recalled, 1):
        print(f"  {i}. {d['document_id']:<18} similarity={d['similarity']:.4f}  {d['text']}")
    print("\nRerank Top-K（按 rerank_score 降序）：")
    for r in reranked_only:
        print(
            f"  {r.final_rank}. {r.document_id:<18} "
            f"rerank={r.rerank_score:.4f}  recall_rank={r.recall_rank}  {r.text}"
        )

    # 关键断言：最终 RAG 使用「保留 14 天」chunk 排在首位
    assert final[0].document_id == RETENTION_ID, (
        f"期望「保留 14 天」排第一，实际 {final[0].document_id}"
    )
    assert final[0].rerank_score == max(r.rerank_score for r in final)
    print(f"\n[通过] 最终 RAG 使用「保留 14 天」chunk 排在首位（final_rank=1）。")

    recall_order = [d["document_id"] for d in recalled]
    rerank_order = [r.document_id for r in reranked_only]
    changed = recall_order != rerank_order
    print(f"[信息] Recall 顺序 == Rerank 顺序：{not changed}"
          f"（{'Rerank 改变了原排序' if changed else '本次 KNN 已较优'}）")

    summary.update(
        recall_top_n=recall_top_n,
        rerank_top_k=rerank_top_k,
        recall_order=recall_order,
        rerank_order=rerank_order,
        recall_seconds=round(recall_elapsed, 4),
        rerank_seconds=round(rerank_elapsed, 4),
        total_pipeline_seconds=round(total_elapsed, 4),
    )

    # ======================================================================
    # 2) 多轮同 session：rewrite → recall → rerank → answer
    # ======================================================================
    session_store = get_session_store()
    sid = "verify:two-stage-multi"
    try:
        print("\n" + "=" * 70)
        print("[多轮] 同 session 两轮（走两阶段检索）")

        # 第一轮
        turn1 = "星河项目每天什么时候进行数据库备份？"
        r1 = answer_with_session(sid, turn1)
        print(f"\n[第一轮] {turn1}")
        print(f"  - answer: {r1.answer}")
        assert "2:30" in r1.answer
        summary["turn1_answer"] = r1.answer

        # 第二轮：先独立展示 rewrite + 两阶段检索结果，再走正式 answer_with_session
        turn2 = "那保留多久？"
        history_before = session_store.get_recent_history(sid)
        rr = rewrite(history_before, turn2)
        print(f"\n[第二轮 rewrite] {turn2}")
        print(f"  - rewritten_query: {rr.query!r}  used_llm={rr.used_llm} fallback={rr.fallback}")
        assert rr.used_llm is True

        reranked2 = retrieve_with_rerank(rr.query, recall_top_n=recall_top_n, rerank_top_k=rerank_top_k)
        print(f"  - Rerank Top-K（rerank 后顺序）：")
        for r in reranked2:
            print(f"    {r.final_rank}. {r.document_id:<18} rerank={r.rerank_score:.4f}")
        assert reranked2[0].document_id == RETENTION_ID, "rerank 后「保留 14 天」应排在前列"
        summary["turn2_rewritten_query"] = rr.query
        summary["turn2_rerank_order"] = [r.document_id for r in reranked2]

        r2 = answer_with_session(sid, turn2)
        print(f"  - answer: {r2.answer}")
        assert "14" in r2.answer
        summary["turn2_answer"] = r2.answer

        # Session 写回：user1/assistant1/user2/assistant2，user2 是原始问题
        history = session_store.get_history(sid)
        print("\n[写回] session 历史：")
        for m in history:
            print(f"  - {m['role']}: {m['content']}")
        assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
        assert history[2]["content"] == turn2
        summary["session_roles"] = [m["role"] for m in history]

        print("\n[结论] 真实两阶段检索（单轮 + 多轮）验证通过。")
    finally:
        session_store.clear_session(sid)
        _build_store().drop_index()
        with io.open("scripts/_verify_two_stage.txt", "w", encoding="utf-8") as f:
            for k, v in summary.items():
                f.write(f"{k} = {v!r}\n")


if __name__ == "__main__":
    main()
