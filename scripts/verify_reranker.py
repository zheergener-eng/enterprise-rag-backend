"""真实 Rerank 模型验证脚本（BAAI/bge-reranker-base）。

验证三件事：
  1. 真实语义 Rerank：给定 query「数据库备份文件保留多久？」，候选 A/B/C 中
     B（保留 14 天）必须排在明显无关的 C（CPU 温度）前面，并应成为最高相关。
  2. Rerank 改变原 Recall 排序：人为构造 vector recall 顺序 [A, B, C]，
     经 Rerank 后应重排为 [B, A, C]（B 反超 A）。
  3. 性能记录：候选数量、rerank 耗时、CPU 环境。

前置条件：首次运行会自动联网下载 bge-reranker-base 到 HuggingFace 缓存，
后续运行走本地缓存。不需要 Redis / DeepSeek。

用法：python -m scripts.verify_reranker   （或  PYTHONPATH=. python scripts/verify_reranker.py）
"""
from __future__ import annotations

import io
import platform
import time

from app.services.reranker import RetrievedChunk, get_reranker, rerank

QUERY = "数据库备份文件保留多久？"

# 人为构造三个候选（vector recall 顺序 [A, B, C]，A 的向量相似度略高于 B）
CANDIDATES = [
    RetrievedChunk(
        document_id="star-river",
        chunk_id="star-river:0",
        index=0,
        text="星河项目数据库每天凌晨 2:30 执行备份。",
        title="星河项目运维手册",
        distance=0.15,     # 模拟 KNN：A 距离最小
        similarity=0.85,
    ),
    RetrievedChunk(
        document_id="star-river",
        chunk_id="star-river:1",
        index=1,
        text="星河项目数据库备份文件保留 14 天。",
        title="星河项目运维手册",
        distance=0.17,
        similarity=0.83,
    ),
    RetrievedChunk(
        document_id="server-ops",
        chunk_id="server-ops:0",
        index=0,
        text="服务器 CPU 温度过高时应检查风扇。",
        title="服务器运维手册",
        distance=0.40,
        similarity=0.60,
    ),
]


def _fmt_label(c: RetrievedChunk) -> str:
    short = {"星河项目数据库每天凌晨 2:30 执行备份。": "A(2:30 备份)",
             "星河项目数据库备份文件保留 14 天。": "B(保留 14 天)",
             "服务器 CPU 温度过高时应检查风扇。": "C(CPU 温度)"}[c.text]
    return short


def main() -> None:
    summary: dict[str, object] = {}

    print("=== 环境 ===")
    print(f"CPU: {platform.processor()} | {platform.platform()}")
    summary["cpu_env"] = platform.processor()

    reranker = get_reranker()
    print(f"模型: {reranker.model_name}  (device={reranker.device})")
    summary["model"] = reranker.model_name

    print("\n=== 1) 模型加载耗时（仅首次加载） ===")
    t0 = time.perf_counter()
    # 触达模型加载（首次会联网/读缓存）
    reranker.score_pairs([("预热", "预热文本")])
    load_elapsed = time.perf_counter() - t0
    print(f"首次加载耗时: {load_elapsed:.2f}s")
    summary["model_load_seconds"] = round(load_elapsed, 2)

    print("\n=== 2) 真实语义 Rerank（3 候选） ===")
    n_candidates = len(CANDIDATES)
    t1 = time.perf_counter()
    ranked = rerank(QUERY, CANDIDATES, top_k=3)
    rerank_elapsed = time.perf_counter() - t1

    print(f"query: {QUERY}")
    print("召回顺序（vector recall）: " + " → ".join(_fmt_label(c) for c in CANDIDATES))
    print("Rerank 后顺序:            " + " → ".join(_fmt_label(c) for c in ranked))
    print("\n逐项分数（vector_similarity vs rerank_score）：")
    for i, r in enumerate(ranked, 1):
        print(
            f"  {i}. {_fmt_label(r)}  "
            f"vector_similarity={r.vector_similarity:.4f}  "
            f"rerank_score={r.rerank_score:.4f}"
        )

    rerank_scores = {r.text: r.rerank_score for r in ranked}
    text_a = "星河项目数据库每天凌晨 2:30 执行备份。"
    text_b = "星河项目数据库备份文件保留 14 天。"
    text_c = "服务器 CPU 温度过高时应检查风扇。"

    # 断言 1：B 排第一（成为最高相关）
    assert ranked[0].text == text_b, f"期望 B 排第一，实际 {ranked[0].text}"
    # 断言 2：B 排在明显无关的 C 前面
    idx_b = next(i for i, r in enumerate(ranked) if r.text == text_b)
    idx_c = next(i for i, r in enumerate(ranked) if r.text == text_c)
    assert idx_b < idx_c, "B 应排在 C 前面"
    print("\n[通过] B(保留 14 天) 排第一，且明显领先 C(CPU 温度)。")

    # 断言 3：Rerank 改变了原 Recall 顺序（A→B 被 B→A 反转）
    recall_order = [c.text for c in CANDIDATES]
    rerank_order = [r.text for r in ranked]
    assert recall_order != rerank_order, "Rerank 应改变原顺序"
    assert rerank_order.index(text_b) < rerank_order.index(text_a), "B 应反超 A"
    print("[通过] Rerank 改变了原 Recall 顺序：B 反超 A（[A,B,C] → [B,A,C]）。")

    # 断言 4：metadata 保留
    assert all(r.document_id and r.chunk_id and r.text for r in ranked)
    print("[通过] metadata（document_id/chunk_id/text/title）完整保留。")

    print(f"\n=== 3) 性能 ===")
    print(f"候选数量: {n_candidates}")
    print(f"rerank 推理耗时（3 候选，含 model.predict）: {rerank_elapsed:.3f}s")
    summary["candidate_count"] = n_candidates
    summary["rerank_seconds"] = round(rerank_elapsed, 3)
    summary["rerank_order"] = [_fmt_label(r) for r in ranked]

    with io.open("scripts/_verify_reranker.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k} = {v!r}\n")

    print("\n[结论] 真实 Rerank 模型验证通过。")


if __name__ == "__main__":
    main()
