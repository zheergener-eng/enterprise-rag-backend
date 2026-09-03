"""Vector Recall / Rerank 离线评测入口。

只评测检索链路（不调 LLM）：对 answerable 样本计算
- Vector Recall 的 Hit@1 / Hit@3 / Hit@5、MRR、Recall@K；
- Rerank 前后正确 chunk rank 变化（MRR / Hit@1 / improved / unchanged / degraded）；
- 性能（recall / rerank 平均与 p50 / p95 延迟、候选数）。

用法：python -m evaluation.evaluate_retrieval
前置：Redis Stack 已运行；bge-small-zh-v1.5 与 bge-reranker-base 已下载。
"""
from __future__ import annotations

import time

from evaluation.dataset import EvalSample, load_dataset
from evaluation.knowledge import (
    build_store,
    cleanup_store,
    eval_retrieve,
)
from evaluation.metrics import (
    mean_hit_at_k,
    mean_reciprocal_rank,
    mean_recall_at_k,
    percentile,
)
from evaluation.report import write_json, write_markdown

DATASET_PATH = "evaluation/dataset.json"
REPORT_DIR = "evaluation/reports"


def _first_rank(ids: list[str], expected: set[str]) -> int | None:
    """返回第一个命中的 expected chunk 的 1 起位次；无命中返回 None。"""
    for i, cid in enumerate(ids, start=1):
        if cid in expected:
            return i
    return None


def run_retrieval_eval(
    store,
    samples: list[EvalSample],
    recall_top_n: int,
    rerank_top_k: int,
) -> dict:
    """对 answerable 样本做两阶段检索评测，返回结构化指标。"""
    answerable = [s for s in samples if s.answerable]

    recall_ranks: list[int | None] = []
    rerank_ranks: list[int | None] = []
    recall_at_k_samples: list[tuple[list[str], list[str]]] = []
    rerank_at_k_samples: list[tuple[list[str], list[str]]] = []

    recall_latencies: list[float] = []
    rerank_latencies: list[float] = []
    candidate_counts: list[int] = []

    improved = unchanged = degraded = 0

    for s in answerable:
        expected = set(s.expected_chunk_ids)

        recalled, reranked, rlat, rklat = eval_retrieve(
            s.question, store, recall_top_n=recall_top_n, rerank_top_k=rerank_top_k
        )
        recall_latencies.append(rlat)
        rerank_latencies.append(rklat)
        candidate_counts.append(len(recalled))

        recall_ids = [d["chunk_id"] for d in recalled]
        rerank_ids = [r.chunk_id for r in reranked]

        rr = _first_rank(recall_ids, expected)
        kr = _first_rank(rerank_ids, expected)
        recall_ranks.append(rr)
        rerank_ranks.append(kr)
        recall_at_k_samples.append((recall_ids, list(expected)))
        rerank_at_k_samples.append((rerank_ids, list(expected)))

        # improved / unchanged / degraded：仅对「正确 chunk 在召回中」的样本统计
        if rr is not None:
            if kr is not None and kr < rr:
                improved += 1
            elif kr is not None and kr == rr:
                unchanged += 1
            else:  # kr is None（被挤出 top-k）或 kr > rr
                degraded += 1

    vector_retrieval = {
        "answerable_count": len(answerable),
        "hit_at_1": round(mean_hit_at_k(recall_ranks, 1), 4),
        "hit_at_3": round(mean_hit_at_k(recall_ranks, 3), 4),
        "hit_at_5": round(mean_hit_at_k(recall_ranks, 5), 4),
        "mrr": round(mean_reciprocal_rank(recall_ranks), 4),
        "recall_at_1": round(mean_recall_at_k(recall_at_k_samples, 1), 4),
        "recall_at_3": round(mean_recall_at_k(recall_at_k_samples, 3), 4),
        "recall_at_5": round(mean_recall_at_k(recall_at_k_samples, 5), 4),
    }

    rerank_result = {
        "mrr_before": round(mean_reciprocal_rank(recall_ranks), 4),
        "mrr_after": round(mean_reciprocal_rank(rerank_ranks), 4),
        "hit1_before": round(mean_hit_at_k(recall_ranks, 1), 4),
        "hit1_after": round(mean_hit_at_k(rerank_ranks, 1), 4),
        "hit5_before": round(mean_hit_at_k(recall_ranks, 5), 4),
        "hit5_after": round(mean_hit_at_k(rerank_ranks, 5), 4),
        "improved_count": improved,
        "unchanged_count": unchanged,
        "degraded_count": degraded,
        "recall_miss_count": sum(1 for r in recall_ranks if r is None),
    }

    latency = {
        "avg_recall_seconds": round(sum(recall_latencies) / len(recall_latencies), 4)
        if recall_latencies
        else 0.0,
        "avg_rerank_seconds": round(sum(rerank_latencies) / len(rerank_latencies), 4)
        if rerank_latencies
        else 0.0,
        "p50_recall_seconds": round(percentile(recall_latencies, 50), 4),
        "p95_recall_seconds": round(percentile(recall_latencies, 95), 4),
        "p50_rerank_seconds": round(percentile(rerank_latencies, 50), 4),
        "p95_rerank_seconds": round(percentile(rerank_latencies, 95), 4),
        "avg_candidates": round(sum(candidate_counts) / len(candidate_counts), 2)
        if candidate_counts
        else 0.0,
        "recall_top_n": recall_top_n,
        "rerank_top_k": rerank_top_k,
    }

    return {
        "vector_retrieval": vector_retrieval,
        "rerank": rerank_result,
        "latency": latency,
    }


def main() -> None:
    from app.config import settings

    recall_top_n = settings.recall_top_n
    rerank_top_k = settings.rerank_top_k

    samples = load_dataset(DATASET_PATH)
    store = build_store()
    # 预热 cross-encoder（embedder 已在 build_store 加载），避免首次懒加载污染计时
    eval_retrieve("预热", store, recall_top_n=3, rerank_top_k=3)
    try:
        result = run_retrieval_eval(store, samples, recall_top_n, rerank_top_k)
        result.update(
            dataset_size=len(samples),
            recall_top_n=recall_top_n,
            rerank_top_k=rerank_top_k,
        )
        write_json(f"{REPORT_DIR}/retrieval.json", result)
        write_markdown(f"{REPORT_DIR}/retrieval.md", result)

        vr = result["vector_retrieval"]
        rr = result["rerank"]
        print("=" * 70)
        print("[Vector Retrieval]")
        print(f"  Hit@1={vr['hit_at_1']}  Hit@3={vr['hit_at_3']}  Hit@5={vr['hit_at_5']}  MRR={vr['mrr']}")
        print(f"  Recall@1={vr['recall_at_1']}  Recall@3={vr['recall_at_3']}  Recall@5={vr['recall_at_5']}")
        print("[Rerank]")
        print(f"  MRR before={rr['mrr_before']}  after={rr['mrr_after']}")
        print(f"  Hit@1 before={rr['hit1_before']}  after={rr['hit1_after']}")
        print(f"  improved={rr['improved_count']}  unchanged={rr['unchanged_count']}  "
              f"degraded={rr['degraded_count']}  recall_miss={rr['recall_miss_count']}")
        print(f"[Latency] avg_recall={result['latency']['avg_recall_seconds']}s  "
              f"avg_rerank={result['latency']['avg_rerank_seconds']}s")
        print(f"报告已写入 {REPORT_DIR}/retrieval.json / retrieval.md")
    finally:
        cleanup_store(store)


if __name__ == "__main__":
    main()
