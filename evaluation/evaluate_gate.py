"""Relevance Gate 离线评测 + Threshold Sweep + 轻量 Answer Check。

用法：python -m evaluation.evaluate_gate
前置：Redis Stack 已运行；bge 模型已下载；.env 已配置 DEEPSEEK_API_KEY（answer check 需要）。

产出：evaluation/reports/latest.json + latest.md（组合完整报告）。

本阶段只观察、不修改正式 RERANK_RELEVANCE_THRESHOLD；threshold sweep 是离线扫描，
不写回 .env。Gate 分类分别统计 answerable / irrelevant / hard_negative，不混成一个总分。
"""
from __future__ import annotations

from app.config import settings
from app.services.llm import LLMError, get_llm_client
from app.services.rag import build_context, build_prompt
from app.services.relevance import evaluate_relevance

from evaluation.dataset import EvalSample, count_by_category, load_dataset
from evaluation.evaluate_retrieval import run_retrieval_eval
from evaluation.knowledge import build_store, cleanup_store, eval_retrieve
from evaluation.metrics import (
    confusion_counts,
    default_thresholds,
    recall,
    summarize_confusion,
    threshold_sweep,
)
from evaluation.report import write_json, write_markdown

DATASET_PATH = "evaluation/dataset.json"
REPORT_DIR = "evaluation/reports"


def _category_stats(rows: list[dict], category: str) -> dict:
    """单类别 Gate 表现：answerable 给出 pass_rate，负类给出 rejection_rate。"""
    cat = [r for r in rows if r["category"] == category]
    pred = [r["gate_decision"] for r in cat]
    exp = [r["expected_answerable"] for r in cat]
    counts = confusion_counts(pred, exp)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    if category == "answerable":
        return {"count": len(cat), "tp": tp, "fn": fn, "pass_rate": round(recall(tp, fn), 4)}
    return {
        "count": len(cat),
        "tn": tn,
        "fp": fp,
        "rejection_rate": round(tn / (tn + fp), 4) if (tn + fp) else 0.0,
    }


def run_gate_eval(
    store,
    samples: list[EvalSample],
    recall_top_n: int,
    rerank_top_k: int,
    threshold: float,
) -> dict:
    """对全部样本做 Relevance Gate 评测，返回结构化结果 + 观察明细。"""
    rows: list[dict] = []
    for s in samples:
        _, reranked, _, _ = eval_retrieve(
            s.question, store, recall_top_n=recall_top_n, rerank_top_k=rerank_top_k
        )
        decision = evaluate_relevance(reranked, threshold=threshold)
        top = reranked[0] if reranked else None
        rows.append(
            {
                "id": s.id,
                "question": s.question,
                "category": s.category,
                "top1_chunk": top.document_id if top else None,
                "top1_chunk_id": top.chunk_id if top else None,
                "vector_similarity": round(top.vector_similarity, 4) if top else None,
                "rerank_score": round(top.rerank_score, 4) if top else None,
                "gate_decision": decision.is_relevant,
                "reason": decision.reason,
                "expected_answerable": s.answerable,
            }
        )

    pred = [r["gate_decision"] for r in rows]
    exp = [r["expected_answerable"] for r in rows]
    overall = summarize_confusion(confusion_counts(pred, exp))

    by_category = {c: _category_stats(rows, c) for c in ("answerable", "irrelevant", "hard_negative")}

    false_positive_examples = [
        {
            "id": r["id"],
            "question": r["question"],
            "category": r["category"],
            "top1_chunk": r["top1_chunk"],
            "rerank_score": r["rerank_score"],
        }
        for r in rows
        if not r["expected_answerable"] and r["gate_decision"]
    ]
    false_negative_examples = [
        {
            "id": r["id"],
            "question": r["question"],
            "category": r["category"],
            "top1_chunk": r["top1_chunk"],
            "rerank_score": r["rerank_score"],
        }
        for r in rows
        if r["expected_answerable"] and not r["gate_decision"]
    ]

    return {
        "threshold": threshold,
        "overall": overall,
        "by_category": by_category,
        "observations": rows,
        "false_positive_examples": false_positive_examples,
        "false_negative_examples": false_negative_examples,
        # 供 threshold sweep 复用的 (top1_rerank_score, answerable) 序列
        "sweep_samples": [
            (r["rerank_score"], r["expected_answerable"]) for r in rows
        ],
    }


def _answer_contains(expected: str, answer: str) -> bool:
    """轻量 answer check：去掉所有空白后，判断关键事实是否为 answer 的子串。"""
    e = "".join(expected.split())
    a = "".join(answer.split())
    return bool(e) and e in a


def run_answer_eval(
    store,
    samples: list[EvalSample],
    recall_top_n: int,
    rerank_top_k: int,
    threshold: float,
) -> dict:
    """轻量 Answer Check：仅对含短事实 expected_answer 的 answerable 样本生成答案并检查。"""
    targets = [s for s in samples if s.answerable and s.expected_answer]
    passed = failed = skipped = 0
    per_sample: list[dict] = []
    llm = get_llm_client()
    for s in targets:
        _, reranked, _, _ = eval_retrieve(
            s.question, store, recall_top_n=recall_top_n, rerank_top_k=rerank_top_k
        )
        decision = evaluate_relevance(reranked, threshold=threshold)
        if not decision.is_relevant:
            per_sample.append({"id": s.id, "question": s.question, "status": "gate_rejected"})
            failed += 1
            continue
        try:
            answer = llm.generate(build_prompt(s.question, build_context(reranked)))
        except LLMError:
            per_sample.append({"id": s.id, "question": s.question, "status": "llm_error"})
            skipped += 1
            continue
        ok = _answer_contains(s.expected_answer, answer)
        per_sample.append(
            {"id": s.id, "question": s.question, "status": "pass" if ok else "fail",
             "expected": s.expected_answer}
        )
        passed += ok
        failed += (not ok)

    total = len(targets)
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "per_sample": per_sample,
    }


def main() -> None:
    recall_top_n = settings.recall_top_n
    rerank_top_k = settings.rerank_top_k
    threshold = settings.rerank_relevance_threshold  # 当前 provisional，不修改

    samples = load_dataset(DATASET_PATH)
    store = build_store()
    # 预热 cross-encoder，避免首次懒加载污染性能统计
    eval_retrieve("预热", store, recall_top_n=3, rerank_top_k=3)
    try:
        retrieval = run_retrieval_eval(store, samples, recall_top_n, rerank_top_k)
        gate = run_gate_eval(store, samples, recall_top_n, rerank_top_k, threshold)
        sweep = threshold_sweep(gate["sweep_samples"], default_thresholds())
        answer_check = run_answer_eval(store, samples, recall_top_n, rerank_top_k, threshold)

        report = {
            "dataset_size": len(samples),
            "categories": count_by_category(samples),
            "recall_top_n": recall_top_n,
            "rerank_top_k": rerank_top_k,
            "threshold": threshold,
            "vector_retrieval": retrieval["vector_retrieval"],
            "rerank": retrieval["rerank"],
            "latency": retrieval["latency"],
            "gate": {
                "threshold": gate["threshold"],
                "overall": gate["overall"],
                "by_category": gate["by_category"],
            },
            "threshold_sweep": sweep,
            "false_positive_examples": gate["false_positive_examples"],
            "false_negative_examples": gate["false_negative_examples"],
            "answer_check": answer_check,
        }

        write_json(f"{REPORT_DIR}/latest.json", report)
        write_markdown(f"{REPORT_DIR}/latest.md", report)

        overall = gate["overall"]
        print("=" * 70)
        print(f"[Gate @ threshold={threshold}]")
        print(f"  Accuracy={overall['accuracy']}  Precision={overall['precision']}  "
              f"Recall={overall['recall']}  F1={overall['f1']}")
        print(f"  FPR={overall['fpr']}  FNR={overall['fnr']}")
        for cat, info in gate["by_category"].items():
            print(f"  {cat}: {info}")
        print(f"[False Positive 例数] {len(gate['false_positive_examples'])}  "
              f"[False Negative 例数] {len(gate['false_negative_examples'])}")
        print(f"[Answer Check] {answer_check['passed']}/{answer_check['total']} "
              f"pass（skipped={answer_check['skipped']}）")
        print(f"报告已写入 {REPORT_DIR}/latest.json / latest.md")
    finally:
        cleanup_store(store)


if __name__ == "__main__":
    main()
