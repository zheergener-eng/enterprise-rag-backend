"""Evaluation 指标计算（纯函数，无 Redis / 模型 / LLM 依赖）。

包含检索 / 排序 / Gate 评测所需的全部指标：
- Hit@K、Recall@K、MRR（检索 / 重排）；
- 混淆矩阵（TP/FP/TN/FN）与 Accuracy / Precision / Recall / F1 / FPR / FNR；
- Threshold Sweep（离线扫描多个阈值）；
- 简单分位数（p50 / p95 性能统计）。
"""
from __future__ import annotations

from statistics import mean


def reciprocal_rank(rank: int | None) -> float:
    """Reciprocal Rank：rank 1→1.0，2→0.5，3→0.333…；未命中(None)→0。"""
    return 1.0 / rank if rank else 0.0


def hit_at_k(rank: int | None, k: int) -> bool:
    """正确 chunk 是否落在 Top-K（rank 为 1 起）。"""
    return rank is not None and 1 <= rank <= k


def mean_hit_at_k(ranks: list[int | None], k: int) -> float:
    """Hit@K：正确 chunk 出现在 Top-K 内的样本占比。空输入返回 0.0。"""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if hit_at_k(r, k)) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """MRR：各样本 reciprocal rank 的均值。空输入返回 0.0。"""
    if not ranks:
        return 0.0
    return mean(reciprocal_rank(r) for r in ranks)


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Recall@K：单个样本被 Top-K 召回的正确 chunk 占比（多 chunk 取均值）。

    ``expected_ids`` 为空时返回 0.0（调用方应跳过无预期 chunk 的样本）。
    """
    if not expected_ids:
        return 0.0
    topk = set(retrieved_ids[:k])
    return len(topk.intersection(expected_ids)) / len(expected_ids)


def mean_recall_at_k(
    samples: list[tuple[list[str], list[str]]], k: int
) -> float:
    """多样本 Recall@K 均值。``samples`` 为 (retrieved_ids, expected_ids) 列表。"""
    if not samples:
        return 0.0
    return mean(recall_at_k(retrieved, expected, k) for retrieved, expected in samples)


def confusion_counts(
    predicted_relevant: list[bool], expected_answerable: list[bool]
) -> dict[str, int]:
    """从预测与期望计算 TP / FP / TN / FN。"""
    tp = fp = tn = fn = 0
    for pred, exp in zip(predicted_relevant, expected_answerable):
        if pred and exp:
            tp += 1
        elif pred and not exp:
            fp += 1
        elif not pred and not exp:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1_score(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    total = tp + fp + tn + fn
    return (tp + tn) / total if total else 0.0


def false_positive_rate(fp: int, tn: int) -> float:
    return fp / (fp + tn) if (fp + tn) else 0.0


def false_negative_rate(fn: int, tp: int) -> float:
    return fn / (fn + tp) if (fn + tp) else 0.0


def summarize_confusion(counts: dict[str, int]) -> dict[str, float]:
    """把混淆计数汇总为 Accuracy / Precision / Recall / F1 / FPR / FNR。"""
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy(tp, fp, tn, fn), 4),
        "precision": round(precision(tp, fp), 4),
        "recall": round(recall(tp, fn), 4),
        "f1": round(f1_score(tp, fp, fn), 4),
        "fpr": round(false_positive_rate(fp, tn), 4),
        "fnr": round(false_negative_rate(fn, tp), 4),
    }


def threshold_sweep(
    samples: list[tuple[float | None, bool]], thresholds: list[float]
) -> list[dict]:
    """对每个阈值计算混淆矩阵摘要。

    Args:
        samples: ``(top1_rerank_score, expected_answerable)`` 列表；score 为 None
            表示检索为空（任何阈值下都判 insufficient）。
        thresholds: 待扫描的阈值列表（升序）。

    Returns:
        每个阈值一个 dict，含 threshold 与 summarize_confusion 的全部字段。
    """
    rows: list[dict] = []
    for t in thresholds:
        predicted = [s is not None and s >= t for s, _ in samples]
        expected = [a for _, a in samples]
        row = summarize_confusion(confusion_counts(predicted, expected))
        row["threshold"] = t
        rows.append(row)
    return rows


def percentile(values: list[float], p: float) -> float:
    """最近秩法分位数（p 为 0~100）。空列表返回 0.0。"""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, round((p / 100.0) * (len(s) - 1))))
    return s[idx]


def default_thresholds() -> list[float]:
    """离线扫描阈值：0.0~0.9 粗粒度 + 0.9~1.0 高分区加密观察。"""
    coarse = [round(i / 10, 2) for i in range(0, 10)]
    fine = [0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]
    return coarse + [t for t in fine if t > 0.9]
