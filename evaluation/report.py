"""Evaluation 报告输出（机器可读 JSON + 人类可读 Markdown）。

JSON 是唯一「机器可读」正式产物；Markdown 仅作快速浏览摘要，不追求复杂排版。
"""
from __future__ import annotations

import json
from pathlib import Path


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_json(path: str | Path, data: dict) -> None:
    """把报告 dict 写成 JSON 文件（UTF-8，缩进 2，保留中文原文）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _section(title: str) -> str:
    return f"\n## {title}\n"


def _kv(rows: list[tuple[str, object]]) -> str:
    return "\n".join(f"- **{k}**: {_fmt(v)}" for k, v in rows)


def write_markdown(path: str | Path, report: dict) -> None:
    """把报告 dict 渲染为 Markdown 摘要。"""
    lines: list[str] = ["# RAG Evaluation Report"]

    lines.append(_section("概览"))
    lines.append(
        _kv(
            [
                ("dataset_size", report.get("dataset_size", 0)),
                ("categories", report.get("categories", {})),
                ("recall_top_n", report.get("recall_top_n")),
                ("rerank_top_k", report.get("rerank_top_k")),
                ("gate_threshold", report.get("threshold")),
            ]
        )
    )

    vr = report.get("vector_retrieval") or {}
    lines.append(_section("Vector Retrieval"))
    lines.append(
        _kv(
            [
                ("Hit@1", vr.get("hit_at_1")),
                ("Hit@3", vr.get("hit_at_3")),
                ("Hit@5", vr.get("hit_at_5")),
                ("MRR", vr.get("mrr")),
                ("Recall@5", vr.get("recall_at_5")),
            ]
        )
    )

    rr = report.get("rerank") or {}
    lines.append(_section("Rerank"))
    lines.append(
        _kv(
            [
                ("MRR before", rr.get("mrr_before")),
                ("MRR after", rr.get("mrr_after")),
                ("Hit@1 before", rr.get("hit1_before")),
                ("Hit@1 after", rr.get("hit1_after")),
                ("improved", rr.get("improved_count")),
                ("unchanged", rr.get("unchanged_count")),
                ("degraded", rr.get("degraded_count")),
            ]
        )
    )

    lat = report.get("latency") or {}
    if lat:
        lines.append(_section("性能（Latency）"))
        lines.append(
            _kv(
                [
                    ("avg_recall_seconds", lat.get("avg_recall_seconds")),
                    ("avg_rerank_seconds", lat.get("avg_rerank_seconds")),
                    ("p50_recall_seconds", lat.get("p50_recall_seconds")),
                    ("p95_recall_seconds", lat.get("p95_recall_seconds")),
                    ("p50_rerank_seconds", lat.get("p50_rerank_seconds")),
                    ("p95_rerank_seconds", lat.get("p95_rerank_seconds")),
                    ("avg_candidates", lat.get("avg_candidates")),
                ]
            )
        )

    gate = report.get("gate") or {}
    overall = gate.get("overall") or {}
    lines.append(_section("Relevance Gate（threshold 见概览）"))
    lines.append(
        _kv(
            [
                ("Accuracy", overall.get("accuracy")),
                ("Precision", overall.get("precision")),
                ("Recall", overall.get("recall")),
                ("F1", overall.get("f1")),
                ("FPR", overall.get("fpr")),
                ("FNR", overall.get("fnr")),
            ]
        )
    )
    for cat in ("answerable", "irrelevant", "hard_negative"):
        info = (gate.get("by_category") or {}).get(cat)
        if info:
            lines.append(f"- **{cat}**: {json.dumps(info, ensure_ascii=False)}")

    sweep = report.get("threshold_sweep") or []
    if sweep:
        lines.append(_section("Threshold Sweep"))
        lines.append("| threshold | TP | FP | TN | FN | Precision | Recall | F1 | FPR | FNR |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in sweep:
            lines.append(
                f"| {row.get('threshold')} | {row.get('tp')} | {row.get('fp')} | "
                f"{row.get('tn')} | {row.get('fn')} | {row.get('precision')} | "
                f"{row.get('recall')} | {row.get('f1')} | {row.get('fpr')} | "
                f"{row.get('fnr')} |"
            )

    fp = report.get("false_positive_examples") or []
    if fp:
        lines.append(_section("False Positive 示例（unanswerable 但 gate 放行）"))
        for ex in fp:
            lines.append(
                f"- `{ex.get('question')}` → top1=`{ex.get('top1_chunk')}` "
                f"score={_fmt(ex.get('rerank_score'))}"
            )

    fn = report.get("false_negative_examples") or []
    if fn:
        lines.append(_section("False Negative 示例（answerable 但 gate 拒绝）"))
        for ex in fn:
            lines.append(
                f"- `{ex.get('question')}` → top1=`{ex.get('top1_chunk')}` "
                f"score={_fmt(ex.get('rerank_score'))}"
            )

    ac = report.get("answer_check") or {}
    if ac:
        lines.append(_section("Answer Check（轻量子串校验）"))
        lines.append(
            _kv(
                [
                    ("passed", ac.get("passed")),
                    ("failed", ac.get("failed")),
                    ("skipped", ac.get("skipped")),
                    ("pass_rate", ac.get("pass_rate")),
                ]
            )
        )

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
