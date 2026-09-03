"""Knowledge Base v2 Ingestion + Chunk Mapping + Evaluation v2。

使用生产级 Chunking / Embedding / Redis Retrieval / Rerank / Relevance Gate 链路，
将 evaluation/knowledge_base_v2 下的 6 份 Markdown 导入一个完全隔离的 Evaluation Redis Index，
然后运行 Evaluation Dataset v2。

关键约定（不可违反）：
- 独立 Redis index / key prefix：``eval:v2:rag:index`` / ``eval:v2:chunk:``，绝不触碰生产
  ``rag:index`` / ``chunk:`` 前缀；
- 复用 app 的 split_document / get_embedder / VectorStore / rerank / evaluate_relevance /
  rewrite / build_prompt 等真实链路，不为评测数据另写简化 chunker；
- ground-truth 由知识库事实 + 真实 chunk 内容确定，绝不根据模型检索结果反推；
- 只观察，不修改 production threshold / RAG / dataset_v2。

用法：
    python -m evaluation.evaluate_v2 --check-only   # 仅做 chunk manifest + ground-truth + 完整性检查
    python -m evaluation.evaluate_v2                # 完整 Evaluation v2

产出：
    evaluation/reports/v2_chunk_manifest.json
    evaluation/reports/v2_ground_truth_mapping.json
    evaluation/reports/evaluation_v2.json
    evaluation/reports/evaluation_v2.md
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from app.config import settings
from app.services.chunking import split_document
from app.services.embeddings import get_embedder
from app.services.llm import LLMError, get_llm_client
from app.services.query_rewrite import rewrite
from app.services.rag import build_context, build_prompt, build_prompt_with_history
from app.services.relevance import evaluate_relevance
from app.services.reranker import RetrievedChunk, rerank
from app.services.vector_store import VectorStore

from evaluation.metrics import (
    confusion_counts,
    mean_hit_at_k,
    mean_reciprocal_rank,
    mean_recall_at_k,
    percentile,
    summarize_confusion,
    threshold_sweep,
)
from evaluation.report import write_json

KB_DIR = Path("evaluation/knowledge_base_v2")
DATASET_PATH = Path("evaluation/dataset_v2.json")
REPORT_DIR = Path("evaluation/reports")

# 完全隔离的 v2 命名空间（与生产 rag:index 及 v1 eval:rag:index 都不同）
V2_INDEX_NAME = "eval:v2:rag:index"
V2_PREFIX = "eval:v2:chunk:"

CATEGORIES = (
    "direct_answerable",
    "paraphrase_answerable",
    "entity_disambiguation",
    "multi_turn",
    "hard_negative",
    "irrelevant",
)
ANSWERABLE_CATEGORIES = (
    "direct_answerable",
    "paraphrase_answerable",
    "entity_disambiguation",
    "multi_turn",
)
NEGATIVE_CATEGORIES = ("hard_negative", "irrelevant")


# --------------------------------------------------------------------------- #
# Dataset v2 加载
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class V2Sample:
    id: str
    category: str
    question: str
    answerable: bool
    expected_answer: str | None
    expected_document_ids: tuple[str, ...]
    expected_chunk_ids: tuple[str, ...]
    session_case_id: str | None
    turn: int | None
    history: tuple[dict, ...]
    expected_rewritten_query: str | None
    notes: str | None


def _as_tuple(value) -> tuple:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def load_v2_dataset(path: Path = DATASET_PATH) -> list[V2Sample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples: list[V2Sample] = []
    for raw in data:
        samples.append(
            V2Sample(
                id=(raw.get("id") or "").strip(),
                category=(raw.get("category") or "").strip(),
                question=(raw.get("question") or "").strip(),
                answerable=bool(raw.get("answerable", False)),
                expected_answer=raw.get("expected_answer"),
                expected_document_ids=_as_tuple(raw.get("expected_document_ids")),
                expected_chunk_ids=_as_tuple(raw.get("expected_chunk_ids")),
                session_case_id=raw.get("session_case_id"),
                turn=raw.get("turn"),
                history=tuple(raw.get("history") or []),
                expected_rewritten_query=raw.get("expected_rewritten_query"),
                notes=raw.get("notes"),
            )
        )
    return samples


def count_by_category(samples: list[V2Sample]) -> dict[str, int]:
    counts = {c: 0 for c in CATEGORIES}
    for s in samples:
        counts[s.category] = counts.get(s.category, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# 文本规范化与 ground-truth chunk 映射
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    """去掉所有空白，便于在 chunk 文本中做子串匹配。"""
    return "".join((text or "").split())


# 对 expected_answer 为「复合/多实体」或「反向查实体」的样本，显式给出定位 ground-truth
# chunk 的精确原子（避免用整句 expected_answer 匹配不到，或误匹配到错误章节）。
# 原子均为「去空白后」的子串；未列出的样本默认用 expected_answer 去空白作为单原子。
_ATOMS: dict[str, list[str]] = {
    # 反向查实体：题目给「保留 N 天」，答案给实体名，定位应指向「保留周期」章节
    "e001": ["30天"],
    "e002": ["14天"],
    "e003": ["7天"],
    "e004": ["最多3天"],
    "e012": ["按需"],
    # 复合事实：一个问题需多个 chunk 才能完整回答
    "e005": ["RDB快照文件", "14天"],
    "e008": ["A3机房", "B1机房"],
    "e009": ["90天", "30天"],
    "e010": ["不是同一个概念"],
    "e011": ["最多3天", "7天"],
    "m002_t2": ["RDB快照文件", "14天"],
}


def _atoms_for(sample: V2Sample) -> list[str]:
    if sample.id in _ATOMS:
        return [_norm(a) for a in _ATOMS[sample.id]]
    if sample.expected_answer:
        return [_norm(sample.expected_answer)]
    return []


def find_relevant_chunks(
    sample: V2Sample, doc_chunks: dict[str, list[tuple[str, str]]]
) -> list[str]:
    """根据 expected_document_ids + 真实 chunk 内容确定 relevant chunk ids。

    doc_chunks: document_id -> [(chunk_id, chunk_text)]（已含标题前缀的完整文本）。
    只在这些预期文档内搜索，命中任一原子的 chunk 视为 relevant。
    """
    atoms = _atoms_for(sample)
    if not atoms:
        return []
    relevant: list[str] = []
    for doc_id in sample.expected_document_ids:
        for chunk_id, text in doc_chunks.get(doc_id, []):
            nt = _norm(text)
            if any(a and a in nt for a in atoms):
                if chunk_id not in relevant:
                    relevant.append(chunk_id)
    return relevant


# --------------------------------------------------------------------------- #
# Knowledge Base v2 加载 + Chunk + Manifest
# --------------------------------------------------------------------------- #
def load_and_chunk_kb() -> tuple[list[dict], dict[str, list[tuple[str, str]]]]:
    """切分 6 份 KB 文档，返回 (manifest_documents, doc_chunks)。"""
    files = sorted(KB_DIR.glob("*.md"))
    manifest_docs: list[dict] = []
    doc_chunks: dict[str, list[tuple[str, str]]] = {}
    for f in files:
        content = f.read_text(encoding="utf-8")
        document_id = f.name  # 例如 01_xinghe_operations.md
        chunks = split_document(content, settings.chunk_size, settings.chunk_overlap)
        chunk_list: list[tuple[str, str]] = []
        chunk_entries: list[dict] = []
        for c in chunks:
            chunk_id = f"{document_id}:{c.index}"
            chunk_list.append((chunk_id, c.text))
            chunk_entries.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": c.index,
                    "text_preview": c.text[:80],
                }
            )
        doc_chunks[document_id] = chunk_list
        manifest_docs.append(
            {
                "document": document_id,
                "document_id": document_id,
                "chunk_count": len(chunks),
                "chunks": chunk_entries,
            }
        )
    return manifest_docs, doc_chunks


def _manifest_summary(manifest_docs: list[dict]) -> dict:
    all_lengths: list[int] = []
    per_doc: dict[str, int] = {}
    total_chunks = 0
    # 需要 chunk 文本长度，故这里从 doc_chunks 另算；简化：manifest 里不含长度，
    # 由调用方传入 doc_chunks 重算。此处只汇总结构计数，长度在 build_manifest 里补。
    return {}


def build_manifest(
    manifest_docs: list[dict], doc_chunks: dict[str, list[tuple[str, str]]]
) -> dict:
    lengths: list[int] = []
    per_doc: dict[str, int] = {}
    for doc_id, chunks in doc_chunks.items():
        per_doc[doc_id] = len(chunks)
        for _, text in chunks:
            lengths.append(len(text))
    total_chunks = len(lengths)
    return {
        "total_documents": len(manifest_docs),
        "total_chunks": total_chunks,
        "per_document_chunk_count": per_doc,
        "avg_chunk_length": round(mean(lengths), 2) if lengths else 0.0,
        "min_chunk_length": min(lengths) if lengths else 0,
        "max_chunk_length": max(lengths) if lengths else 0,
        "documents": manifest_docs,
    }


# --------------------------------------------------------------------------- #
# 数据完整性检查
# --------------------------------------------------------------------------- #
# 若知识库中出现这些「答案信号」，说明对应 hard_negative 被意外补齐。均为不含歧义的
# 具体缺失项（刻意避开「分钟/秒/负责人」等会误报的通用词）。
FORBIDDEN_ANSWER_SIGNALS = [
    "加密算法", "AES", "RSA", "DES", "SM4", "压缩算法",
    "分片算法", "跨可用区", "可用区",
    "序列号", "保修", "采购", "厂商", "品牌", "详细地址", "门牌",
    "审批人", "负责人姓名", "安全负责人", "人名",
    "重试", "锁定", "锁定阈值",
    "切换耗时", "恢复耗时", "恢复平均",
    "台数", "存储空间", "GB", "TB", "工作日",
]


def run_integrity_checks(
    samples: list[V2Sample],
    doc_chunks: dict[str, list[tuple[str, str]]],
    mapping: dict[str, list[str]],
) -> tuple[bool, list[str], dict]:
    """返回 (passed, problems, report)。任一 fatal problem 使 passed=False。"""
    problems: list[str] = []
    doc_ids = set(doc_chunks.keys())

    # 1/4. 每个 answerable 样本都能映射到 >=1 个 relevant chunk
    unmapped = [s.id for s in samples if s.answerable and not mapping.get(s.id)]
    if unmapped:
        problems.append(f"answerable 但无法映射 relevant chunk: {unmapped}")

    # 2. hard_negative 仍无答案（结构 + 关键词反查）
    for s in samples:
        if s.category == "hard_negative":
            if s.answerable:
                problems.append(f"{s.id}: hard_negative 却 answerable=true")
            if s.expected_document_ids or s.expected_chunk_ids:
                problems.append(f"{s.id}: hard_negative 却携带 expected ids")

    all_text = "\n".join(
        text for chunks in doc_chunks.values() for _, text in chunks
    )
    leaked = [term for term in FORBIDDEN_ANSWER_SIGNALS if term in all_text]
    if leaked:
        problems.append(f"知识库出现疑似答案信号（hard_negative 可能被补齐）: {leaked}")

    # 3. expected_document_ids 都存在
    for s in samples:
        for d in s.expected_document_ids:
            if d not in doc_ids:
                problems.append(f"{s.id}: 未知 expected_document_id {d!r}")

    # 5. category / answerable 一致性
    for s in samples:
        if s.category in ANSWERABLE_CATEGORIES and not s.answerable:
            problems.append(f"{s.id}: category={s.category} 却 answerable=false")
        if s.category in NEGATIVE_CATEGORIES and s.answerable:
            problems.append(f"{s.id}: category={s.category} 却 answerable=true")

    # 6. 重复 id
    seen: set[str] = set()
    for s in samples:
        if s.id in seen:
            problems.append(f"重复 id: {s.id}")
        seen.add(s.id)

    report = {
        "dataset_size": len(samples),
        "categories": count_by_category(samples),
        "answerable_unmapped": unmapped,
        "forbidden_signal_leaks": leaked,
        "hard_negative_count": sum(1 for s in samples if s.category == "hard_negative"),
        "irrelevant_count": sum(1 for s in samples if s.category == "irrelevant"),
    }
    return (len(problems) == 0), problems, report


# --------------------------------------------------------------------------- #
# 两阶段检索（含 embedding / recall / rerank 三段计时）
# --------------------------------------------------------------------------- #
def v2_retrieve(
    query: str, store: VectorStore, recall_top_n: int, rerank_top_k: int
) -> tuple[list[dict], list, float, float, float]:
    """对隔离 store 做真实两阶段检索，返回 (recalled, reranked, t_embed, t_recall, t_rerank)。"""
    embedder = get_embedder()
    t0 = time.perf_counter()
    qv = embedder.embed_query(query)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    recalled = store.search(qv, top_k=recall_top_n)
    t_recall = time.perf_counter() - t0

    candidates = [RetrievedChunk.from_dict(d) for d in recalled]
    t0 = time.perf_counter()
    reranked = rerank(query, candidates, top_k=rerank_top_k)
    t_rerank = time.perf_counter() - t0

    return recalled, reranked, t_embed, t_recall, t_rerank


# --------------------------------------------------------------------------- #
# 指标辅助
# --------------------------------------------------------------------------- #
def _first_rank(ids: list[str], expected: set[str]) -> int | None:
    for i, cid in enumerate(ids, start=1):
        if cid in expected:
            return i
    return None


def _dist(values: list[float]) -> dict:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None,
                "p25": None, "p75": None, "p90": None, "p95": None}
    return {
        "count": len(vals),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "mean": round(mean(vals), 4),
        "median": round(median(vals), 4),
        "p25": round(percentile(vals, 25), 4),
        "p75": round(percentile(vals, 75), 4),
        "p90": round(percentile(vals, 90), 4),
        "p95": round(percentile(vals, 95), 4),
    }


def _vector_metrics(ranks: list[int | None], recall_samples, top_n: int) -> dict:
    return {
        "hit_at_1": round(mean_hit_at_k(ranks, 1), 4),
        "hit_at_3": round(mean_hit_at_k(ranks, 3), 4),
        "hit_at_5": round(mean_hit_at_k(ranks, 5), 4),
        "mrr": round(mean_reciprocal_rank(ranks), 4),
        "recall_at_5": round(mean_recall_at_k(recall_samples, 5), 4),
    }


# --------------------------------------------------------------------------- #
# 主评测流程
# --------------------------------------------------------------------------- #
def run_eval(
    samples: list[V2Sample],
    store: VectorStore,
    doc_chunks: dict[str, list[tuple[str, str]]],
    mapping: dict[str, list[str]],
    recall_top_n: int,
    rerank_top_k: int,
    threshold: float,
) -> dict:
    answerable = [s for s in samples if s.answerable]
    recall_ranks: dict[str, int | None] = {}
    rerank_ranks: dict[str, int | None] = {}
    recall_at_k_samples: list[tuple[list[str], list[str]]] = []
    rerank_at_k_samples: list[tuple[list[str], list[str]]] = []

    # 逐样本详情（供 rerank improved/degraded 与 multi-turn 报告）
    detail: dict[str, dict] = {}

    recall_latencies: list[float] = []
    rerank_latencies: list[float] = []
    embed_latencies: list[float] = []
    candidate_counts: list[int] = []

    # gate rows：id -> top1_score, gate_decision, category, question
    gate_rows: list[dict] = []

    for s in samples:
        query = s.question
        rewrite_info = None
        if s.category == "multi_turn" and s.history:
            # 真实多轮链路：History -> Query Rewrite -> Retrieval
            rw = rewrite(list(s.history), s.question)
            query = rw.query
            rewrite_info = {
                "used_llm": rw.used_llm,
                "fallback": rw.fallback,
                "rewritten_query": rw.query,
                "expected_rewritten_query": s.expected_rewritten_query,
            }

        recalled, reranked, t_embed, t_recall, t_rerank = v2_retrieve(
            query, store, recall_top_n, rerank_top_k
        )
        embed_latencies.append(t_embed)
        recall_latencies.append(t_recall)
        rerank_latencies.append(t_rerank)
        candidate_counts.append(len(recalled))

        recall_ids = [d["chunk_id"] for d in recalled]
        rerank_ids = [r.chunk_id for r in reranked]

        correct = set(mapping.get(s.id, []))
        rr = _first_rank(recall_ids, correct)
        kr = _first_rank(rerank_ids, correct)
        recall_ranks[s.id] = rr
        rerank_ranks[s.id] = kr

        if s.answerable:
            recall_at_k_samples.append((recall_ids, list(correct)))
            rerank_at_k_samples.append((rerank_ids, list(correct)))

        decision = evaluate_relevance(reranked, threshold=threshold)
        top = reranked[0] if reranked else None
        gate_rows.append(
            {
                "id": s.id,
                "question": s.question,
                "category": s.category,
                "top1_chunk_id": top.chunk_id if top else None,
                "vector_similarity": round(top.vector_similarity, 4) if top else None,
                "rerank_score": round(top.rerank_score, 4) if top else None,
                "gate_decision": decision.is_relevant,
                "reason": decision.reason,
                "answerable": s.answerable,
            }
        )

        # 正确 chunk（召回位次最早者）的向量相似度与 rerank 分数
        correct_at_recall = recall_ids[rr - 1] if rr else None
        vec_sim = recalled[rr - 1]["similarity"] if rr else None
        rerank_score = None
        if kr:
            rerank_score = reranked[kr - 1].rerank_score
        elif correct_at_recall:
            for r in reranked:
                if r.chunk_id == correct_at_recall:
                    rerank_score = r.rerank_score
                    break

        # 检索详情（供 rerank 分析；含正确 chunk 的 recall/rerank 位次与分数）
        detail[s.id] = {
            "question": s.question,
            "category": s.category,
            "query": query,
            "correct_chunks": list(correct),
            "recall_rank": rr,
            "rerank_rank": kr,
            "recall_ids": recall_ids,
            "rerank_ids": rerank_ids,
            "vector_similarity_of_correct": round(vec_sim, 4) if vec_sim is not None else None,
            "rerank_score_of_correct": round(rerank_score, 4) if rerank_score is not None else None,
            "rewrite": rewrite_info,
        }

    # ---- 向量召回指标（overall + 4 类 answerable） ----
    vector_retrieval: dict = {"overall": None, "by_category": {}}
    vr_overall_ranks = [recall_ranks[s.id] for s in answerable]
    vr_overall_samples = recall_at_k_samples
    vector_retrieval["overall"] = _vector_metrics(vr_overall_ranks, vr_overall_samples, recall_top_n)
    for cat in ANSWERABLE_CATEGORIES:
        ids = [s.id for s in answerable if s.category == cat]
        ranks = [recall_ranks[i] for i in ids]
        cat_samples = [(detail[i]["recall_ids"], list(mapping[i])) for i in ids]
        vector_retrieval["by_category"][cat] = _vector_metrics(ranks, cat_samples, recall_top_n)

    # ---- Rerank 前后指标 ----
    r_before = [recall_ranks[s.id] for s in answerable]
    r_after = [rerank_ranks[s.id] for s in answerable]
    improved = unchanged = degraded = 0
    improved_examples: list[dict] = []
    degraded_examples: list[dict] = []
    for s in answerable:
        rr = recall_ranks[s.id]
        kr = rerank_ranks[s.id]
        d = detail[s.id]
        if rr is None:
            continue  # 召回未命中，不进 improved/degraded 对比
        if kr is not None and kr < rr:
            improved += 1
        elif kr is not None and kr == rr:
            unchanged += 1
        else:
            degraded += 1

        # 例子的「正确 chunk」取召回位次最早的那个 correct chunk
        correct_at_rr = d["recall_ids"][rr - 1] if rr - 1 < len(d["recall_ids"]) else None
        example = {
            "id": s.id,
            "question": s.question,
            "correct_chunk": correct_at_rr,
            "recall_rank": rr,
            "rerank_rank": kr,
            "vector_similarity": d["vector_similarity_of_correct"],
            "rerank_score": d["rerank_score_of_correct"],
        }
        if kr is not None and kr < rr:
            improved_examples.append(example)
        elif kr is None or (kr is not None and kr > rr):
            degraded_examples.append(example)

    improved_examples.sort(key=lambda e: (e["recall_rank"] or 0) - (e["rerank_rank"] or 0), reverse=True)
    degraded_examples.sort(key=lambda e: (e["rerank_rank"] or (rerank_top_k + 1)) - (e["recall_rank"] or 0), reverse=True)

    rerank_result = {
        "mrr_before": round(mean_reciprocal_rank(r_before), 4),
        "mrr_after": round(mean_reciprocal_rank(r_after), 4),
        "hit1_before": round(mean_hit_at_k(r_before, 1), 4),
        "hit1_after": round(mean_hit_at_k(r_after, 1), 4),
        "improved_count": improved,
        "unchanged_count": unchanged,
        "degraded_count": degraded,
        "recall_miss_count": sum(1 for r in r_before if r is None),
        "improved_examples_top10": improved_examples[:10],
        "degraded_examples_top10": degraded_examples[:10],
    }

    # ---- Gate ----
    pred = [r["gate_decision"] for r in gate_rows]
    exp = [r["answerable"] for r in gate_rows]
    gate_overall = summarize_confusion(confusion_counts(pred, exp))
    gate_by_category = {}
    for cat in ("answerable", "hard_negative", "irrelevant"):
        rows = [r for r in gate_rows if (r["category"] == cat) or (cat == "answerable" and r["answerable"])]
        if cat == "answerable":
            rows = [r for r in gate_rows if r["answerable"]]
        cp = [r["gate_decision"] for r in rows]
        ce = [r["answerable"] for r in rows]
        cc = confusion_counts(cp, ce)
        if cat == "answerable":
            gate_by_category[cat] = {
                "count": len(rows),
                "tp": cc["tp"],
                "fn": cc["fn"],
                "pass_rate": round(cc["tp"] / (cc["tp"] + cc["fn"]), 4) if (cc["tp"] + cc["fn"]) else 0.0,
            }
        else:
            gate_by_category[cat] = {
                "count": len(rows),
                "tn": cc["tn"],
                "fp": cc["fp"],
                "rejection_rate": round(cc["tn"] / (cc["tn"] + cc["fp"]), 4) if (cc["tn"] + cc["fp"]) else 0.0,
            }

    false_positive_examples = [
        {"id": r["id"], "question": r["question"], "category": r["category"],
         "top1_chunk_id": r["top1_chunk_id"], "rerank_score": r["rerank_score"]}
        for r in gate_rows if not r["answerable"] and r["gate_decision"]
    ]
    false_negative_examples = [
        {"id": r["id"], "question": r["question"], "category": r["category"],
         "top1_chunk_id": r["top1_chunk_id"], "rerank_score": r["rerank_score"]}
        for r in gate_rows if r["answerable"] and not r["gate_decision"]
    ]

    # ---- Score distribution ----
    score_by_class = {"answerable": [], "hard_negative": [], "irrelevant": []}
    for r in gate_rows:
        if r["answerable"]:
            score_by_class["answerable"].append(r["rerank_score"])
        elif r["category"] == "hard_negative":
            score_by_class["hard_negative"].append(r["rerank_score"])
        else:
            score_by_class["irrelevant"].append(r["rerank_score"])
    score_distribution = {k: _dist(v) for k, v in score_by_class.items()}
    score_distribution["overlap_analysis"] = _overlap_analysis(score_distribution)

    # ---- Threshold sweep ----
    sweep_samples = [(r["rerank_score"], r["answerable"]) for r in gate_rows]
    sweep_thresholds = [round(i / 10, 1) for i in range(10)] + [0.95, 0.98]
    sweep = threshold_sweep(sweep_samples, sweep_thresholds)

    # ---- Latency ----
    latency = {
        "avg_embed_seconds": round(mean(embed_latencies), 4) if embed_latencies else 0.0,
        "avg_recall_seconds": round(mean(recall_latencies), 4) if recall_latencies else 0.0,
        "avg_rerank_seconds": round(mean(rerank_latencies), 4) if rerank_latencies else 0.0,
        "p95_embed_seconds": round(percentile(embed_latencies, 95), 4),
        "p95_recall_seconds": round(percentile(recall_latencies, 95), 4),
        "p95_rerank_seconds": round(percentile(rerank_latencies, 95), 4),
        "avg_total_retrieval_seconds": round(
            mean(e + rc + rk for e, rc, rk in zip(embed_latencies, recall_latencies, rerank_latencies)), 4
        ) if embed_latencies else 0.0,
        "avg_candidates": round(mean(candidate_counts), 2) if candidate_counts else 0.0,
        "recall_top_n": recall_top_n,
        "rerank_top_k": rerank_top_k,
    }

    return {
        "vector_retrieval": vector_retrieval,
        "rerank": rerank_result,
        "gate": {
            "threshold": threshold,
            "overall": gate_overall,
            "by_category": gate_by_category,
            "false_positive_examples": false_positive_examples,
            "false_negative_examples": false_negative_examples,
        },
        "score_distribution": score_distribution,
        "threshold_sweep": sweep,
        "latency": latency,
        "detail": detail,
    }


def _overlap_analysis(score_distribution: dict) -> dict:
    a = score_distribution.get("answerable", {})
    h = score_distribution.get("hard_negative", {})
    i = score_distribution.get("irrelevant", {})
    a_min, a_max = a.get("min"), a.get("max")
    h_min, h_max = h.get("min"), h.get("max")
    i_min, i_max = i.get("min"), i.get("max")

    def overlap(x0, x1, y0, y1):
        if None in (x0, x1, y0, y1):
            return None
        lo, hi = max(x0, y0), min(x1, y1)
        return max(0.0, hi - lo)

    a_hn = overlap(a_min, a_max, h_min, h_max)
    a_ir = overlap(a_min, a_max, i_min, i_max)
    a_range = (a_max - a_min) if (a_min is not None and a_max is not None) else None

    def verdict(ov, rng):
        if ov is None or rng in (None, 0):
            return "无法判定"
        ratio = ov / rng
        if ratio > 0.5:
            return "明显"
        if ratio > 0.2:
            return "中等"
        return "很小"

    return {
        "answerable_range": [a_min, a_max],
        "hard_negative_range": [h_min, h_max],
        "irrelevant_range": [i_min, i_max],
        "answerable_hard_negative_overlap": round(a_hn, 4) if a_hn is not None else None,
        "answerable_irrelevant_overlap": round(a_ir, 4) if a_ir is not None else None,
        "answerable_hard_negative_verdict": verdict(a_hn, a_range),
        "answerable_irrelevant_verdict": verdict(a_ir, a_range),
    }


# --------------------------------------------------------------------------- #
# Answer Check（轻量 deterministic substring check）
# --------------------------------------------------------------------------- #
def _is_atomic_answer(text: str | None) -> bool:
    if not text:
        return False
    return not any(ch in text for ch in ("，", "、", "（", "）", "与", "比", "长于"))


def _answer_contains(expected: str, answer: str) -> bool:
    return bool(expected) and _norm(expected) in _norm(answer)


def run_answer_check(samples, store, mapping, detail, recall_top_n, rerank_top_k, threshold) -> dict:
    """仅对 atomic expected_answer 的 answerable 样本做 LLM 生成 + 子串检查。"""
    targets = [
        s for s in samples
        if s.answerable and _is_atomic_answer(s.expected_answer)
    ]
    passed = failed = skipped = 0
    compound_skipped = sum(
        1 for s in samples if s.answerable and not _is_atomic_answer(s.expected_answer)
    )
    per_sample: list[dict] = []
    llm = get_llm_client()
    for s in targets:
        # 使用评测过程中已算好的检索结果（避免重复检索）
        query = detail[s.id]["query"]
        recalled, reranked, _, _, _ = v2_retrieve(query, store, recall_top_n, rerank_top_k)
        decision = evaluate_relevance(reranked, threshold=threshold)
        if not decision.is_relevant:
            per_sample.append({"id": s.id, "question": s.question, "status": "gate_rejected"})
            failed += 1
            continue
        context = build_context(reranked)
        if s.category == "multi_turn" and s.history:
            prompt = build_prompt_with_history(list(s.history), context, s.question)
        else:
            prompt = build_prompt(s.question, context)
        try:
            answer = llm.generate(prompt)
        except LLMError:
            per_sample.append({"id": s.id, "question": s.question, "status": "llm_error"})
            skipped += 1
            continue
        ok = _answer_contains(s.expected_answer or "", answer)
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
        "compound_skipped": compound_skipped,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "note": "Answer Check 为轻量 deterministic substring check，不能等同于完整语义正确性评估。",
        "per_sample": per_sample,
    }


# --------------------------------------------------------------------------- #
# Multi-turn / Query Rewrite 专项
# --------------------------------------------------------------------------- #
def run_multi_turn_report(samples, detail, mapping) -> dict:
    mt = [s for s in samples if s.category == "multi_turn"]
    per_turn: list[dict] = []
    for s in sorted(mt, key=lambda x: (x.session_case_id or "", x.turn or 0)):
        d = detail[s.id]
        rw = d.get("rewrite") or {}
        correct = set(mapping.get(s.id, []))
        recalled_hit = d["recall_rank"] is not None
        reranked_hit = d["rerank_rank"] is not None
        per_turn.append(
            {
                "id": s.id,
                "session_case_id": s.session_case_id,
                "turn": s.turn,
                "question": s.question,
                "rewrite_used_llm": rw.get("used_llm"),
                "rewrite_fallback": rw.get("fallback"),
                "rewritten_query": rw.get("rewritten_query"),
                "expected_rewritten_query": rw.get("expected_rewritten_query"),
                "recalled_hit": recalled_hit,
                "reranked_hit": reranked_hit,
                "relevant_chunks": list(correct),
            }
        )
    n = len(per_turn)
    summary = {
        "turn_count": n,
        "rewrite_used_llm": sum(1 for t in per_turn if t["rewrite_used_llm"]),
        "rewrite_fallback": sum(1 for t in per_turn if t["rewrite_fallback"]),
        "recalled_hit": sum(1 for t in per_turn if t["recalled_hit"]),
        "reranked_hit": sum(1 for t in per_turn if t["reranked_hit"]),
        "recalled_hit_rate": round(sum(1 for t in per_turn if t["recalled_hit"]) / n, 4) if n else 0.0,
        "reranked_hit_rate": round(sum(1 for t in per_turn if t["reranked_hit"]) / n, 4) if n else 0.0,
    }
    return {"summary": summary, "per_turn": per_turn}


# --------------------------------------------------------------------------- #
# 报告输出
# --------------------------------------------------------------------------- #
def build_report(
    samples, manifest, mapping_docs, integrity, eval_result, answer_check, multi_turn,
    recall_top_n, rerank_top_k, threshold,
) -> dict:
    return {
        "dataset_size": len(samples),
        "categories": count_by_category(samples),
        "recall_top_n": recall_top_n,
        "rerank_top_k": rerank_top_k,
        "threshold": threshold,
        "chunk_manifest": manifest,
        "ground_truth_mapping": mapping_docs,
        "integrity_checks": integrity,
        "vector_retrieval": eval_result["vector_retrieval"],
        "rerank": eval_result["rerank"],
        "gate": eval_result["gate"],
        "score_distribution": eval_result["score_distribution"],
        "threshold_sweep": eval_result["threshold_sweep"],
        "multi_turn": multi_turn,
        "answer_check": answer_check,
        "latency": eval_result["latency"],
    }


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_v2_markdown(path: Path, report: dict) -> None:
    lines: list[str] = ["# Evaluation v2 Report（Knowledge Base v2）"]
    lines.append("")
    lines.append("## 概览")
    lines += [
        f"- dataset_size: {report['dataset_size']}",
        f"- categories: {json.dumps(report['categories'], ensure_ascii=False)}",
        f"- recall_top_n: {report['recall_top_n']}",
        f"- rerank_top_k: {report['rerank_top_k']}",
        f"- gate_threshold: {report['threshold']}",
    ]

    cm = report["chunk_manifest"]
    lines += ["", "## Chunk Manifest"]
    lines += [
        f"- total_documents: {cm['total_documents']}",
        f"- total_chunks: {cm['total_chunks']}",
        f"- per_document_chunk_count: {json.dumps(cm['per_document_chunk_count'], ensure_ascii=False)}",
        f"- avg_chunk_length: {cm['avg_chunk_length']}",
        f"- min_chunk_length: {cm['min_chunk_length']}",
        f"- max_chunk_length: {cm['max_chunk_length']}",
    ]

    ic = report["integrity_checks"]
    lines += ["", "## 数据完整性检查"]
    lines += [f"- passed: {ic.get('passed', ic.get('integrity_passed'))}"]
    lines += [f"- answerable_unmapped: {json.dumps(ic.get('answerable_unmapped'), ensure_ascii=False)}"]
    lines += [f"- forbidden_signal_leaks: {json.dumps(ic.get('forbidden_signal_leaks'), ensure_ascii=False)}"]

    vr = report["vector_retrieval"]
    lines += ["", "## Vector Retrieval（Recall 阶段）"]
    lines += ["| scope | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR |"]
    lines += ["|---|---|---|---|---|---|"]
    for scope in ["overall"] + list(ANSWERABLE_CATEGORIES):
        m = vr["by_category"].get(scope) if scope != "overall" else vr["overall"]
        if m:
            lines.append(f"| {scope} | {m['hit_at_1']} | {m['hit_at_3']} | {m['hit_at_5']} | {m['recall_at_5']} | {m['mrr']} |")

    rr = report["rerank"]
    lines += ["", "## Rerank（Before → After）"]
    lines += [
        f"- MRR before={rr['mrr_before']}  after={rr['mrr_after']}",
        f"- Hit@1 before={rr['hit1_before']}  after={rr['hit1_after']}",
        f"- improved={rr['improved_count']}  unchanged={rr['unchanged_count']}  degraded={rr['degraded_count']}  recall_miss={rr['recall_miss_count']}",
    ]
    if rr["improved_examples_top10"]:
        lines += ["", "### Top improved examples", "| question | correct_chunk | recall_rank | rerank_rank | vec_sim | rerank_score |", "|---|---|---|---|---|---|"]
        for e in rr["improved_examples_top10"]:
            lines.append(f"| {e['question']} | {e['correct_chunk']} | {e['recall_rank']} | {e['rerank_rank']} | {e['vector_similarity']} | {e['rerank_score']} |")
    if rr["degraded_examples_top10"]:
        lines += ["", "### Top degraded examples", "| question | correct_chunk | recall_rank | rerank_rank | vec_sim | rerank_score |", "|---|---|---|---|---|---|"]
        for e in rr["degraded_examples_top10"]:
            lines.append(f"| {e['question']} | {e['correct_chunk']} | {e['recall_rank']} | {e['rerank_rank']} | {e['vector_similarity']} | {e['rerank_score']} |")

    gate = report["gate"]
    ov = gate["overall"]
    lines += ["", "## Relevance Gate"]
    lines += [
        f"- Accuracy={ov['accuracy']}  Precision={ov['precision']}  Recall={ov['recall']}  F1={ov['f1']}",
        f"- FPR={ov['fpr']}  FNR={ov['fnr']}  (TP={ov['tp']} FP={ov['fp']} TN={ov['tn']} FN={ov['fn']})",
    ]
    for cat in ("answerable", "hard_negative", "irrelevant"):
        info = gate["by_category"].get(cat)
        if info:
            lines.append(f"- {cat}: {json.dumps(info, ensure_ascii=False)}")

    sd = report["score_distribution"]
    lines += ["", "## Score Distribution（top-1 rerank score）"]
    lines += ["| class | count | min | max | mean | median | p25 | p75 | p90 | p95 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for cls in ("answerable", "hard_negative", "irrelevant"):
        d = sd[cls]
        lines.append(f"| {cls} | {d['count']} | {d['min']} | {d['max']} | {d['mean']} | {d['median']} | {d['p25']} | {d['p75']} | {d['p90']} | {d['p95']} |")
    ovl = sd["overlap_analysis"]
    lines += [
        f"- answerable_range: {ovl['answerable_range']}",
        f"- hard_negative_range: {ovl['hard_negative_range']}",
        f"- irrelevant_range: {ovl['irrelevant_range']}",
        f"- answerable↔hard_negative overlap: {ovl['answerable_hard_negative_overlap']}（{ovl['answerable_hard_negative_verdict']}）",
        f"- answerable↔irrelevant overlap: {ovl['answerable_irrelevant_overlap']}（{ovl['answerable_irrelevant_verdict']}）",
    ]

    sweep = report["threshold_sweep"]
    lines += ["", "## Threshold Sweep", "| threshold | TP | FP | TN | FN | Precision | Recall | F1 | FPR | FNR |", "|---|---|---|---|---|---|---|---|---|---|"]
    for row in sweep:
        lines.append(f"| {row['threshold']} | {row['tp']} | {row['fp']} | {row['tn']} | {row['fn']} | {row['precision']} | {row['recall']} | {row['f1']} | {row['fpr']} | {row['fnr']} |")

    mt = report["multi_turn"]["summary"]
    lines += ["", "## Multi-turn / Query Rewrite"]
    lines += [
        f"- turn_count={mt['turn_count']}  rewrite_used_llm={mt['rewrite_used_llm']}  rewrite_fallback={mt['rewrite_fallback']}",
        f"- recalled_hit_rate={mt['recalled_hit_rate']}  reranked_hit_rate={mt['reranked_hit_rate']}",
    ]
    lines += ["| id | turn | rewritten_query | expected_rewritten_query | recalled_hit | reranked_hit |", "|---|---|---|---|---|---|"]
    for t in report["multi_turn"]["per_turn"]:
        lines.append(f"| {t['id']} | {t['turn']} | {t['rewritten_query']} | {t['expected_rewritten_query']} | {t['recalled_hit']} | {t['reranked_hit']} |")

    ac = report["answer_check"]
    lines += ["", "## Answer Check（轻量 deterministic substring check）"]
    lines += [
        f"- passed={ac['passed']}  failed={ac['failed']}  skipped={ac['skipped']}  compound_skipped={ac['compound_skipped']}  pass_rate={ac['pass_rate']}",
        f"- note: {ac['note']}",
    ]

    lat = report["latency"]
    lines += ["", "## Latency"]
    lines += [
        f"- avg_embed={lat['avg_embed_seconds']}s  p95_embed={lat['p95_embed_seconds']}s",
        f"- avg_recall={lat['avg_recall_seconds']}s  p95_recall={lat['p95_recall_seconds']}s",
        f"- avg_rerank={lat['avg_rerank_seconds']}s  p95_rerank={lat['p95_rerank_seconds']}s",
        f"- avg_total_retrieval={lat['avg_total_retrieval_seconds']}s  avg_candidates={lat['avg_candidates']}",
    ]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> None:
    check_only = "--check-only" in sys.argv
    recall_top_n = settings.recall_top_n
    rerank_top_k = settings.rerank_top_k
    threshold = settings.rerank_relevance_threshold

    # 1. 切分 KB v2 + manifest
    manifest_docs, doc_chunks = load_and_chunk_kb()
    manifest = build_manifest(manifest_docs, doc_chunks)
    write_json(REPORT_DIR / "v2_chunk_manifest.json", manifest)

    # 2. 加载 dataset v2 + ground-truth 映射
    samples = load_v2_dataset(DATASET_PATH)
    mapping: dict[str, list[str]] = {}
    mapping_docs: list[dict] = []
    for s in samples:
        rel = find_relevant_chunks(s, doc_chunks) if s.answerable else []
        mapping[s.id] = rel
        mapping_docs.append(
            {
                "case_id": s.id,
                "expected_document_ids": list(s.expected_document_ids),
                "relevant_chunk_ids": rel,
                "relevant_chunk_text_preview": [
                    _preview_for_chunk(cid, doc_chunks) for cid in rel
                ],
            }
        )
    write_json(REPORT_DIR / "v2_ground_truth_mapping.json", mapping_docs)

    # 3. 数据完整性检查
    passed, problems, integrity_report = run_integrity_checks(samples, doc_chunks, mapping)
    integrity_report["passed"] = passed
    integrity_report["problems"] = problems
    write_json(REPORT_DIR / "v2_integrity_check.json", integrity_report)

    print("=" * 70)
    print("[Chunk Manifest]")
    print(f"  total_documents={manifest['total_documents']}  total_chunks={manifest['total_chunks']}")
    print(f"  per_document_chunk_count={manifest['per_document_chunk_count']}")
    print(f"  avg/min/max chunk_length={manifest['avg_chunk_length']}/{manifest['min_chunk_length']}/{manifest['max_chunk_length']}")
    print("[数据完整性检查]")
    print(f"  passed={passed}")
    for p in problems:
        print(f"  - {p}")
    print(f"  类别分布={integrity_report['categories']}")
    print(f"  answerable_unmapped={integrity_report['answerable_unmapped']}")
    print(f"  forbidden_signal_leaks={integrity_report['forbidden_signal_leaks']}")

    if check_only:
        print("[check-only] 已生成 manifest + ground-truth + integrity，未运行 Evaluation。")
        return

    if not passed:
        print("[停止] 数据完整性检查未通过，未运行 Evaluation。请先人工检查问题。")
        return

    # 4. 构建隔离 store + 导入 KB v2
    store = VectorStore(
        redis_url=settings.redis_url,
        index_name=V2_INDEX_NAME,
        dim=settings.embedding_dim,
        prefix=V2_PREFIX,
    )
    store.drop_index()
    store.ensure_index()
    embedder = get_embedder()
    try:
        for doc_id, chunks in doc_chunks.items():
            texts = [t for _, t in chunks]
            vectors = embedder.embed_documents(texts)
            store.add_document(doc_id, texts, vectors, title="")
        # 预热 reranker（embedder 已加载），避免首次懒加载污染计时
        v2_retrieve("预热", store, recall_top_n=3, rerank_top_k=3)

        # 5. 完整 Evaluation
        eval_result = run_eval(
            samples, store, doc_chunks, mapping, recall_top_n, rerank_top_k, threshold
        )
        answer_check = run_answer_check(
            samples, store, mapping, eval_result["detail"], recall_top_n, rerank_top_k, threshold
        )
        multi_turn = run_multi_turn_report(samples, eval_result["detail"], mapping)
        report = build_report(
            samples, manifest, mapping_docs, integrity_report, eval_result,
            answer_check, multi_turn, recall_top_n, rerank_top_k, threshold,
        )
        write_json(REPORT_DIR / "evaluation_v2.json", report)
        write_v2_markdown(REPORT_DIR / "evaluation_v2.md", report)

        print("=" * 70)
        print("[Vector Retrieval] overall:", eval_result["vector_retrieval"]["overall"])
        print("[Rerank]", eval_result["rerank"])
        print("[Gate] overall:", eval_result["gate"]["overall"])
        print("[Gate] by_category:", eval_result["gate"]["by_category"])
        print("[Score Distribution]", eval_result["score_distribution"]["overlap_analysis"])
        print("[Multi-turn summary]", multi_turn["summary"])
        print("[Answer Check]", {k: answer_check[k] for k in ("total", "passed", "failed", "skipped", "pass_rate")})
        print("[Latency]", eval_result["latency"])
        print(f"报告已写入 evaluation/reports/evaluation_v2.json / evaluation_v2.md")
    finally:
        store.drop_index()  # 只清理 v2 自己的 index，不动生产 rag:index


def _preview_for_chunk(chunk_id: str, doc_chunks: dict[str, list[tuple[str, str]]]) -> str:
    doc_id = chunk_id.rsplit(":", 1)[0]
    for cid, text in doc_chunks.get(doc_id, []):
        if cid == chunk_id:
            return text[:80]
    return ""


if __name__ == "__main__":
    main()
