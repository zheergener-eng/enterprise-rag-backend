"""Evaluation Framework 单元测试（纯函数 / 数据集 / 报告，不触碰 Redis、模型或 LLM）。

覆盖 spec 第 14 节的 10 项测试要求：
1. 数据集 schema（加载 / 规范化 / 校验）
2. 类别定义与统计
3. Hit@K
4. MRR
5. Precision / Recall / F1（含 Accuracy / FPR / FNR）
6. Threshold Sweep
7. 多 expected_chunk_ids 支持
8. 空数据集指标（返回 0.0 而非报错）
9. Eval 使用隔离 index，绝不动默认 rag:index
10. 报告 JSON 可写、可读、保中文
"""
from __future__ import annotations

import json

import pytest

from evaluation.dataset import (
    CATEGORIES,
    EvalSample,
    count_by_category,
    load_dataset,
    validate_samples,
)
from evaluation.knowledge import EVAL_INDEX_NAME, EVAL_PREFIX
from evaluation.metrics import (
    accuracy,
    confusion_counts,
    default_thresholds,
    false_negative_rate,
    false_positive_rate,
    f1_score,
    hit_at_k,
    mean_hit_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    percentile,
    precision,
    recall,
    recall_at_k,
    reciprocal_rank,
    summarize_confusion,
    threshold_sweep,
)
from evaluation.report import write_json, write_markdown


# ---------------------------------------------------------------------------
# 1. 数据集 schema
# ---------------------------------------------------------------------------

def _write_dataset(tmp_path, data: list[dict]) -> str:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_load_dataset_parses_schema(tmp_path):
    path = _write_dataset(
        tmp_path,
        [
            {
                "id": "q001",
                "question": "数据库每天什么时候备份？",
                "category": "answerable",
                "answerable": True,
                "expected_answer": "2:30",
                "expected_document_id": "backup-time",
                "expected_chunk_id": "backup-time:0",
            },
            {
                "id": "q002",
                "question": "明天天气怎么样？",
                "category": "irrelevant",
                "answerable": False,
                "expected_answer": None,
                "expected_document_id": None,
                "expected_chunk_id": None,
            },
        ],
    )
    samples = load_dataset(path)
    assert len(samples) == 2

    s = samples[0]
    assert isinstance(s, EvalSample)
    assert s.id == "q001"
    assert s.question == "数据库每天什么时候备份？"
    assert s.category == "answerable"
    assert s.answerable is True
    assert s.expected_answer == "2:30"
    assert s.expected_document_id == "backup-time"
    assert s.expected_chunk_ids == ("backup-time:0",)
    assert s.expected_chunk_id == "backup-time:0"

    assert samples[1].expected_chunk_ids == ()
    assert samples[1].expected_chunk_id is None


@pytest.mark.parametrize(
    "mutator,msg_part",
    [
        (lambda s: s.update(id=""), "id"),
        (lambda s: s.update(question=""), "question"),
        (lambda s: s.update(category="bogus"), "category"),
        (lambda s: s.update(category="answerable", answerable=False), "answerable"),
        (lambda s: s.update(answerable=True, expected_chunk_id=None, expected_chunk_ids=None), "expected_chunk_id"),
    ],
)
def test_validate_samples_rejects_invalid(mutator, msg_part):
    base = {
        "id": "q001",
        "question": "问题",
        "category": "answerable",
        "answerable": True,
        "expected_chunk_id": "doc:0",
    }
    mutator(base)
    sample = EvalSample(
        id=base["id"],
        question=base["question"],
        category=base["category"],
        answerable=base["answerable"],
        expected_answer=None,
        expected_document_id=None,
        expected_chunk_ids=tuple(base["expected_chunk_ids"] or [])
        if base.get("expected_chunk_ids") is not None
        else ((base["expected_chunk_id"],) if base.get("expected_chunk_id") else ()),
    )
    with pytest.raises(ValueError, match=msg_part):
        validate_samples([sample])


def test_validate_samples_rejects_duplicate_id():
    samples = [
        EvalSample(id="q001", question="a", category="answerable", answerable=True,
                   expected_chunk_ids=("doc:0",)),
        EvalSample(id="q001", question="b", category="answerable", answerable=True,
                   expected_chunk_ids=("doc:0",)),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        validate_samples(samples)


# ---------------------------------------------------------------------------
# 2. 类别定义与统计
# ---------------------------------------------------------------------------

def test_categories_definition():
    assert CATEGORIES == ("answerable", "irrelevant", "hard_negative")


def test_count_by_category():
    samples = [
        EvalSample(id="a", question="x", category="answerable", answerable=True,
                   expected_chunk_ids=("c:0",)),
        EvalSample(id="b", question="y", category="irrelevant", answerable=False),
        EvalSample(id="c", question="z", category="irrelevant", answerable=False),
        EvalSample(id="d", question="w", category="hard_negative", answerable=False),
    ]
    assert count_by_category(samples) == {
        "answerable": 1,
        "irrelevant": 2,
        "hard_negative": 1,
    }


# ---------------------------------------------------------------------------
# 3. Hit@K
# ---------------------------------------------------------------------------

def test_hit_at_k():
    assert hit_at_k(1, 1) is True
    assert hit_at_k(3, 1) is False
    assert hit_at_k(5, 5) is True
    assert hit_at_k(None, 3) is False


def test_mean_hit_at_k():
    assert mean_hit_at_k([1, None, 5], 3) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# 4. MRR
# ---------------------------------------------------------------------------

def test_reciprocal_rank():
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(2) == 0.5
    assert reciprocal_rank(3) == pytest.approx(1 / 3)
    assert reciprocal_rank(None) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([1, 2, None]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5. Precision / Recall / F1（含 Accuracy / FPR / FNR）
# ---------------------------------------------------------------------------

def test_confusion_summary_metrics():
    counts = confusion_counts(
        predicted_relevant=[True, True, False, False],
        expected_answerable=[True, False, True, False],
    )
    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}

    assert precision(1, 1) == 0.5
    assert recall(1, 1) == 0.5
    assert f1_score(1, 1, 1) == 0.5
    assert accuracy(1, 1, 1, 1) == 0.5
    assert false_positive_rate(1, 1) == 0.5
    assert false_negative_rate(1, 1) == 0.5


def test_metrics_zero_denominator():
    # 无正例 / 无负例时不除零
    assert precision(0, 0) == 0.0
    assert recall(0, 0) == 0.0
    assert f1_score(0, 0, 0) == 0.0
    assert false_positive_rate(0, 0) == 0.0
    assert false_negative_rate(0, 0) == 0.0


def test_summarize_confusion_rounds():
    summary = summarize_confusion({"tp": 1, "fp": 1, "tn": 1, "fn": 1})
    assert summary["accuracy"] == 0.5
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["f1"] == 0.5
    assert summary["fpr"] == 0.5
    assert summary["fnr"] == 0.5


# ---------------------------------------------------------------------------
# 6. Threshold Sweep
# ---------------------------------------------------------------------------

def test_threshold_sweep():
    samples = [(0.9, True), (0.5, False), (None, False)]
    rows = threshold_sweep(samples, [0.6])
    assert len(rows) == 1
    row = rows[0]
    assert row["threshold"] == 0.6
    assert row["tp"] == 1  # 0.9 >= 0.6 且 answerable
    assert row["tn"] == 2  # 0.5 < 0.6 且 None → insufficient
    assert row["fp"] == 0
    assert row["fn"] == 0
    assert row["accuracy"] == 1.0


def test_default_thresholds_covers_high_region():
    ts = default_thresholds()
    assert ts == sorted(ts)
    assert 0.0 in ts and 0.5 in ts and 0.9 in ts
    assert 0.95 in ts and 0.98 in ts and 0.995 in ts


# ---------------------------------------------------------------------------
# 7. 多 expected_chunk_ids 支持
# ---------------------------------------------------------------------------

def test_load_dataset_multi_expected_chunk_ids(tmp_path):
    path = _write_dataset(
        tmp_path,
        [
            {
                "id": "q001",
                "question": "问题",
                "category": "answerable",
                "answerable": True,
                "expected_chunk_ids": ["doc:0", "doc:1"],
            }
        ],
    )
    s = load_dataset(path)[0]
    assert s.expected_chunk_ids == ("doc:0", "doc:1")
    assert s.expected_chunk_id == "doc:0"


def test_recall_at_k_multi_chunk():
    retrieved = ["doc:0", "doc:2", "doc:1"]
    expected = ["doc:0", "doc:1"]
    assert recall_at_k(retrieved, expected, 3) == 1.0  # 两个都召回
    assert recall_at_k(retrieved, expected, 2) == 0.5  # 只召回 doc:0


# ---------------------------------------------------------------------------
# 8. 空数据集指标
# ---------------------------------------------------------------------------

def test_metrics_empty_input_return_zero():
    assert mean_hit_at_k([], 3) == 0.0
    assert mean_reciprocal_rank([]) == 0.0
    assert mean_recall_at_k([], 3) == 0.0
    assert percentile([], 95) == 0.0


# ---------------------------------------------------------------------------
# 9. Eval 隔离 index，绝不动默认 rag:index
# ---------------------------------------------------------------------------

def test_eval_uses_isolated_index():
    assert EVAL_INDEX_NAME != "rag:index"
    assert EVAL_INDEX_NAME == "eval:rag:index"
    assert EVAL_PREFIX == "eval:chunk:"
    # 前缀不能与默认生产前缀相同
    assert EVAL_PREFIX != "chunk:"


# ---------------------------------------------------------------------------
# 10. 报告 JSON 可写 / 可读 / 保中文
# ---------------------------------------------------------------------------

def test_write_json_roundtrip_preserves_chinese(tmp_path):
    data = {"dataset_size": 36, "gate": {"threshold": 0.5, "note": "中文注释"}}
    path = tmp_path / "latest.json"
    write_json(path, data)

    raw = path.read_text(encoding="utf-8")
    assert "中文注释" in raw  # ensure_ascii=False 直接写入中文
    loaded = json.loads(raw)
    assert loaded == data


def test_write_markdown_writes_file(tmp_path):
    path = tmp_path / "latest.md"
    write_markdown(path, {"dataset_size": 36, "gate": {"overall": {"accuracy": 0.9}}})
    assert path.exists()
    assert "RAG Evaluation Report" in path.read_text(encoding="utf-8")
