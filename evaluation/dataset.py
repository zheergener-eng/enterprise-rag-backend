"""Evaluation Dataset 加载、规范化与校验。

每条样本（JSON）结构：

    {
      "id": "q001",
      "question": "...",
      "category": "answerable | irrelevant | hard_negative",
      "answerable": true,
      "expected_answer": "14天" | null,
      "expected_document_id": "backup-retention" | null,
      "expected_chunk_id": "backup-retention:0" | null,
      "expected_chunk_ids": ["backup-retention:0"]  // 可选：多正确 chunk
    }

本模块只负责「读取 + 规范化 + 校验」，不触碰 Redis / 模型 / LLM。
数据集不写死在测试代码中，而是从 JSON 文件加载。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CATEGORIES = ("answerable", "irrelevant", "hard_negative")


@dataclass(frozen=True)
class EvalSample:
    """一条评测样本（规范化后）。

    Attributes:
        id: 唯一标识。
        question: 用户问题。
        category: answerable / irrelevant / hard_negative。
        answerable: 知识库是否存在可回答该问题的答案。
        expected_answer: 明确短事实（轻量 answer check 用，可为 None）。
        expected_document_id: 期望正确 chunk 所属 document_id（可为 None）。
        expected_chunk_ids: 期望正确 chunk 列表（多正确 chunk 时取全部）。
    """

    id: str
    question: str
    category: str
    answerable: bool
    expected_answer: str | None = None
    expected_document_id: str | None = None
    expected_chunk_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def expected_chunk_id(self) -> str | None:
        """单正确答案 chunk 的便捷访问（多 chunk 时返回第一个）。"""
        return self.expected_chunk_ids[0] if self.expected_chunk_ids else None


def _normalize(raw: dict) -> EvalSample:
    """把原始 JSON dict 规范化为 EvalSample（expected_chunk_id(s) 统一成 tuple）。"""
    chunk_ids = raw.get("expected_chunk_ids")
    if isinstance(chunk_ids, str):
        chunk_ids = [chunk_ids]
    if not chunk_ids:
        single = raw.get("expected_chunk_id")
        chunk_ids = [single] if single else []
    return EvalSample(
        id=(raw.get("id") or "").strip(),
        question=(raw.get("question") or "").strip(),
        category=(raw.get("category") or "").strip(),
        answerable=bool(raw.get("answerable", False)),
        expected_answer=raw.get("expected_answer"),
        expected_document_id=raw.get("expected_document_id"),
        expected_chunk_ids=tuple(chunk_ids),
    )


def validate_samples(samples: list[EvalSample]) -> None:
    """校验样本列表；非法时抛 ValueError（带样本 id 的清晰信息）。"""
    seen: set[str] = set()
    for s in samples:
        if not s.id:
            raise ValueError("sample id must be non-empty")
        if s.id in seen:
            raise ValueError(f"duplicate sample id: {s.id!r}")
        seen.add(s.id)
        if not s.question:
            raise ValueError(f"sample {s.id!r}: question must be non-empty")
        if s.category not in CATEGORIES:
            raise ValueError(
                f"sample {s.id!r}: invalid category {s.category!r}, "
                f"expected one of {CATEGORIES}"
            )
        # answerable 与 category 一致性
        if s.category == "answerable" and not s.answerable:
            raise ValueError(f"sample {s.id!r}: category 'answerable' requires answerable=true")
        if s.category != "answerable" and s.answerable:
            raise ValueError(
                f"sample {s.id!r}: category {s.category!r} requires answerable=false"
            )
        # answerable 必须有至少一个期望正确 chunk
        if s.answerable and not s.expected_chunk_ids:
            raise ValueError(
                f"sample {s.id!r}: answerable sample must have expected_chunk_id(s)"
            )


def load_dataset(path: str | Path) -> list[EvalSample]:
    """从 JSON 文件加载并校验数据集，返回规范化后的样本列表。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = [_normalize(item) for item in data]
    validate_samples(samples)
    return samples


def count_by_category(samples: list[EvalSample]) -> dict[str, int]:
    """按 category 统计样本数量。"""
    counts = {c: 0 for c in CATEGORIES}
    for s in samples:
        counts[s.category] += 1
    return counts
