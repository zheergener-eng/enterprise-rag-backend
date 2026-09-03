"""Chunking 切片模块单元测试。

覆盖：普通长文本、短文本、空文本、chunk overlap、超长单段文本，
以及 Markdown 标题上下文保留与非法参数校验。
"""
from __future__ import annotations

import pytest

from app.services.chunking import split_document


def test_normal_long_text():
    """普通长文本：应切成多个不超过 chunk_size 的片段，index 连续。"""
    text = ("第一段内容。\n" * 30) + "\n\n" + ("第二段内容。\n" * 30)
    chunk_size, overlap = 100, 20

    chunks = split_document(text, chunk_size, overlap)

    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert 0 < len(c.text) <= chunk_size


def test_short_text():
    """短文本：应作为单个片段返回。"""
    chunks = split_document("这是一段很短的文本。", chunk_size=100, overlap=20)

    assert len(chunks) == 1
    assert chunks[0].text == "这是一段很短的文本。"
    assert chunks[0].index == 0


def test_empty_text():
    """空文本 / 纯空白：应返回空列表。"""
    assert split_document("", chunk_size=100, overlap=20) == []
    assert split_document("   \n\t  ", chunk_size=100, overlap=20) == []


def test_chunk_overlap():
    """无句子边界的超长文本：硬切后相邻片段应有 overlap 衔接。"""
    text = "A" * 250  # 无任何标点/换行，强制走硬切
    chunk_size, overlap = 100, 20

    chunks = split_document(text, chunk_size, overlap)

    assert len(chunks) > 1
    for c in chunks:
        assert 0 < len(c.text) <= chunk_size
    # 相邻片段重叠部分应一致
    assert chunks[0].text[-overlap:] == chunks[1].text[:overlap]


def test_overlong_single_paragraph():
    """超长单段文本（含句子边界）：应被进一步切分为多块。"""
    text = "这是一段用于测试的超长文本。" * 30  # 约 420 字符
    chunk_size, overlap = 100, 20

    chunks = split_document(text, chunk_size, overlap)

    assert len(chunks) > 1
    for c in chunks:
        assert 0 < len(c.text) <= chunk_size
    assert all(c.text.strip() for c in chunks)


def test_markdown_headings():
    """Markdown 标题应作为上下文前缀保留在对应片段中。"""
    text = (
        "# 安装指南\n\n"
        "## 第一步\n\n"
        "请下载安装包。\n\n"
        "## 第二步\n\n"
        "运行安装程序。"
    )

    chunks = split_document(text, chunk_size=100, overlap=20)

    all_text = "\n".join(c.text for c in chunks)
    assert "安装指南" in all_text
    assert "第一步" in all_text
    assert "第二步" in all_text


def test_invalid_params():
    """非法参数应抛出 ValueError。"""
    with pytest.raises(ValueError):
        split_document("x", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        split_document("x", chunk_size=100, overlap=100)  # overlap >= chunk_size
    with pytest.raises(ValueError):
        split_document("x", chunk_size=100, overlap=-1)
