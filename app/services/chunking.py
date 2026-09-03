"""切片策略：将长文档拆分为适合检索的片段（Chunks）。

切分算法（自顶向下，边界优先，尽量不切断语义）：
  1. 解析 Markdown 标题与段落边界：按空行分段，识别 `#{1,6}` 标题并维护标题栈，
     每个自然段携带其所属的标题路径（用 " > " 连接）作为上下文前缀。
  2. 对每个「标题前缀 + 段落正文」，若超过 chunk_size，按句子边界贪心累积；
     单个句子仍超长时，按字符硬切并保留 overlap。
  3. 合并相邻且同标题的短片段，减少碎片（合并后不超过 chunk_size）。

后续衔接设计（本模块暂不实现，仅保证接口可衔接）：
  - 本模块只做纯文本切分，不感知 document_id。
  - 返回的 Chunk 含 text 与文档内序号 index；上层写入向量库时按
    `chunk_id = f"{document_id}:{index}"` 组合出全局唯一 chunk_id，
    并携带 document_id / title 等元数据（由 VectorStore 层负责）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """单个文本片段。

    Attributes:
        text: 片段正文（已含标题上下文前缀）。
        index: 文档内片段序号，用于后续拼出全局唯一 chunk_id。
    """

    text: str
    index: int


@dataclass
class _Section:
    """解析出的一个自然段及其所属标题路径。"""

    heading: str  # 标题路径前缀（可空，形如 "安装指南 > 第一步"）
    body: str     # 段落正文


@dataclass
class _Unit:
    """待合并的片段单元（text 已含标题前缀，heading 用于主题一致性判断）。"""

    text: str
    heading: str


# Markdown ATX 标题：`#` 至 `######` 后跟空格与标题文本
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# 句子结束边界：中英文句末标点 + 换行（不含英文句点，避免误切小数/缩写）
_SENTENCE_END_RE = re.compile(r"([。！？；!?;\n]+)")


def split_document(content: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """将文档切分为片段列表。

    Args:
        content: 文档正文（Markdown 或纯文本）。
        chunk_size: 目标片长（字符数），必须 >= 1。
        overlap: 硬切时相邻片重叠字符数，必须满足 0 <= overlap < chunk_size。

    Returns:
        片段列表（输入为空文本时返回空列表）。
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    if not content or not content.strip():
        return []

    sections = _extract_sections(content)

    # 每个自然段展开为「标题前缀 + 正文」，超长则进一步切分
    units: list[_Unit] = []
    for sec in sections:
        full = f"{sec.heading}\n{sec.body}" if sec.heading else sec.body
        for piece in _split_long_text(full, chunk_size, overlap):
            units.append(_Unit(text=piece, heading=sec.heading))

    merged = _merge_units(units, chunk_size)

    return [Chunk(text=t, index=i) for i, t in enumerate(merged)]


def _extract_sections(content: str) -> list[_Section]:
    """按 Markdown 标题与空行段落边界解析文档。

    空行分隔自然段；标题行更新标题栈（丢弃级别 >= 当前级别的旧标题），
    标题行本身不进入正文（信息已通过 heading 前缀保留）。
    """
    sections: list[_Section] = []
    heading_stack: list[str] = []
    current: list[str] = []

    def flush() -> None:
        body = "\n".join(current).strip()
        if body:
            sections.append(
                _Section(
                    heading=" > ".join(heading_stack) if heading_stack else "",
                    body=body,
                )
            )
        current.clear()

    for raw in content.split("\n"):
        line = raw.strip()
        if line == "":
            # 空行视为段落分隔
            flush()
            continue

        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_stack = heading_stack[: level - 1] + [title]
        else:
            current.append(raw)

    flush()
    return sections


def _split_sentences(text: str) -> list[str]:
    """按句子结束标点与换行切分文本，标点归属句尾。"""
    parts = _SENTENCE_END_RE.split(text)
    sentences: list[str] = []
    for i in range(0, len(parts) - 1, 2):
        sentences.append(parts[i] + parts[i + 1])
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1])
    return [s for s in sentences if s.strip()]


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """将超长文本按句子边界贪心累积，超长句子则硬切并保留 overlap。"""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    buf = ""
    for sentence in _split_sentences(text):
        if not sentence:
            continue

        if not buf:
            # 首个句子：若自身超长则硬切，否则作为累积起点
            if len(sentence) > chunk_size:
                chunks.extend(_hard_split(sentence, chunk_size, overlap))
            else:
                buf = sentence
        elif len(buf) + len(sentence) <= chunk_size:
            buf += sentence
        else:
            chunks.append(buf)
            if len(sentence) > chunk_size:
                chunks.extend(_hard_split(sentence, chunk_size, overlap))
                buf = ""
            else:
                buf = sentence

    if buf:
        chunks.append(buf)
    return chunks


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """对单个超长句子按字符硬切，相邻片重叠 overlap 字符。"""
    chunks: list[str] = []
    step = chunk_size - overlap  # > 0，已由入参校验保证
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks


def _merge_units(units: list[_Unit], chunk_size: int) -> list[str]:
    """合并相邻且同标题的短片段，减少碎片（合并后不超过 chunk_size）。"""
    merged: list[str] = []
    buf_text = ""
    buf_heading = ""
    for u in units:
        if not buf_text:
            buf_text, buf_heading = u.text, u.heading
        elif u.heading == buf_heading and len(buf_text) + 1 + len(u.text) <= chunk_size:
            buf_text += "\n" + u.text
        else:
            merged.append(buf_text)
            buf_text, buf_heading = u.text, u.heading

    if buf_text:
        merged.append(buf_text)
    return merged
