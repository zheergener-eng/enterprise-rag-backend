"""Query Rewrite 服务。

根据最近的对话历史 + 当前用户问题，把依赖上下文的问题改写为一条
独立、完整、适合语义检索的 standalone query，供后续 Retrieval 使用。

本服务独立于 RAG / Session / API route，仅负责「问题改写」，不检索、不回答。

关键约定：
- 无历史 → 原样返回，不调用 LLM（新 session 自然退化为单轮检索）。
- 有历史 → 用独立的改写 prompt 调用 DeepSeek，仅做改写。
- 改写结果只用于 Retrieval；最终回答仍用用户原始 question（由上层 RAG 保证）。
- ``history`` 由调用方通过 ``get_recent_history`` 取最近 N 条（受
  ``max_history_messages`` 限制），不要把完整 Redis 历史直接传入。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.llm import get_llm_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewriteResult:
    """Query Rewrite 的返回结果。

    Attributes:
        query: 最终用于 Retrieval 的 standalone query 文本。
        used_llm: 是否调用了 LLM（无历史时为 False）。
        fallback: LLM 调用失败 / 返回空时是否回退到原始 question。
    """

    query: str
    used_llm: bool
    fallback: bool


# Query Rewrite 专用 prompt（与 RAG 回答 prompt 分离）
_QUERY_REWRITE_INSTRUCTION = (
    "你的任务是把【用户当前问题】改写为一条独立、完整、适合语义检索的查询文本"
    "（standalone query）。你可以参考【对话历史】来消除问题中的指代"
    "（如“它 / 这个 / 那”）和省略（如“那多久？”“怎么处理？”）。\n"
    "\n"
    "规则：\n"
    "1. 只做“改写”，不做“回答”，严禁输出答案、解释或过程说明。\n"
    "2. 保留用户原始意图，不改变问题含义，不做无意义的扩写。\n"
    "3. 若问题本身已经完整、明确、独立，尽量原样返回。\n"
    "4. 不得根据对话历史或你自己的知识补充任何新事实；历史中 assistant 的回答可能有误，"
    "只能作为理解用户当前指向对象的线索，不能当作事实来源。\n"
    "5. 只输出一条查询文本，不要输出 JSON、Markdown、引号或任何前缀说明。"
)

# 角色标签（历史消息 role → prompt 中的可读标签）
_ROLE_LABEL = {"user": "User", "assistant": "Assistant"}


def _format_history(history: list[dict]) -> str:
    """把历史消息格式化为多行文本（仅 role + content，不含 ts）。"""
    lines: list[str] = []
    for m in history:
        role = _ROLE_LABEL.get(m.get("role"), m.get("role"))
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def _build_prompt(history: list[dict], question: str) -> str:
    """构造改写 prompt：指令 + 对话历史 + 当前问题。"""
    return (
        f"{_QUERY_REWRITE_INSTRUCTION}\n"
        "\n"
        f"【对话历史】\n{_format_history(history)}\n"
        "\n"
        f"【用户当前问题】\n{question}"
    )


def _clean(raw: str) -> str:
    """对模型输出做最小清理。

    - strip 首尾空白；
    - 去除少量常见前缀（“查询：/ 改写：/ Query:”等）；
    - 若含多行，取首个非空行作为 query（模型偶尔附解释，query 通常在前）。

    不做复杂解析器；返回空串表示“无可用的 query”。
    """
    text = (raw or "").strip()
    if not text:
        return ""
    for prefix in ("查询：", "改写：", "改写结果：", "Query:", "Rewritten query:", "Rewritten:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def rewrite(history: list[dict], question: str) -> RewriteResult:
    """把当前问题改写为独立、完整、适合检索的 standalone query。

    Args:
        history: 最近对话历史（``get_recent_history`` 返回的 message 列表），
            仅用于消除指代 / 省略；不用于回答或补充事实。
        question: 用户当前问题（去除首尾空白后不得为空）。

    Returns:
        RewriteResult：
        - 无历史：query = 原问题，used_llm=False，不调用 LLM；
        - 有历史且 LLM 正常：query = 改写结果，used_llm=True，fallback=False；
        - 有历史但 LLM 失败 / 返回空：query = 原问题，used_llm=True，fallback=True。

    Raises:
        ValueError: question 为空或仅空白。
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty or blank")

    # 无历史：直接返回原问题，不调用 LLM
    if not history:
        return RewriteResult(query=question, used_llm=False, fallback=False)

    prompt = _build_prompt(history, question)
    try:
        raw = get_llm_client().generate(prompt)
    except Exception as exc:  # LLMError（调用失败 / 空内容）或其它异常
        logger.warning(
            "Query rewrite LLM call failed, fallback to original question: %s", exc
        )
        return RewriteResult(query=question, used_llm=True, fallback=True)

    cleaned = _clean(raw)
    if not cleaned:
        logger.warning(
            "Query rewrite returned empty after cleanup, fallback to original question"
        )
        return RewriteResult(query=question, used_llm=True, fallback=True)

    return RewriteResult(query=cleaned, used_llm=True, fallback=False)
