"""RAG 服务（单轮 + 多轮，非流式 + 流式）。

组合「两阶段检索（向量召回 + Rerank）+ Relevance Gate」与 DeepSeek LLM Client，
实现问答：

- 单轮：`answer_question` —— question → Recall Top-N → Rerank Top-K → Relevance Gate
  → relevant 时 context → prompt → generate；insufficient 时返回统一 no-answer。
- 多轮非流式：`answer_with_session` —— session → get_recent_history → query_rewrite
  → Recall Top-N → Rerank Top-K → Relevance Gate → build_prompt_with_history
  → generate → 写回 session（insufficient 时写回 no-answer）。
- 多轮流式：`answer_with_session_stream` —— 与上述相同的前置（history / rewrite /
  两阶段检索 / Relevance Gate / prompt 构造，经 `_prepare_turn` 复用），唯一差异是用
  `stream_generate` 流式产出最终回答，并延迟到流正常结束后写回 session。

多轮中 rewritten query 仅用于 Retrieval（向量召回与 Rerank 均用它），
最终 Prompt 仍用用户原始 question。单轮 / 多轮 / 流式统一复用
`retrieve_with_rerank` + `evaluate_relevance`，不出现非流式与流式检索逻辑漂移。
Relevance Gate 判定 insufficient 是正常业务结果（非 error），不调用 DeepSeek，
返回统一「知识库无依据」响应，避免把低相关 chunks 当作依据或让模型凭常识补答。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.llm import LLMError, get_llm_client
from app.services.query_rewrite import RewriteResult, rewrite
from app.services.relevance import RelevanceResult, evaluate_relevance
from app.services.reranker import RerankedChunk
from app.services.retrieval import retrieve_with_rerank
from app.services.session import get_session_store


# 统一「知识库无依据」响应：检索为空或 Relevance Gate 判定 insufficient 时返回，
# 不调用 DeepSeek，不让模型用自身常识补充答案。
_NO_ANSWER = "根据当前知识库无法确定。"


@dataclass(frozen=True)
class RAGResult:
    """RAG 的返回结果（单轮 / 多轮共用）。

    Attributes:
        answer: 最终回答文本。
        sources: 检索到的 chunk 元数据（document_id / chunk_id / index / text / title），
            即 retrieved chunks。
        retrieval_count: 检索到的 chunk 数量。
        answered: 是否基于知识库作答（检索为空时为 False）。
        original_question: 用户原始问题（多轮时最终 Prompt 用它，而非改写 query）。
        rewritten_query: Query Rewrite 产生的检索用 query；单轮为 None。
        rewrite_used_llm: Query Rewrite 是否调用了 LLM。
        rewrite_fallback: Query Rewrite 是否因 LLM 失败回退到原始问题。
        relevance: Relevance Gate 决策结果（is_relevant / top_score / threshold / reason），
            供内部对象 / Evaluation 访问；不暴露给普通 API 客户端。
    """

    answer: str
    sources: list[dict]
    retrieval_count: int
    answered: bool
    original_question: str = ""
    rewritten_query: str | None = None
    rewrite_used_llm: bool = False
    rewrite_fallback: bool = False
    relevance: RelevanceResult | None = None


# 系统角色与回答规则（RAG 核心约束）
_SYSTEM_INSTRUCTION = (
    "你是一个严谨的企业知识库问答助手。请严格依据下方【知识库内容】回答【用户问题】。\n"
    "\n"
    "回答规则：\n"
    "1. 优先并严格依据知识库内容回答，不得脱离知识库。\n"
    "2. 不得把模型自身知识伪装成知识库内容。\n"
    "3. 如果知识库内容不足以回答问题，必须明确说明“根据当前知识库无法确定”。\n"
    "4. 不得编造知识库中不存在的信息。"
)

# 多轮 RAG 系统指令：在单轮规则基础上增加「对话历史」约束
_SYSTEM_INSTRUCTION_MULTI = (
    "你是一个严谨的企业知识库问答助手。请严格依据下方【知识库内容】回答【当前问题】。\n"
    "\n"
    "回答规则：\n"
    "1. 事实性内容必须依据【知识库内容】，不得脱离知识库。\n"
    "2. 【对话历史】只用于理解指代、上下文和连续对话，"
    "不得把历史中的 assistant 回答直接当作可靠知识来源。\n"
    "3. 不得把模型自身知识伪装成知识库内容。\n"
    "4. 如果知识库内容不足以回答当前问题，必须明确说明“根据当前知识库无法确定”。\n"
    "5. 不得编造知识库中不存在的信息。"
)


def build_context(chunks: list[RerankedChunk]) -> str:
    """将检索结果（两阶段后的 RerankedChunk）组织为 context 文本。

    每项格式：`[Document N] 标题 / 内容`；不含 embedding / 距离 / 分数等向量细节。
    """
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.title or "（无标题）"
        blocks.append(f"[Document {i}]\n标题：{title}\n内容：{chunk.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    """构造 RAG prompt：系统规则 + 知识库 context + 用户问题。"""
    return (
        f"{_SYSTEM_INSTRUCTION}\n"
        "\n"
        f"【知识库内容】\n{context}\n"
        "\n"
        f"【用户问题】\n{question}"
    )


# 历史消息 role → prompt 中的可读标签
_HISTORY_ROLE_LABEL = {"user": "User", "assistant": "Assistant"}


def _format_history(history: list[dict]) -> str:
    """把对话历史格式化为多行文本（User / Assistant 逐行）。"""
    lines: list[str] = []
    for m in history:
        role = _HISTORY_ROLE_LABEL.get(m.get("role"), m.get("role"))
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def build_prompt_with_history(
    history: list[dict], context: str, question: str
) -> str:
    """构造多轮 RAG prompt：对话历史 + 知识库 context + 当前问题（三部分清晰区分）。"""
    return (
        f"{_SYSTEM_INSTRUCTION_MULTI}\n"
        "\n"
        f"【对话历史】\n{_format_history(history)}\n"
        "\n"
        f"【知识库内容】\n{context}\n"
        "\n"
        f"【当前问题】\n{question}"
    )


def answer_question(
    question: str,
    recall_top_n: int | None = None,
    rerank_top_k: int | None = None,
) -> RAGResult:
    """单轮 RAG：两阶段检索（Recall → Rerank）→ Relevance Gate → 构造 prompt → 生成答案。

    Args:
        question: 用户问题（去除首尾空白后不得为空）。
        recall_top_n: 向量召回数量；缺省使用 settings.recall_top_n。
        rerank_top_k: Rerank 后保留数量；缺省使用 settings.rerank_top_k。

    Returns:
        RAGResult。当 Relevance Gate 判定 insufficient（含检索为空）时，
        不调用 LLM，直接返回统一 no-answer。

    Raises:
        ValueError: question 为空或仅空白。
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty or blank")

    chunks = retrieve_with_rerank(
        question, recall_top_n=recall_top_n, rerank_top_k=rerank_top_k
    )

    # Relevance Gate：检索为空或 top-1 分数低于阈值 → 不调用 DeepSeek，
    # 返回统一 no-answer，不把低相关 chunks 当作知识库依据。
    relevance = evaluate_relevance(chunks)
    if not relevance.is_relevant:
        return RAGResult(
            answer=_NO_ANSWER,
            sources=[],
            retrieval_count=len(chunks),
            answered=False,
            original_question=question,
            relevance=relevance,
        )

    context = build_context(chunks)
    prompt = build_prompt(question, context)
    answer = get_llm_client().generate(prompt)

    sources = _build_sources(chunks)

    return RAGResult(
        answer=answer,
        sources=sources,
        retrieval_count=len(chunks),
        answered=True,
        original_question=question,
        relevance=relevance,
    )


def _build_sources(chunks: list[RerankedChunk]) -> list[dict]:
    """把检索到的 chunk 转成 sources 元数据列表（不含向量 / 分数细节）。"""
    return [
        {
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "index": chunk.index,
            "text": chunk.text,
            "title": chunk.title,
        }
        for chunk in chunks
    ]


@dataclass(frozen=True)
class _PreparedTurn:
    """answer_with_session 与 answer_with_session_stream 共用的前置准备结果。

    Attributes:
        question: 去除首尾空白后的用户原始问题。
        history: 最近历史消息（受 max_history_messages 限制）。
        rewrite_result: Query Rewrite 结果（query 用于 Retrieval）。
        chunks: 两阶段检索后的 chunk 列表（RerankedChunk，可能为空）。
        sources: 检索结果的元数据（可能为空）。
        prompt: 最终送入 LLM 的 prompt；Relevance Gate 判定 insufficient 或
            chunks 为空时为 ""。
        relevance: Relevance Gate 决策结果（决定走 RAG 还是 no-answer）。
    """

    question: str
    history: list[dict]
    rewrite_result: RewriteResult
    chunks: list[RerankedChunk]
    sources: list[dict]
    prompt: str
    relevance: RelevanceResult


def _prepare_turn(
    session_id: str,
    question: str,
    recall_top_n: int | None,
    rerank_top_k: int | None,
) -> _PreparedTurn:
    """多轮 RAG 的共享前置：校验问题 → 读历史 → Query Rewrite → 两阶段检索 → 构造 prompt。

    供非流式 `answer_with_session` 与流式 `answer_with_session_stream` 复用，
    避免维护两套容易漂移的业务逻辑。

    Raises:
        ValueError: question 为空或仅空白。
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty or blank")

    store = get_session_store()
    history = store.get_recent_history(session_id)  # 受 max_history_messages 限制

    # 1. Query Rewrite（无历史时内部短路，直接返回原问题，不调用 LLM）
    rewrite_result = rewrite(history, question)

    # 2. 两阶段检索：向量召回 + Rerank（始终使用 rewritten query，而非原始 question）
    chunks = retrieve_with_rerank(
        rewrite_result.query, recall_top_n=recall_top_n, rerank_top_k=rerank_top_k
    )

    # 3. Relevance Gate：检索为空或 top-1 分数低于阈值 → insufficient，不构造 prompt
    relevance = evaluate_relevance(chunks)

    sources = _build_sources(chunks)

    # 4. 构造 prompt：仅当 relevant 时构造；历史非空含【对话历史】段，否则退化为单轮
    if relevance.is_relevant:
        context = build_context(chunks)
        if history:
            prompt = build_prompt_with_history(history, context, question)
        else:
            prompt = build_prompt(question, context)
    else:
        prompt = ""

    return _PreparedTurn(
        question=question,
        history=history,
        rewrite_result=rewrite_result,
        chunks=chunks,
        sources=sources,
        prompt=prompt,
        relevance=relevance,
    )


def answer_with_session(
    session_id: str,
    question: str,
    recall_top_n: int | None = None,
    rerank_top_k: int | None = None,
) -> RAGResult:
    """多轮 RAG：读历史 → Query Rewrite → 两阶段检索 → Relevance Gate → 多轮 prompt → 生成 → 写回。

    执行流程：
        session_id → get_recent_history → rewrite(history, question)
        → Recall Top-N → Rerank Top-K → Relevance Gate
        → relevant：build_context / build_prompt_with_history → generate
        → insufficient：统一 no-answer
        → 写回 user(原始问题) + assistant(答案 / no-answer)

    关键约定：
    - rewritten query 只用于 Retrieval（向量召回与 Rerank 均用它），最终 Prompt 仍用原始 question；
    - 无历史时 rewrite 短路返回原问题，多轮自动退化为单轮；
    - Relevance Gate insufficient（含检索为空）：不调用 LLM，返回统一 no-answer，并写回该轮；
    - LLM 调用失败：异常向上传播，不写回任何消息（不产生虚假 answer）。

    Args:
        session_id: 会话标识。
        question: 用户原始问题（去除首尾空白后不得为空）。
        recall_top_n: 向量召回数量；缺省 settings.recall_top_n。
        rerank_top_k: Rerank 后保留数量；缺省 settings.rerank_top_k。

    Returns:
        RAGResult（含 rewrite 状态与原始问题，供调试 / Evaluation）。

    Raises:
        ValueError: question 为空或仅空白。
        LLMError: LLM 调用失败（由 generate 抛出，向上传播）。
    """
    prepared = _prepare_turn(session_id, question, recall_top_n, rerank_top_k)
    store = get_session_store()

    # Relevance Gate insufficient（含检索为空）：不调用 LLM，返回统一 no-answer，
    # 并写回本轮 user + assistant(no-answer)。
    if not prepared.relevance.is_relevant:
        store.add_message(session_id, "user", prepared.question)
        store.add_message(session_id, "assistant", _NO_ANSWER)
        return RAGResult(
            answer=_NO_ANSWER,
            sources=[],
            retrieval_count=len(prepared.chunks),
            answered=False,
            original_question=prepared.question,
            rewritten_query=prepared.rewrite_result.query,
            rewrite_used_llm=prepared.rewrite_result.used_llm,
            rewrite_fallback=prepared.rewrite_result.fallback,
            relevance=prepared.relevance,
        )

    # 调用 LLM（失败抛异常向上传播；不写回任何消息）
    answer = get_llm_client().generate(prepared.prompt)

    # 写回 session：user 用原始问题，assistant 用最终答案
    store.add_message(session_id, "user", prepared.question)
    store.add_message(session_id, "assistant", answer)

    return RAGResult(
        answer=answer,
        sources=prepared.sources,
        retrieval_count=len(prepared.chunks),
        answered=True,
        original_question=prepared.question,
        rewritten_query=prepared.rewrite_result.query,
        rewrite_used_llm=prepared.rewrite_result.used_llm,
        rewrite_fallback=prepared.rewrite_result.fallback,
        relevance=prepared.relevance,
    )


def answer_with_session_stream(
    session_id: str,
    question: str,
    recall_top_n: int | None = None,
    rerank_top_k: int | None = None,
):
    """多轮 RAG 流式：复用前置准备，逐步 yield answer 文本增量，正常结束后写回 session。

    与 `answer_with_session` 共用 `_prepare_turn`（历史 / Query Rewrite / 两阶段检索 /
    Relevance Gate / prompt 构造完全一致），唯一差异是最终回答改用 `stream_generate`
    流式产出，并把 Session 写回延迟到流正常结束后。因此非流式与流式使用完全相同的检索顺序。

    执行流程：
        session_id → get_recent_history → rewrite → Recall Top-N → Rerank Top-K
        → Relevance Gate → relevant：build prompt → stream_generate() → yield chunk
          → 累积 full_answer → 写回；insufficient：单次 yield 统一 no-answer。

    关键约定：
    - Relevance Gate insufficient（含检索为空）：单次 yield 统一 no-answer，
      写回 user + assistant，不调用 LLM；
    - 流式生成：逐步 yield 非空 chunk，同时在内存累积 full_answer；
    - 写回时机：仅当流正常结束（迭代耗尽）后，写回 user(原始问题) + assistant(full_answer)；
      中途异常会向上传播，跳过写回（不产生 partial assistant answer）；
    - assistant 只写一条完整消息，不逐 chunk 写入 Redis。

    Args:
        session_id: 会话标识。
        question: 用户原始问题（去除首尾空白后不得为空）。
        recall_top_n: 向量召回数量；缺省 settings.recall_top_n。
        rerank_top_k: Rerank 后保留数量；缺省 settings.rerank_top_k。

    Yields:
        回答文本增量（str）。检索为空时 yield 单条确定性“无法回答”。

    Raises:
        ValueError: question 为空或仅空白。
        LLMError: 流式生成失败（由 stream_generate 抛出，向上传播），或流无内容。
    """
    prepared = _prepare_turn(session_id, question, recall_top_n, rerank_top_k)
    store = get_session_store()

    # Relevance Gate insufficient（含检索为空）：不调用 LLM 流式生成，
    # 单次 yield 统一 no-answer，并写回本轮 user + assistant(no-answer)。
    # 这是正常业务结果，不是系统 error。
    if not prepared.relevance.is_relevant:
        store.add_message(session_id, "user", prepared.question)
        store.add_message(session_id, "assistant", _NO_ANSWER)
        yield _NO_ANSWER
        return

    # 流式生成 + 累积（中途异常向上传播，跳过写回）
    full_answer = ""
    for chunk in get_llm_client().stream_generate(prepared.prompt):
        full_answer += chunk
        yield chunk

    if not full_answer.strip():
        raise LLMError("DeepSeek returned empty stream (no content chunks)")

    # 只有流正常结束后才写回：user 用原始问题，assistant 用完整答案
    store.add_message(session_id, "user", prepared.question)
    store.add_message(session_id, "assistant", full_answer)
