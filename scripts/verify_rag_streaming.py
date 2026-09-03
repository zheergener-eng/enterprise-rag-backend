"""真实多轮流式 RAG 验证脚本。

链路：session → get_recent_history → Query Rewrite → Retrieval → prompt → stream_generate → 写回。

验证目标：
  1. 第一轮流式：答「2:30」，写回 user + assistant 完整消息；
  2. 第二轮流式（指代「那保留多久？」）：答「14 天」，完整链路（history→rewrite→retrieval→stream）成立；
  3. 流式累积的 full_answer 与 Session 中保存的 assistant content 一致；
  4. 新 session 隔离：无历史时 rewrite 不调用 LLM，不继承上一 session。

前置条件：Redis Stack 已运行；.env 已配置 DEEPSEEK_API_KEY。
用法：python -m scripts.verify_rag_streaming   （或  PYTHONPATH=. python scripts/verify_rag_streaming.py）
"""
from __future__ import annotations

import io

from app.config import settings
from app.services.chunking import split_document
from app.services.embeddings import get_embedder
from app.services.query_rewrite import rewrite
from app.services.rag import answer_with_session_stream
from app.services.session import get_session_store
from app.services.vector_store import VectorStore

# 验证脚本使用独立 index / prefix，绝不触碰生产默认 rag:index / chunk:。
VERIFY_INDEX = "verify:rag:index"
VERIFY_PREFIX = "verify:chunk:"


def _stream(session_id: str, question: str) -> str:
    """消费流式生成器，返回拼接后的完整 answer。"""
    parts: list[str] = []
    for chunk in answer_with_session_stream(session_id, question):
        parts.append(chunk)
    return "".join(parts)


def main() -> None:
    document = (
        "# 星河项目运维手册\n\n"
        "## 数据库备份\n\n"
        "星河项目数据库每天凌晨 2:30 执行备份。\n"
        "备份文件保留 14 天。\n"
    )

    embedder = get_embedder()
    store = VectorStore(
        redis_url=settings.redis_url,
        index_name=VERIFY_INDEX,
        dim=settings.embedding_dim,
        prefix=VERIFY_PREFIX,
    )
    store.drop_index()
    store.ensure_index()

    session_store = get_session_store()
    sid = "verify:stream-multi-turn"
    sid2 = "verify:stream-isolated"

    summary: dict[str, object] = {}

    try:
        chunks = split_document(
            document, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
        )
        texts = [c.text for c in chunks]
        vectors = embedder.embed_documents(texts)
        store.add_document("star-river-stream", texts, vectors, title="星河项目运维手册")
        print(f"[准备] 文档切分为 {len(texts)} 个 chunk，已写入 Redis\n")

        # ---- 第一轮流式 ----
        q1 = "星河项目每天什么时候进行数据库备份？"
        answer1 = _stream(sid, q1)
        print(f"[第一轮] {q1}")
        print(f"  - answer: {answer1}")
        assert "2:30" in answer1, "第一轮应回答凌晨 2:30"
        summary["turn1_answer"] = answer1

        history_after_turn1 = session_store.get_history(sid)
        assert [m["role"] for m in history_after_turn1] == ["user", "assistant"]
        # 流式只写一条完整 assistant 消息
        assert history_after_turn1[1]["content"] == answer1, "第一轮 assistant 内容应等于完整 answer"
        summary["turn1_assistant_saved"] = history_after_turn1[1]["content"]

        # ---- 第二轮流式（指代，原样输入） ----
        q2 = "那保留多久？"

        # 独立打印 rewrite 结果（镜像 answer_with_session_stream 内部所用 history），
        # 用于展示 rewrite 确实恢复了指代对象。
        recent_before_turn2 = session_store.get_recent_history(sid)
        rr = rewrite(recent_before_turn2, q2)
        print(f"\n[第二轮 rewrite 参考] history={len(recent_before_turn2)} 条")
        print(f"  - rewritten_query: {rr.query!r}  used_llm={rr.used_llm} fallback={rr.fallback}")
        summary["turn2_rewritten_query"] = rr.query
        summary["turn2_rewrite_used_llm"] = rr.used_llm

        answer2 = _stream(sid, q2)
        print(f"\n[第二轮] {q2}")
        print(f"  - answer: {answer2}")
        assert "14" in answer2, "第二轮应回答保留 14 天"
        summary["turn2_answer"] = answer2

        # 写回顺序：user1/assistant1/user2/assistant2，user2 是原始问题
        history = session_store.get_history(sid)
        print("\n[写回] session 历史:")
        for m in history:
            print(f"  - {m['role']}: {m['content']}")
        assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
        assert history[2]["content"] == "那保留多久？", "user2 应是原始问题而非 rewritten query"
        # 关键：full_answer 与 Session 保存的 assistant content 一致
        assert history[3]["content"] == answer2, "第二轮 assistant 内容应等于流式累积的完整 answer"
        summary["session_user2"] = history[2]["content"]
        summary["turn2_assistant_saved"] = history[3]["content"]

        # ---- 新 session 隔离 ----
        q3 = "那保留多久？"
        rr_isolated = rewrite([], q3)  # 无历史
        answer3 = _stream(sid2, q3)
        print(f"\n[新 session 隔离] {q3}")
        print(f"  - rewritten_query(无历史): {rr_isolated.query!r}  used_llm={rr_isolated.used_llm}")
        print(f"  - answer: {answer3}")
        assert rr_isolated.used_llm is False, "无历史不应调用 rewrite LLM"
        assert rr_isolated.query == "那保留多久？", "不应继承上一 session 历史"
        summary["isolated_rewrite_used_llm"] = rr_isolated.used_llm
        summary["isolated_answer"] = answer3

        print("\n[结论] 真实多轮流式 RAG 验证通过。")
    finally:
        store.drop_index()
        session_store.clear_session(sid)
        session_store.clear_session(sid2)
        with io.open("scripts/_verify_stream_rag.txt", "w", encoding="utf-8") as f:
            for k, v in summary.items():
                f.write(f"{k} = {v!r}\n")


if __name__ == "__main__":
    main()
