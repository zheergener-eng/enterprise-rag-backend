"""真实多轮 RAG 验证脚本。

链路：session → get_recent_history → Query Rewrite → Retrieval → 多轮 prompt → DeepSeek → 写回。

验证目标：
  1. 同一 session 两轮：第一轮答「2:30」，第二轮指代问题「那保留多久？」答「14 天」；
  2. 第二轮 rewritten query 恢复指代，但写入 Session 的是原始问题；
  3. 新 session 直接问「那保留多久？」不继承上一 session 历史。

前置条件：Redis Stack 已运行；.env 已配置 DEEPSEEK_API_KEY。
用法：python -m scripts.verify_rag_session   （或  PYTHONPATH=. python scripts/verify_rag_session.py）
"""
from __future__ import annotations

import io

from app.config import settings
from app.services.chunking import split_document
from app.services.embeddings import get_embedder
from app.services.rag import answer_with_session
from app.services.session import get_session_store
from app.services.vector_store import VectorStore

# 验证脚本使用独立 index / prefix，绝不触碰生产默认 rag:index / chunk:。
VERIFY_INDEX = "verify:rag:index"
VERIFY_PREFIX = "verify:chunk:"


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
    sid = "verify:multi-turn"
    sid2 = "verify:multi-turn-isolated"

    summary: dict[str, object] = {}

    try:
        chunks = split_document(
            document, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
        )
        texts = [c.text for c in chunks]
        vectors = embedder.embed_documents(texts)
        store.add_document("star-river-multi", texts, vectors, title="星河项目运维手册")
        print(f"[准备] 文档切分为 {len(texts)} 个 chunk，已写入 Redis\n")

        # ---- 第一轮 ----
        q1 = "星河项目每天什么时候进行数据库备份？"
        r1 = answer_with_session(sid, q1)
        print(f"[第一轮] {q1}")
        print(f"  - answer: {r1.answer}")
        assert "2:30" in r1.answer, "第一轮应回答凌晨 2:30"
        summary["turn1_answer"] = r1.answer

        history_after_turn1 = session_store.get_history(sid)
        assert len(history_after_turn1) == 2, "第一轮后应写回 user + assistant 两条"
        assert history_after_turn1[0]["role"] == "user"
        assert history_after_turn1[1]["role"] == "assistant"

        # ---- 第二轮（指代，原样输入，不人为补全） ----
        q2 = "那保留多久？"
        r2 = answer_with_session(sid, q2)
        print(f"\n[第二轮] {q2}")
        print(f"  - original_question: {r2.original_question!r}")
        print(f"  - rewritten_query: {r2.rewritten_query!r}")
        print(f"  - rewrite_used_llm={r2.rewrite_used_llm} fallback={r2.rewrite_fallback}")
        print(f"  - retrieval_count={r2.retrieval_count}")
        print(f"  - answer: {r2.answer}")

        assert r2.original_question == "那保留多久？", "最终问题应是原始问题"
        assert r2.rewrite_used_llm is True, "第二轮有历史应调用 rewrite LLM"
        assert "星河项目" in r2.rewritten_query, "应恢复指代对象：星河项目"
        assert any(k in r2.rewritten_query for k in ("保留", "多久", "多少")), "应保留意图"
        assert "14" in r2.answer, "第二轮应回答保留 14 天"

        summary.update(
            turn2_original_question=r2.original_question,
            turn2_rewritten_query=r2.rewritten_query,
            turn2_rewrite_used_llm=r2.rewrite_used_llm,
            turn2_answer=r2.answer,
            turn2_retrieval_count=r2.retrieval_count,
            turn2_top_chunk=(r2.sources[0]["text"] if r2.sources else ""),
        )

        # 写回顺序：user1/assistant1/user2/assistant2，且 user2 是原始问题
        history = session_store.get_history(sid)
        print("\n[写回] session 历史:")
        for m in history:
            print(f"  - {m['role']}: {m['content']}")
        assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
        assert history[2]["content"] == "那保留多久？", "写入的是原始问题而非 rewritten query"
        assert "星河项目" not in history[2]["content"]
        summary["session_user2"] = history[2]["content"]

        # ---- 新 session 隔离 ----
        q3 = "那保留多久？"
        r3 = answer_with_session(sid2, q3)
        print(f"\n[新 session 隔离] {q3}")
        print(f"  - original_question: {r3.original_question!r}")
        print(f"  - rewritten_query: {r3.rewritten_query!r}")
        print(f"  - rewrite_used_llm={r3.rewrite_used_llm} fallback={r3.rewrite_fallback}")
        print(f"  - answer: {r3.answer}")
        assert r3.rewrite_used_llm is False, "新 session 无历史，不应调用 rewrite LLM"
        assert r3.rewritten_query == "那保留多久？", "新 session 不应继承上一 session 历史"
        assert "星河项目" not in r3.rewritten_query
        summary.update(
            isolated_rewritten_query=r3.rewritten_query,
            isolated_rewrite_used_llm=r3.rewrite_used_llm,
            isolated_answer=r3.answer,
        )

        print("\n[结论] 真实多轮 RAG 验证通过。")
    finally:
        store.drop_index()
        session_store.clear_session(sid)
        session_store.clear_session(sid2)
        # 输出 UTF-8 摘要文件，便于汇报时读取精确结果（规避终端显示编码）
        with io.open("scripts/_verify_out.txt", "w", encoding="utf-8") as f:
            for k, v in summary.items():
                f.write(f"{k} = {v!r}\n")


if __name__ == "__main__":
    main()
