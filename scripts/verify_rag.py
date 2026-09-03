"""真实端到端 RAG 验证脚本。

链路：文档 → Chunking → Embedding → Redis → 提问 → Retrieval → Prompt → DeepSeek → Answer。

验证目标：
  1. 正确问题能检索到正确 chunk，回答依据知识库给出明确事实（凌晨 2:30）；
  2. 无依据问题不编造，明确表示知识库无法确定。

前置条件：Redis Stack 已运行；.env 已配置 DEEPSEEK_API_KEY。
用法：python -m scripts.verify_rag   （或  PYTHONPATH=. python scripts/verify_rag.py）
"""
from __future__ import annotations

from app.config import settings
from app.services.chunking import split_document
from app.services.embeddings import get_embedder
from app.services.rag import answer_question, build_context, build_prompt
from app.services.retrieval import retrieve_with_rerank
from app.services.vector_store import VectorStore

# 验证脚本使用独立 index / prefix，绝不触碰生产默认 rag:index / chunk:。
VERIFY_INDEX = "verify:rag:index"
VERIFY_PREFIX = "verify:chunk:"


def main() -> None:
    document = (
        "# 星河项目运维手册\n\n"
        "## 数据库备份\n\n"
        "星河项目的数据库备份时间为每天凌晨 2:30，备份文件保留 14 天。\n\n"
        "## 备份策略\n\n"
        "数据库采用每日增量备份策略，全量备份每周一次。\n\n"
        "## 服务器部署\n\n"
        "生产服务器部署在华东机房，共 8 台节点。\n"
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

    try:
        chunks = split_document(
            document, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
        )
        texts = [c.text for c in chunks]
        vectors = embedder.embed_documents(texts)
        store.add_document("star-river", texts, vectors, title="星河项目运维手册")
        print(f"[准备] 文档切分为 {len(texts)} 个 chunk，已写入 Redis\n")

        # ---- 问题 1：有明确依据 ----
        q1 = "星河项目每天什么时候进行数据库备份？"
        chunks1 = retrieve_with_rerank(q1)
        print(f"[问题1] {q1}")
        print(f"  - 检索到 {len(chunks1)} 个 chunk，首个：{chunks1[0].text}")
        prompt1 = build_prompt(q1, build_context(chunks1))
        print(f"  - prompt 含 '2:30'：{'2:30' in prompt1}")
        result1 = answer_question(q1)
        print(f"  - 最终回答：{result1.answer}")
        assert "2:30" in chunks1[0].text, "正确 chunk 未被检索到"
        assert "2:30" in result1.answer, "回答未依据知识库给出备份时间"

        # ---- 问题 2：知识库无依据 ----
        q2 = "星河项目负责人是谁？"
        chunks2 = retrieve_with_rerank(q2)
        result2 = answer_question(q2)
        print(f"\n[问题2] {q2}")
        print(f"  - 检索到 {len(chunks2)} 个 chunk（相关但均不含负责人信息）")
        print(f"  - 最终回答：{result2.answer}")
        # 不编造：回答应明确表示无法确定，而非给出具体人名
        assert any(k in result2.answer for k in ("无法确定", "不确定", "没有")), \
            "无依据问题应明确表示无法确定"

        print("\n[结论] 真实端到端 RAG 验证通过。")
    finally:
        store.drop_index()


if __name__ == "__main__":
    main()
