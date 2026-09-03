"""真实 DeepSeek stream=True 验证脚本。

验证 stream_generate() 对真实 DeepSeek OpenAI 兼容 API 的行为：
- 收到的非空 chunk 数量；
- 拼接后的完整 answer；
- 所有 chunk 均非空。

前置条件：.env 已配置 DEEPSEEK_API_KEY（需联网）。
用法：python -m scripts.verify_llm_streaming   （或  PYTHONPATH=. python scripts/verify_llm_streaming.py）
"""
from __future__ import annotations

import io

from app.services.llm import get_llm_client


def main() -> None:
    prompt = (
        "请用中文回答：'Retrieval-Augmented Generation' 的核心思想是什么？"
        "用 2-3 句话说明，不要使用列表或 Markdown。"
    )

    client = get_llm_client()
    chunks: list[str] = []
    for chunk in client.stream_generate(prompt):
        chunks.append(chunk)

    full_answer = "".join(chunks)

    # 校验：所有增量非空
    assert chunks, "应至少收到一个非空 chunk"
    assert all(c for c in chunks), "不应存在空字符串 chunk"

    print(f"[chunk 数量] {len(chunks)}")
    print(f"[完整 answer] {full_answer}")

    summary = {
        "prompt": prompt,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "full_answer": full_answer,
        "full_answer_len": len(full_answer),
    }
    with io.open("scripts/_verify_stream_llm.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k} = {v!r}\n")

    print("\n[结论] 真实 DeepSeek 流式验证通过。")


if __name__ == "__main__":
    main()
