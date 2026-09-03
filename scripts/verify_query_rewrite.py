"""真实 Query Rewrite 验证脚本。

验证目标：
  1. 有历史 + 指代问题 → 改写为独立 standalone query（不回答 14 天、不解释、单条 query）；
  2. 无历史 → 直接返回原问题，不调用 DeepSeek。

前置条件：.env 已配置 DEEPSEEK_API_KEY。
用法：python -m scripts.verify_query_rewrite   （或  PYTHONPATH=. python scripts/verify_query_rewrite.py）
"""
from __future__ import annotations

from app.services.query_rewrite import rewrite


def main() -> None:
    # ---- 情况 1：有历史，指代问题 ----
    history = [
        {"role": "user", "content": "星河项目每天什么时候进行数据库备份？"},
        {"role": "assistant", "content": "每天凌晨 2:30。"},
    ]
    q1 = "那保留多久？"
    r1 = rewrite(history, q1)
    print(f"[指代问题] 历史 2 条，当前问题：{q1}")
    print(f"  - used_llm={r1.used_llm} fallback={r1.fallback}")
    print(f"  - rewrite 结果：{r1.query}")

    assert r1.used_llm is True and r1.fallback is False, "有历史应调用 LLM 且不 fallback"
    assert "星河项目" in r1.query, "应恢复指代对象：星河项目"
    assert "备份" in r1.query, "应保留话题：备份"
    assert any(k in r1.query for k in ("保留", "多久", "多少")), "应保留原始意图"
    assert "14" not in r1.query, "不应回答 14 天"
    assert "\n" not in r1.query, "只输出一条 query，不应含多行解释"

    # ---- 情况 2：无历史，直接返回原问题 ----
    q2 = "如何创建 Redis 向量索引？"
    r2 = rewrite([], q2)
    print(f"\n[无历史] 当前问题：{q2}")
    print(f"  - used_llm={r2.used_llm} fallback={r2.fallback}")
    print(f"  - rewrite 结果：{r2.query}")
    assert r2.used_llm is False, "无历史不应调用 LLM"
    assert r2.fallback is False
    assert r2.query == q2

    print("\n[结论] 真实 Query Rewrite 验证通过。")


if __name__ == "__main__":
    main()
