"""Query Rewrite 服务单元测试。

mock DeepSeek LLM（不消耗真实 Token），验证 rewrite 的编排逻辑：
无历史短路、指代 / 省略问题触发改写、已完整问题透传、历史隔离、
空问题拒绝、LLM 空内容 / 异常 fallback、prompt 规则完整性、最小清理。
"""
from __future__ import annotations

import pytest

from app.services.llm import LLMError
from app.services.query_rewrite import rewrite


class FakeLLM:
    """记录 generate 的 prompt，返回预设结果或抛出预设异常。"""

    def __init__(self, result="星河项目的数据库备份文件保留多久？", error=None):
        self.result = result
        self.error = error
        self.calls: list[str] = []

    def generate(self, prompt):
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def fake_llm(monkeypatch):
    """将 rewrite 依赖的 get_llm_client 替换为假实现。"""
    fake = FakeLLM()
    monkeypatch.setattr("app.services.query_rewrite.get_llm_client", lambda: fake)
    return fake


# --------------------------------------------------------------------------
# 无历史 / 透传
# --------------------------------------------------------------------------

def test_no_history_returns_original_no_llm(fake_llm):
    """无历史：直接返回原问题，且不调用 LLM。"""
    q = "Redis 如何创建向量索引？"
    r = rewrite([], q)
    assert r.query == q
    assert r.used_llm is False
    assert r.fallback is False
    assert fake_llm.calls == []


def test_complete_question_still_rewritten_with_context(fake_llm):
    """已完整问题：仍走 LLM 改写，但 prompt 保留原问题原文。"""
    q = "星河项目数据库备份文件保留多少天？"
    history = [{"role": "user", "content": "之前的对话"}]
    r = rewrite(history, q)
    assert r.used_llm is True
    assert q in fake_llm.calls[0]


# --------------------------------------------------------------------------
# 指代 / 省略问题触发改写
# --------------------------------------------------------------------------

def test_referential_question_calls_llm_with_context(fake_llm):
    """指代问题：LLM 被调用，prompt 含历史与当前问题。"""
    history = [
        {"role": "user", "content": "星河项目每天什么时候备份？"},
        {"role": "assistant", "content": "每天凌晨 2:30。"},
    ]
    r = rewrite(history, "那保留多久？")
    assert r.used_llm is True
    assert r.fallback is False
    assert r.query == fake_llm.result

    prompt = fake_llm.calls[0]
    assert "星河项目每天什么时候备份？" in prompt
    assert "每天凌晨 2:30。" in prompt
    assert "那保留多久？" in prompt


def test_elliptical_question_calls_llm(fake_llm):
    """省略问题：LLM 被调用，prompt 含历史与当前问题。"""
    history = [
        {"role": "user", "content": "服务器 CPU 温度过高怎么处理？"},
        {"role": "assistant", "content": "先检查风扇。"},
    ]
    r = rewrite(history, "如果正常呢？")
    assert r.used_llm is True
    prompt = fake_llm.calls[0]
    assert "服务器 CPU 温度过高怎么处理？" in prompt
    assert "如果正常呢？" in prompt


# --------------------------------------------------------------------------
# 历史隔离
# --------------------------------------------------------------------------

def test_history_not_leaked_across_calls(fake_llm):
    """不同 history 的调用各自只携带自己的历史，互不串话。"""
    h1 = [{"role": "user", "content": "会话A的历史"}]
    h2 = [{"role": "user", "content": "会话B的历史"}]
    rewrite(h1, "那怎么办？")
    rewrite(h2, "那怎么办？")

    assert "会话A的历史" in fake_llm.calls[0]
    assert "会话B的历史" not in fake_llm.calls[0]
    assert "会话B的历史" in fake_llm.calls[1]
    assert "会话A的历史" not in fake_llm.calls[1]


# --------------------------------------------------------------------------
# prompt 规则完整性
# --------------------------------------------------------------------------

def test_prompt_contains_rewrite_instruction(fake_llm):
    """prompt 应明确「改写而非回答」、禁止新事实、只输出一条 query。"""
    rewrite([{"role": "user", "content": "历史"}], "那怎么办？")
    prompt = fake_llm.calls[0]
    assert "改写" in prompt
    assert "回答" in prompt
    assert "新事实" in prompt
    assert "一条" in prompt


# --------------------------------------------------------------------------
# 空问题拒绝
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_empty_question_rejected(bad):
    """空问题 / 纯空白问题应被拒绝（ValueError）。"""
    with pytest.raises(ValueError):
        rewrite([], bad)


# --------------------------------------------------------------------------
# LLM 失败 / 空内容 → fallback 到原问题
# --------------------------------------------------------------------------

def test_llm_empty_content_falls_back(fake_llm):
    """LLM 返回空白：回退到原问题，fallback=True。"""
    fake_llm.result = "   "
    q = "那保留多久？"
    r = rewrite([{"role": "user", "content": "历史"}], q)
    assert r.query == q
    assert r.used_llm is True
    assert r.fallback is True


def test_llm_raises_llmerror_falls_back(monkeypatch):
    """LLM 抛 LLMError（如 DeepSeek 返回空内容）：回退到原问题。"""
    fake = FakeLLM(error=LLMError("DeepSeek returned empty content"))
    monkeypatch.setattr("app.services.query_rewrite.get_llm_client", lambda: fake)
    q = "那保留多久？"
    r = rewrite([{"role": "user", "content": "历史"}], q)
    assert r.query == q
    assert r.fallback is True


def test_llm_exception_falls_back(fake_llm):
    """LLM 调用异常（如网络失败）：回退到原问题，且不抛异常。"""
    fake_llm.error = RuntimeError("connection timeout")
    q = "那保留多久？"
    r = rewrite([{"role": "user", "content": "历史"}], q)
    assert r.query == q
    assert r.used_llm is True
    assert r.fallback is True


# --------------------------------------------------------------------------
# 最小清理
# --------------------------------------------------------------------------

def test_cleanup_strips_prefix_and_multiline(fake_llm):
    """模型输出带前缀 + 多行解释时，应清理为单条 query。"""
    fake_llm.result = "改写结果：星河项目的数据库备份文件保留多久？\n下面是解释……"
    r = rewrite([{"role": "user", "content": "历史"}], "那保留多久？")
    assert r.query == "星河项目的数据库备份文件保留多久？"
    assert r.fallback is False
