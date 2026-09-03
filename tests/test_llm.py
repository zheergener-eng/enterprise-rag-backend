"""DeepSeek LLM Client 单元测试。

mock OpenAI SDK（不真实消耗 Token），验证 generate() 的正常返回与错误处理：
API Key 缺失 / 调用失败 / 空内容。
"""
from __future__ import annotations

import pytest

from app.services.llm import DeepSeekClient, LLMError


# --------------------------------------------------------------------------
# Fake OpenAI 对象（模拟 openai.OpenAI 的结构）
# --------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """记录 create 调用，返回预设 response 或抛出预设异常。"""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()


@pytest.fixture
def fake_completions(monkeypatch):
    """monkeypatch app.services.llm.OpenAI，返回可控的假 client。"""
    completions = _FakeCompletions(response=_FakeResponse("你好，我是 DeepSeek。"))
    fake = _FakeClient(completions)
    monkeypatch.setattr("app.services.llm.OpenAI", lambda **kw: fake)
    return completions


def _make_client(api_key="sk-test-key", model="deepseek-chat") -> DeepSeekClient:
    return DeepSeekClient(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model=model,
    )


# --------------------------------------------------------------------------
# 测试
# --------------------------------------------------------------------------

def test_generate_returns_content(fake_completions):
    """正常：返回模型文本，且 model / messages 正确传入。"""
    client = _make_client()
    result = client.generate("你好")
    assert result == "你好，我是 DeepSeek。"

    call = fake_completions.calls[0]
    assert call["model"] == "deepseek-chat"
    assert call["messages"] == [{"role": "user", "content": "你好"}]


def test_generate_missing_api_key(fake_completions):
    """API Key 缺失：抛 LLMError，且不发起真实调用。"""
    client = _make_client(api_key="")
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        client.generate("你好")
    assert fake_completions.calls == []  # 未调用 create


def test_generate_api_call_failure(fake_completions):
    """API 调用失败（网络/服务端）：包装为 LLMError。"""
    fake_completions.error = RuntimeError("connection timeout")
    client = _make_client()
    with pytest.raises(LLMError, match="call failed"):
        client.generate("你好")


@pytest.mark.parametrize("content", [None, "", "   "])
def test_generate_empty_content(fake_completions, content):
    """模型返回空内容：抛 LLMError。"""
    fake_completions.response = _FakeResponse(content)
    client = _make_client()
    with pytest.raises(LLMError, match="empty content"):
        client.generate("你好")
