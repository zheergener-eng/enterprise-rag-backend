"""DeepSeek LLM Client 流式单元测试。

mock OpenAI SDK（不真实消耗 Token），验证 stream_generate() 的正常产出与错误处理：
多 chunk 顺序 yield / None 与空串跳过 / API Key 缺失 / 建流前失败 / 中途失败 /
多次调用复用 client / 流资源关闭。
"""
from __future__ import annotations

import pytest

from app.services.llm import DeepSeekClient, LLMError


# --------------------------------------------------------------------------
# Fake OpenAI 流式对象（模拟 openai.OpenAI 的 streaming 结构）
# --------------------------------------------------------------------------

class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChunk:
    """模拟 ChatCompletionChunk：choices[0].delta.content。"""

    def __init__(self, content):
        # 传入 None 表示 chunk 无 choices；否则构造带 delta 的 choice
        if content is _NO_CHOICES:
            self.choices = []
        else:
            self.choices = [type("Choice", (), {"delta": _FakeDelta(content)})()]


class _FakeStream:
    """可迭代 + 上下文管理器，模拟 OpenAI 的 Stream 对象。"""

    def __init__(self, contents, error=None):
        self.contents = list(contents)
        self.error = error  # 迭代到末尾后抛出（模拟中途失败）
        self._it = iter(self.contents)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return _FakeChunk(next(self._it))
        except StopIteration:
            if self.error is not None:
                raise self.error
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False


class _FakeCompletions:
    """记录 create 调用；stream=True 返回 _FakeStream，否则抛预设异常。"""

    def __init__(self, contents=(), error=None, create_error=None):
        self.contents = list(contents)
        self.error = error  # 流迭代中途异常
        self.create_error = create_error  # create() 本身异常（建流前）
        self.calls: list[dict] = []
        self.streams: list[_FakeStream] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.create_error is not None:
            raise self.create_error
        stream = _FakeStream(self.contents, error=self.error)
        self.streams.append(stream)
        return stream


class _FakeClient:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()


# 哨兵：表示该 chunk 没有 choices
_NO_CHOICES = object()


@pytest.fixture
def fake_completions(monkeypatch):
    completions = _FakeCompletions()
    monkeypatch.setattr(
        "app.services.llm.OpenAI", lambda **kw: _FakeClient(completions)
    )
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

def test_stream_yields_chunks_in_order(fake_completions):
    """正常：多个非空 chunk 按顺序 yield，且 stream=True / model / messages 正确。"""
    fake_completions.contents = ["星河", "项目", "每天", "凌晨 2:30", "备份。"]
    client = _make_client()
    assert list(client.stream_generate("你好")) == [
        "星河",
        "项目",
        "每天",
        "凌晨 2:30",
        "备份。",
    ]
    call = fake_completions.calls[0]
    assert call["stream"] is True
    assert call["model"] == "deepseek-chat"
    assert call["messages"] == [{"role": "user", "content": "你好"}]


def test_stream_skips_none_content(fake_completions):
    """None content 被跳过，只 yield 非空文本。"""
    fake_completions.contents = ["星河", None, "项目", None, "备份。"]
    client = _make_client()
    assert list(client.stream_generate("x")) == ["星河", "项目", "备份。"]


def test_stream_skips_empty_string(fake_completions):
    """空字符串 content 被跳过。"""
    fake_completions.contents = ["星河", "", "项目", "", "备份。"]
    client = _make_client()
    assert list(client.stream_generate("x")) == ["星河", "项目", "备份。"]


def test_stream_skips_chunk_without_choices(fake_completions):
    """没有 choices 的 chunk 被跳过。"""
    fake_completions.contents = ["星河", _NO_CHOICES, "项目", _NO_CHOICES]
    client = _make_client()
    assert list(client.stream_generate("x")) == ["星河", "项目"]


def test_stream_missing_api_key(fake_completions):
    """API Key 缺失：迭代时抛 LLMError，且不发起 create。"""
    client = _make_client(api_key="")
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        list(client.stream_generate("你好"))
    assert fake_completions.calls == []


def test_stream_create_failure_before_first_token(fake_completions):
    """建立流前失败（create 抛异常）：包装为 LLMError，标记“首个 token 前”。"""
    fake_completions.create_error = RuntimeError("connection refused")
    client = _make_client()
    with pytest.raises(LLMError, match="before first token"):
        list(client.stream_generate("你好"))


def test_stream_mid_generation_failure(fake_completions):
    """流中途异常：已产出的 chunk 先 yield，随后抛 LLMError 标记“中途”。"""
    fake_completions.contents = ["星河", "项目"]
    fake_completions.error = RuntimeError("connection reset")

    client = _make_client()
    collected: list[str] = []
    with pytest.raises(LLMError, match="mid-generation"):
        for chunk in client.stream_generate("你好"):
            collected.append(chunk)
    assert collected == ["星河", "项目"]  # 失败前已产出部分内容


def test_stream_reuses_single_openai_client(monkeypatch):
    """多次 stream_generate 复用同一 OpenAI client（OpenAI 仅被构造一次）。"""
    constructed: list = []
    monkeypatch.setattr(
        "app.services.llm.OpenAI",
        lambda **kw: constructed.append(kw) or _FakeClient(_FakeCompletions(["a"])),
    )
    client = _make_client()
    list(client.stream_generate("q1"))
    list(client.stream_generate("q2"))
    assert len(constructed) == 1  # _get_client 懒加载后复用同一实例


def test_stream_closes_connection_on_success(fake_completions):
    """正常结束：流对象通过 with 正确关闭（资源释放）。"""
    fake_completions.contents = ["星河", "项目"]
    client = _make_client()
    list(client.stream_generate("x"))
    assert fake_completions.streams[0].closed is True


def test_stream_closes_connection_on_error(fake_completions):
    """异常结束：流对象同样被正确关闭。"""
    fake_completions.contents = ["星河"]
    fake_completions.error = RuntimeError("boom")
    client = _make_client()
    with pytest.raises(LLMError):
        list(client.stream_generate("x"))
    assert fake_completions.streams[0].closed is True
