"""Session / Conversation History 阶段测试。

包含三类：
- 基于假 Redis 客户端的单元测试（快，验证 SessionStore 逻辑）
- 基于 TestClient + 假 session service 的 API 测试（GET history / DELETE session）
- 基于真实 Redis 的集成测试（往返读写、TTL、隔离、清理）
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.session import SessionStore


# --------------------------------------------------------------------------
# 假 Redis（单元测试用，不依赖真实 Redis）
# --------------------------------------------------------------------------

class FakeRedis:
    """模拟 redis.Redis 的 List 接口（内存实现，语义对齐 LRANGE/RPUSH/EXPIRE/DELETE）。"""

    def __init__(self):
        self.lists: dict[str, list[str]] = {}
        self.ttls: dict[str, int] = {}
        self.deleted_keys: list[str] = []

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        n = len(items)
        if n == 0:
            return []

        def norm(i):
            return i + n if i < 0 else i

        s = max(norm(start), 0)
        e = min(norm(end), n - 1)
        if s > e:
            return []
        return list(items[s : e + 1])

    def expire(self, key, ttl):
        self.ttls[key] = ttl
        return True

    def delete(self, key):
        self.deleted_keys.append(key)
        if key in self.lists:
            del self.lists[key]
            self.ttls.pop(key, None)
            return 1
        return 0


@pytest.fixture
def fake_redis(monkeypatch):
    """将 SessionStore 依赖的 Redis 替换为内存假实现。"""
    fake = FakeRedis()

    class _FakeRedisClient:
        @classmethod
        def from_url(cls, url, **kwargs):
            return fake

    monkeypatch.setattr("app.services.session.Redis", _FakeRedisClient)
    return fake


def _store(ttl: int = 3600) -> SessionStore:
    return SessionStore(redis_url="redis://localhost:6379", ttl_seconds=ttl)


# --------------------------------------------------------------------------
# 单元测试：SessionStore 逻辑
# --------------------------------------------------------------------------

def test_new_session_history_empty(fake_redis):
    """新 session 初始历史为空。"""
    assert _store().get_history("sess-new") == []


def test_add_user_message(fake_redis):
    """add user 消息成功，返回含 role/content/ts 的消息。"""
    store = _store()
    store.add_message("sess-1", "user", "你好")
    h = store.get_history("sess-1")
    assert len(h) == 1
    assert h[0]["role"] == "user"
    assert h[0]["content"] == "你好"
    assert isinstance(h[0]["ts"], str) and h[0]["ts"]


def test_add_assistant_message(fake_redis):
    """add assistant 消息成功，与 user 消息并存。"""
    store = _store()
    store.add_message("sess-1", "user", "你好")
    store.add_message("sess-1", "assistant", "你好，有什么可以帮你？")
    h = store.get_history("sess-1")
    assert [m["role"] for m in h] == ["user", "assistant"]
    assert h[1]["content"] == "你好，有什么可以帮你？"


def test_message_order_preserved(fake_redis):
    """消息按写入顺序排列（旧 → 新）。"""
    store = _store()
    for i in range(3):
        store.add_message("sess-1", "user", f"消息{i}")
    assert [m["content"] for m in store.get_history("sess-1")] == ["消息0", "消息1", "消息2"]


def test_sessions_isolated(fake_redis):
    """不同 session 相互隔离，互不影响。"""
    store = _store()
    store.add_message("sess-a", "user", "A")
    store.add_message("sess-b", "user", "B")
    assert [m["content"] for m in store.get_history("sess-a")] == ["A"]
    assert [m["content"] for m in store.get_history("sess-b")] == ["B"]


def test_get_history_returns_full(fake_redis):
    """get_history 返回完整历史（不被 recent 限制）。"""
    store = _store()
    for i in range(5):
        store.add_message("sess-1", "user", f"消息{i}")
    assert len(store.get_history("sess-1")) == 5


def test_get_recent_history_limits(fake_redis):
    """get_recent_history 能正确限制数量，且仍按旧 → 新顺序。"""
    store = _store()
    for i in range(5):
        store.add_message("sess-1", "user", f"消息{i}")
    recent = store.get_recent_history("sess-1", max_messages=2)
    assert [m["content"] for m in recent] == ["消息3", "消息4"]


def test_get_recent_history_default_uses_settings(fake_redis):
    """未显式传 max_messages 时，使用 settings.max_history_messages。"""
    store = _store()
    total = settings.max_history_messages + 3
    for i in range(total):
        store.add_message("sess-1", "user", f"消息{i}")
    recent = store.get_recent_history("sess-1")
    assert len(recent) == settings.max_history_messages
    # 取的是「最近」的 N 条
    assert recent[0]["content"] == f"消息{total - settings.max_history_messages}"


def test_get_recent_history_non_positive_empty(fake_redis):
    """max_messages <= 0 时返回空列表。"""
    store = _store()
    store.add_message("sess-1", "user", "你好")
    assert store.get_recent_history("sess-1", max_messages=0) == []
    assert store.get_recent_history("sess-1", max_messages=-1) == []


def test_clear_session_empties(fake_redis):
    """clear_session 删除后，历史为空，且返回删除的 key 数 1。"""
    store = _store()
    store.add_message("sess-1", "user", "你好")
    assert store.clear_session("sess-1") == 1
    assert store.get_history("sess-1") == []


def test_clear_missing_session_returns_zero(fake_redis):
    """清除不存在的 session 返回 0。"""
    assert _store().clear_session("sess-missing") == 0


def test_add_message_invalid_role(fake_redis):
    """非法 role 被拒绝（ValueError）。"""
    store = _store()
    for bad in ["system", "tool", ""]:
        with pytest.raises(ValueError):
            store.add_message("sess-1", bad, "内容")


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_add_message_empty_content(fake_redis, bad):
    """空 / 纯空白 content 被拒绝（ValueError）。"""
    with pytest.raises(ValueError):
        _store().add_message("sess-1", "user", bad)


def test_ttl_set_on_add(fake_redis):
    """add_message 写入后应设置 TTL（用于自动过期）。"""
    store = _store(ttl=1234)
    store.add_message("sess-1", "user", "你好")
    assert fake_redis.ttls["session:sess-1"] == 1234


def test_custom_session_prefix(fake_redis):
    """自定义 prefix 时，session key 使用该前缀（测试 / Evaluation 可隔离）。"""
    store = SessionStore(redis_url="redis://localhost:6379", ttl_seconds=3600, prefix="test:")
    store.add_message("sess-1", "user", "你好")
    assert "test:sess-1" in fake_redis.lists
    assert "session:sess-1" not in fake_redis.lists


# --------------------------------------------------------------------------
# API 测试（mock session service，验证 GET history / DELETE session）
# --------------------------------------------------------------------------

class FakeSessionService:
    """记录调用、返回预设历史，模拟 clear 返回值。"""

    def __init__(self, history=None, deleted: int = 1):
        self.history = history or []
        self.deleted = deleted
        self.get_calls: list[str] = []
        self.clear_calls: list[str] = []

    def get_history(self, session_id):
        self.get_calls.append(session_id)
        return self.history

    def clear_session(self, session_id):
        self.clear_calls.append(session_id)
        return self.deleted


@pytest.fixture
def fake_session_service(monkeypatch):
    service = FakeSessionService(
        history=[
            {"role": "user", "content": "你好", "ts": "2026-09-01T10:00:00+00:00"},
            {"role": "assistant", "content": "你好，有什么可以帮你？", "ts": "2026-09-01T10:00:01+00:00"},
        ]
    )
    monkeypatch.setattr("app.api.v1.chat.get_session_store", lambda: service)
    return service


def test_history_api_returns_history(fake_session_service):
    """GET /chat/history/{id} 正常返回完整历史。"""
    with TestClient(app) as client:
        resp = client.get("/api/v1/chat/history/sess-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-1"
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "你好"
    assert data["messages"][0]["ts"]
    assert fake_session_service.get_calls == ["sess-1"]


def test_history_api_empty_session(monkeypatch):
    """新 session 的 history API 返回空 messages 列表。"""
    service = FakeSessionService(history=[])
    monkeypatch.setattr("app.api.v1.chat.get_session_store", lambda: service)
    with TestClient(app) as client:
        resp = client.get("/api/v1/chat/history/sess-empty")
    assert resp.status_code == 200
    assert resp.json() == {"session_id": "sess-empty", "messages": []}


def test_delete_session_api_cleared(fake_session_service):
    """DELETE /chat/session/{id}：会话存在时返回 cleared=true。"""
    fake_session_service.deleted = 1
    with TestClient(app) as client:
        resp = client.delete("/api/v1/chat/session/sess-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-1"
    assert data["cleared"] is True
    assert fake_session_service.clear_calls == ["sess-1"]


def test_delete_session_api_already_empty(fake_session_service):
    """DELETE /chat/session/{id}：会话不存在时返回 cleared=false（幂等，仍 200）。"""
    fake_session_service.deleted = 0
    with TestClient(app) as client:
        resp = client.delete("/api/v1/chat/session/sess-none")
    assert resp.status_code == 200
    assert resp.json()["cleared"] is False


# --------------------------------------------------------------------------
# 真实 Redis 集成测试（需要真实 Redis Stack 运行）
# --------------------------------------------------------------------------

@pytest.fixture
def real_store():
    """真实 Redis 的 SessionStore（短 TTL），测试结束后清理产生的 key。"""
    store = SessionStore(redis_url=settings.redis_url, ttl_seconds=60)
    created: list[str] = []

    def track(sid: str) -> str:
        created.append(sid)
        return sid

    yield store, track
    for sid in created:
        store.clear_session(sid)


@pytest.mark.integration
def test_real_new_session_empty(real_store):
    """真实 Redis：新 session 历史为空。"""
    store, track = real_store
    sid = track("test:session:new-empty")
    assert store.get_history(sid) == []


@pytest.mark.integration
def test_real_add_and_get_history(real_store):
    """真实 Redis：写入 user/assistant 消息后正确读回，含 ts。"""
    store, track = real_store
    sid = track("test:session:add-get")
    store.add_message(sid, "user", "你好")
    store.add_message(sid, "assistant", "你好，有什么可以帮你？")
    h = store.get_history(sid)
    assert [m["role"] for m in h] == ["user", "assistant"]
    assert h[0]["content"] == "你好"
    assert h[1]["content"] == "你好，有什么可以帮你？"
    assert all("ts" in m and m["ts"] for m in h)


@pytest.mark.integration
def test_real_recent_history_limits(real_store):
    """真实 Redis：get_recent_history 限制数量，完整历史仍保留。"""
    store, track = real_store
    sid = track("test:session:recent")
    for i in range(5):
        store.add_message(sid, "user" if i % 2 == 0 else "assistant", f"消息{i}")
    recent = store.get_recent_history(sid, max_messages=2)
    assert [m["content"] for m in recent] == ["消息3", "消息4"]
    assert len(store.get_history(sid)) == 5


@pytest.mark.integration
def test_real_sessions_isolated(real_store):
    """真实 Redis：不同 session 相互隔离。"""
    store, track = real_store
    s1 = track("test:session:iso-a")
    s2 = track("test:session:iso-b")
    store.add_message(s1, "user", "A")
    store.add_message(s2, "user", "B")
    assert [m["content"] for m in store.get_history(s1)] == ["A"]
    assert [m["content"] for m in store.get_history(s2)] == ["B"]


@pytest.mark.integration
def test_real_clear_session(real_store):
    """真实 Redis：clear_session 删除历史，二次删除返回 0。"""
    store, track = real_store
    sid = track("test:session:clear")
    store.add_message(sid, "user", "你好")
    assert store.clear_session(sid) == 1
    assert store.get_history(sid) == []
    assert store.clear_session(sid) == 0
