"""会话 / 对话历史存储封装。

基于 Redis Stack（与向量存储共用同一 Redis 实例，但使用独立 key 命名空间）：

- 向量数据：`chunk:{document_id}:{index}`（见 vector_store.py）
- 会话数据：`session:{session_id}`（本模块）

每个 session 是一个 Redis List，按时间顺序存放 JSON 序列化的消息，
每条消息为 ``{"role": "user"|"assistant", "content": str, "ts": ISO8601}``。

设计策略（简单、可解释）：
- 完整历史始终保留在 Redis，受 TTL 约束（每次写入续期，活跃会话不自动过期）；
- ``get_recent_history`` 仅取最近 N 条，供后续把有限窗口历史送入 LLM，
  避免无限增长的历史直接灌给模型。本阶段不做 token counting / 摘要 / 向量化记忆。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache

from redis import Redis

from app.config import settings

# 允许的消息角色（后续 chat/completions 接线只会有这两种）
_ALLOWED_ROLES = {"user", "assistant"}


class SessionStore:
    """会话历史存取（Redis List 存储，带 TTL 自动过期）。

    会话 key 前缀默认取 ``settings.session_prefix``（生产默认 ``session:``），
    与向量 ``chunk:`` 前缀隔离。测试 / Evaluation / 集成可通过 ``prefix`` 参数
    传入独立前缀，避免与生产会话数据互相污染。
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        prefix: str | None = None,
    ) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix if prefix is not None else settings.session_prefix
        # 会话数据均为文本（JSON），无需保留二进制，故 decode_responses=True
        self._client = Redis.from_url(redis_url, decode_responses=True)

    # ---- 内部辅助 ----

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ---- 读 ----

    def get_history(self, session_id: str) -> list[dict]:
        """返回某个 session 的完整历史消息（按时间顺序，旧 → 新）。

        session 不存在时返回空列表。
        """
        raw = self._client.lrange(self._key(session_id), 0, -1)
        return [json.loads(item) for item in raw]

    def get_recent_history(
        self, session_id: str, max_messages: int | None = None
    ) -> list[dict]:
        """返回最近 ``max_messages`` 条消息（仍按时间顺序，旧 → 新）。

        Args:
            session_id: 会话标识。
            max_messages: 取最近的消息条数；缺省使用 settings.max_history_messages。
                非正数返回空列表。

        用于后续把有限窗口的历史送入 LLM；完整历史仍由 get_history 提供。
        """
        n = max_messages if max_messages is not None else settings.max_history_messages
        if n <= 0:
            return []
        raw = self._client.lrange(self._key(session_id), -n, -1)
        return [json.loads(item) for item in raw]

    # ---- 写 ----

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """向会话追加一条 user / assistant 消息。

        Args:
            role: 消息角色，仅允许 "user" / "assistant"。
            content: 消息文本（去除首尾空白后不得为空）。

        Raises:
            ValueError: role 非法，或 content 非字符串 / 为空。
        """
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"role must be one of {sorted(_ALLOWED_ROLES)}, got {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")

        message = {"role": role, "content": content, "ts": self._now_iso()}
        key = self._key(session_id)
        self._client.rpush(key, json.dumps(message, ensure_ascii=False))
        # 每次写入续期 TTL：活跃会话持续存在，静默超时的会话被自动清理
        self._client.expire(key, self.ttl_seconds)

    def clear_session(self, session_id: str) -> int:
        """清除（删除）指定 session，返回被删除的 key 数量（0 或 1）。"""
        return self._client.delete(self._key(session_id))


@lru_cache
def get_session_store() -> SessionStore:
    """返回进程级单例 SessionStore。"""
    return SessionStore(
        redis_url=settings.redis_url,
        ttl_seconds=settings.session_ttl_seconds,
        prefix=settings.session_prefix,
    )


# 便捷单例：`from app.services.session import session_store`
session_store = get_session_store()
