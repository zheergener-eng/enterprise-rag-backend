"""极简 SSE（Server-Sent Events）编码辅助。

只负责把 (event, data) 编码为符合 text/event-stream 规范的字符串片段：

    event: <event>
    data: <json>

    <空行>

不引入任何 SSE 框架；FastAPI 的 StreamingResponse 已足够承载流式输出。
"""
from __future__ import annotations

import json
from typing import Any


def encode_sse(event: str, data: dict[str, Any]) -> str:
    """编码单个 SSE event。

    Args:
        event: event 名（如 "delta" / "done" / "error"）。
        data: 将作为 JSON 序列化到 data 行；使用 ``ensure_ascii=False`` 保留中文原文。

    Returns:
        形如 ``event: delta\\ndata: {"content": "..."}\\n\\n`` 的字符串片段，
        以空行结束一个 event。
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
