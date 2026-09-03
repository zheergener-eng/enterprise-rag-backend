"""真实 API 端到端验证脚本。

启动 uvicorn（真实 Redis + DeepSeek），用 httpx 依次验证：
  POST   /api/v1/chat/completions   第一轮（答「2:30」）
  POST   /api/v1/chat/completions   第二轮「那保留多久？」（答「14 天」）
  GET    /api/v1/chat/history/{id}  四条历史 user/assistant/user/assistant
  DELETE /api/v1/chat/session/{id}  cleared=true
  GET    /api/v1/chat/history/{id}  空历史

前置条件：Redis Stack 已运行；.env 已配置 DEEPSEEK_API_KEY。
用法：python -m scripts.verify_api   （或  PYTHONPATH=. python scripts/verify_api.py）
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import time

import httpx

from app.config import settings
from app.services.chunking import split_document
from app.services.embeddings import get_embedder
from app.services.vector_store import VectorStore

HOST = "127.0.0.1"
PORT = 8765
BASE = f"http://{HOST}:{PORT}"

# 验证脚本使用独立 index / prefix，绝不触碰生产默认 rag:index / chunk:。
VERIFY_INDEX = "verify:rag:index"
VERIFY_PREFIX = "verify:chunk:"


def _build_store() -> VectorStore:
    return VectorStore(
        redis_url=settings.redis_url,
        index_name=VERIFY_INDEX,
        dim=settings.embedding_dim,
        prefix=VERIFY_PREFIX,
    )


def _seed_knowledge_base() -> None:
    """直接向 Redis 写入单文档知识库（含「2:30 备份」「保留 14 天」）。"""
    document = (
        "# 星河项目运维手册\n\n"
        "## 数据库备份\n\n"
        "星河项目数据库每天凌晨 2:30 执行备份。\n"
        "备份文件保留 14 天。\n"
    )
    embedder = get_embedder()
    store = _build_store()
    store.drop_index()
    store.ensure_index()
    chunks = split_document(document, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    texts = [c.text for c in chunks]
    store.add_document("star-river-api", texts, embedder.embed_documents(texts), title="星河项目运维手册")


def _start_server() -> subprocess.Popen:
    log = io.open("scripts/_verify_api_server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", HOST, "--port", str(PORT)],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        if proc.poll() is not None:
            log.close()
            with io.open("scripts/_verify_api_server.log", "r", encoding="utf-8") as f:
                raise RuntimeError(f"uvicorn 提前退出:\n{f.read()}")
        try:
            if httpx.get(f"{BASE}/health", timeout=1, trust_env=False).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(0.25)
    proc.terminate()
    log.close()
    with io.open("scripts/_verify_api_server.log", "r", encoding="utf-8") as f:
        raise RuntimeError(f"uvicorn 未能在预期时间内就绪:\n{f.read()}")


def _post_sse(client: httpx.Client, payload: dict) -> tuple[int, str, list[tuple[str, dict]]]:
    """POST /chat/completions，流式读取 SSE，返回 (status, content_type, events)。"""
    events: list[tuple[str, dict]] = []
    with client.stream("POST", f"{BASE}/api/v1/chat/completions", json=payload) as resp:
        status = resp.status_code
        content_type = resp.headers.get("content-type", "")
        event = None
        data_lines: list[str] = []

        def flush():
            nonlocal event, data_lines
            if event is not None:
                events.append((event, json.loads("".join(data_lines))))
            event = None
            data_lines = []

        for line in resp.iter_lines():
            if line == "":
                flush()
            elif line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: "):])
        flush()
    return status, content_type, events


def main() -> None:
    summary: dict[str, object] = {}
    proc = None
    try:
        _seed_knowledge_base()
        proc = _start_server()

        # trust_env=False：避免本机环境代理拦截 127.0.0.1 回环请求（否则 health/chat 请求会走代理）
        with httpx.Client(timeout=60, trust_env=False) as client:
            # ---- 第一轮 ----
            q1 = "星河项目每天什么时候进行数据库备份？"
            status, ctype, events = _post_sse(client, {"session_id": "portfolio-demo-001", "message": q1})
            names = [e for e, _ in events]
            answer1 = "".join(d["content"] for e, d in events if e == "delta")
            print(f"[第一轮] {q1}")
            print(f"  - status={status} content-type={ctype!r}")
            print(f"  - events={names}")
            print(f"  - answer: {answer1}")
            assert status == 200
            assert ctype.startswith("text/event-stream")
            assert "delta" in names and names[-1] == "done"
            assert "2:30" in answer1
            summary["turn1_answer"] = answer1

            # ---- 第二轮（同一 session，指代） ----
            q2 = "那保留多久？"
            status, ctype, events = _post_sse(client, {"session_id": "portfolio-demo-001", "message": q2})
            names = [e for e, _ in events]
            answer2 = "".join(d["content"] for e, d in events if e == "delta")
            print(f"\n[第二轮] {q2}")
            print(f"  - status={status} events={names}")
            print(f"  - answer: {answer2}")
            assert status == 200
            assert names[-1] == "done"
            assert "14" in answer2
            summary["turn2_answer"] = answer2

            # ---- GET history ----
            hist = client.get(f"{BASE}/api/v1/chat/history/portfolio-demo-001").json()
            print("\n[GET history]")
            for m in hist["messages"]:
                print(f"  - {m['role']}: {m['content']}")
            roles = [m["role"] for m in hist["messages"]]
            assert roles == ["user", "assistant", "user", "assistant"]
            assert hist["messages"][0]["content"] == q1
            assert hist["messages"][2]["content"] == q2
            summary["history_roles"] = roles

            # ---- DELETE ----
            deleted = client.delete(f"{BASE}/api/v1/chat/session/portfolio-demo-001").json()
            print(f"\n[DELETE] {deleted}")
            assert deleted["cleared"] is True
            summary["delete_cleared"] = deleted["cleared"]

            # ---- GET 再次（空） ----
            hist2 = client.get(f"{BASE}/api/v1/chat/history/portfolio-demo-001").json()
            print(f"[GET after delete] messages={hist2['messages']}")
            assert hist2["messages"] == []
            summary["history_after_delete"] = hist2["messages"]

        print("\n[结论] 真实 API 端到端验证通过。")
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=5)
        _build_store().drop_index()
        with io.open("scripts/_verify_api.txt", "w", encoding="utf-8") as f:
            for k, v in summary.items():
                f.write(f"{k} = {v!r}\n")


if __name__ == "__main__":
    main()
