"""API 层骨架测试。

当前仅验证应用可启动、核心路由已挂载；后续阶段补充完整业务测试。
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health():
    """健康检查应返回 200。"""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_routes_registered():
    """确认四个核心接口均已注册。

    注：本版本 FastAPI 的 include_router 会生成惰性 _IncludedRouter 对象，
    直接遍历 app.routes 无法取到 path，故改用 app.openapi() 暴露的真实路径。
    """
    paths = set(app.openapi()["paths"].keys())
    assert "/api/v1/document/import" in paths
    assert "/api/v1/chat/completions" in paths
    assert "/api/v1/chat/history/{session_id}" in paths
    assert "/api/v1/chat/session/{session_id}" in paths
