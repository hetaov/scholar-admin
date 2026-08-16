"""路由表冒烟测试：main.app 挂载完整性（防「新增路由未 include / 路径拼错」导致线上 404）

背景：4.6.5c 真机联调发现 /eval/translate 线上 404 —— 云托管未部署新代码；
本地 main.app 的路由表完整性此前无测试兜底（routes_* 单测均为独立 FastAPI 挂载，不经过 main.py）。
本测试遍历 main.app 全部路由（含 _IncludedRouter 内层），断言小程序端实际依赖的关键路由已挂载。
"""
from __future__ import annotations

from main import app

# 小程序端实际依赖的关键路由（缺一即线上 404）
EXPECTED_ROUTES = {
    ("GET", "/health"),
    ("POST", "/eval/translate"),          # 4.6.5b 翻译评估（routes_eval）
    ("POST", "/match/dialogue"),          # 对话匹配（routes_dialogue）
    ("POST", "/tracking/state"),          # 单句状态上报 / Skill Attempt（routes_state）
    ("GET", "/tracking/{scholar_id}"),    # 单学员掌握度查询（routes_tracking）
}


def _collect_routes():
    """展开 main.app 路由表：_IncludedRouter 无 .routes，需取 .original_router（APIRouter）内层。"""
    collected = set()
    for route in app.routes:
        inner = getattr(route, "original_router", None)
        candidates = inner.routes if inner is not None else [route]
        for r in candidates:
            methods = getattr(r, "methods", None)
            path = getattr(r, "path", None)
            if methods and path:
                for m in methods:
                    collected.add((m, path))
    return collected


def test_key_routes_mounted():
    routes = _collect_routes()
    missing = EXPECTED_ROUTES - routes
    assert not missing, "main.app 缺失路由: " + ", ".join(
        f"{m} {p}" for m, p in sorted(missing)
    )
