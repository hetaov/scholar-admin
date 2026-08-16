"""pytest 共享 fixtures 与测试结构约定。

目录结构:
    tests/
    ├── conftest.py            # 本文件:共享 fixture
    ├── fakes/                 # 外部依赖的内存替身(数据库/视觉服务)
    ├── unit/                  # 单元测试:纯函数/算法,不触网不连库
    └── integration/           # 集成测试:TestClient + FakeDB 走接口链路

使用约定:
- 需要"假数据库"的测试直接请求 fake_db fixture,或 client fixture(已注入 FakeDB)。
- 新接口/新模块的共享 fixture 加在本文件,避免各测试文件重复定义。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# 保证以项目根目录为包根路径(conftest 在收集阶段先于测试模块加载)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.routes_tracking import router as tracking_router  # noqa: E402
from tests.fakes.fake_db import FakeDB  # noqa: E402


@pytest.fixture()
def fake_db() -> FakeDB:
    """内存假数据库,默认空。测试按需用 fake_db.add 预置新表数据。"""
    return FakeDB()


@pytest.fixture()
def client(monkeypatch, fake_db):
    """FastAPI TestClient,get_db 注入为 FakeDB。

    默认替换 services.routes_tracking 中的 get_db 引用;
    其他路由模块如需假库,在各自测试中追加:
        monkeypatch.setattr("services.<module>.get_db", lambda: fake_db)
    """
    monkeypatch.setattr("services.routes_tracking.get_db", lambda: fake_db)
    app = FastAPI()
    app.include_router(tracking_router)
    return TestClient(app)
