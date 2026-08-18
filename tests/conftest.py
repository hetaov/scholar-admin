"""pytest 共享 fixtures 与测试结构约定。

目录结构:
    tests/
    ├── conftest.py            # 本文件:共享 fixture
    ├── fakes/                 # 共享造数工厂(seed_factory)与外部依赖替身(fake_providers)
    ├── unit/                  # 单元测试:纯函数/算法,不触网不连库
    ├── integration/           # 集成测试:TestClient + FakeDB 走接口链路
    └── e2e/                   # 端到端全链路验收测试

使用约定（详见 docs/e2e-test-infra-plan.md §8 约定固化）:
- 造数统一走 tests/fakes/seed_factory（seed_content / seed_task / seed_attempt /
  seed_speech / seed_skill_states / speech_payload），禁止在测试文件内重复定义
  本地 _seed_* / _payload；差异用工厂参数或 overrides 表达。
- 外部依赖替身统一走 tests/fakes/fake_providers（FakeTtsProvider /
  FakeSpeechProvider / FakeAsrService），禁止自建 Fake*Provider 类。
- 客户端统一用 make_client 工厂（唯一入口），自动完成 get_db 双通道注入
  （方式 A: setattr 模块 get_db；方式 B: dependency_overrides）。
- 外部调用（_call_judge / _call_volcano）由 integration / e2e 各自 conftest 的
  no_external_calls autouse 默认屏蔽，测试无需也不应再显式 mock。
- 新接口/新模块的共享 fixture 加在本文件或 tests/fakes/,避免各测试文件重复定义。
"""

from __future__ import annotations

import importlib
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


@pytest.fixture()
def make_client(monkeypatch, fake_db):
    """构建带 FakeDB 注入的 TestClient 工厂（T1.2 公共基建）。

    自动完成 get_db 双通道注入,测试无需再手写 _client/_patch_db:

    - 方式 A:对每个 router 所在服务模块 setattr get_db(模块在函数体内调用
      get_db() 的场景,如 routes_tracking / routes_state / routes_dialogue)。
    - 方式 B:app.dependency_overrides 覆盖 services.dependencies.get_db
      (路由用 Depends(get_db) 捕获的场景,如 routes_evaluation / routes_tts)。
    - patch_modules:额外需要 setattr get_db 的非路由模块名
      (如 "services.dialogue_task")。
    - overrides:额外 FastAPI 依赖覆盖,如假 Provider 工厂
      {get_tts_provider: fake_provider}。

    用法:
        client = make_client(state_router, tracking_router)
        client = make_client(evaluation_router, overrides={dep: fake})
        client = make_client(dialogue_router, patch_modules=("services.dialogue_task",))
    """
    from services.dependencies import get_db as dependencies_get_db

    def _make(
        *routers,
        patch_modules: tuple[str, ...] = (),
        overrides: dict | None = None,
    ) -> TestClient:
        for router in routers:
            mod = importlib.import_module(router.__module__)
            monkeypatch.setattr(mod, "get_db", lambda: fake_db, raising=False)
        for modname in patch_modules:
            mod = importlib.import_module(modname)
            monkeypatch.setattr(mod, "get_db", lambda: fake_db, raising=False)
        app = FastAPI()
        for router in routers:
            app.include_router(router)
        # 方式 B 兜底:Depends(get_db) 场景统一覆盖
        app.dependency_overrides[dependencies_get_db] = lambda: fake_db
        if overrides:
            app.dependency_overrides.update(overrides)
        return TestClient(app)

    return _make
