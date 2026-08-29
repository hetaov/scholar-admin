"""e2e 端到端全链路验收测试专用 conftest。

与 tests/conftest.py（根）的关系：
- 复用根 conftest 的 fake_db / make_client（get_db 方式 B dependency_overrides
  注入、Provider overrides 注入基建）。
- 本目录追加 autouse 屏蔽真实外部 LLM/大模型调用（与 integration 目录一致），
  保证 e2e 验收不触网；ASR / SOE-N / TTS 等 Provider 由各测试文件
  显式注入 tests/fakes/fake_providers.py 替身。

关于 get_db 方式 A（函数体内直接调用 get_db() 的模块）：
根 conftest.make_client 通过 router.__module__ 定位服务模块，但 APIRouter
实例的 __module__ 恒为 fastapi.routing，该注入对方式 A 模块无效；
故此处复刻 integration 目录的 fake_db_auto_inject：AST 检测所有
services.* 模块源码中是否存在 `get_db(...)` 直接调用，命中则 monkeypatch
指向当前测试 fake_db。e2e 因此无需为每个路由模块手写注入。

e2e 与 integration 的区别：
- integration：单路由/单接口的局部集成，重点验证单个端点的输入输出。
- e2e：跨路由的 P0 业务全链路闭环（评估 → 状态上报 → 追踪查询），
  模拟真实前端调用顺序，验证数据在链路间正确流转与落库。
"""

from __future__ import annotations

import ast
import inspect
import sys

import pytest


def _calls_get_db_in_body(mod) -> bool:
    """模块源码中是否存在 `get_db(...)` 直接调用（非 Depends(get_db) 参数形式）。"""
    try:
        source = inspect.getsource(mod)
    except (OSError, TypeError):
        return False
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "get_db":
                return True
    return False


@pytest.fixture(autouse=True)
def fake_db_auto_inject(fake_db, monkeypatch):
    """把函数体内直接调用 get_db() 的 services 模块指向当前测试的 FakeDB。"""
    for name, mod in list(sys.modules.items()):
        if not name.startswith("services."):
            continue
        if not hasattr(mod, "get_db"):
            continue
        if _calls_get_db_in_body(mod):
            monkeypatch.setattr(mod, "get_db", lambda: fake_db, raising=False)


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    """默认屏蔽真实外部 LLM/大模型调用，保证 e2e 测试不触网。

    覆盖：
    - services.evaluation_engine._call_judge（LLM Judge，评估回落 L1 规则）
    - services.evaluator._call_volcano（火山方舟大模型）
    - services.translation_eval._call_translation_llm（翻译评估 v2 LLM 调用，
      返回 None → 任务 failed + EVAL_UNAVAILABLE，不触真实火山）
    """
    monkeypatch.setattr(
        "services.evaluation_engine._call_judge", lambda *a, **k: None
    )
    monkeypatch.setattr("services.evaluator._call_volcano", lambda *a, **k: None)
    monkeypatch.setattr(
        "services.translation_eval._call_translation_llm", lambda *a, **k: None
    )
