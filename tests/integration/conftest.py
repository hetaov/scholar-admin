"""集成测试专用 conftest:T1.3 FakeDB 全局统一注入。

自动把"函数体内直接调用 get_db()"的 services 模块替换为指向当前测试
fake_db 的 lambda,覆盖方式 A 场景(如 routes_tracking / routes_state /
routes_dialogue / dialogue_task / routes_build),集成测试因此无需再手写
monkeypatch.setattr("services.X.get_db", ...)。

为什么用 AST 判断而非全局替换:
- 所有模块的 get_db 都来自 services.dependencies(同一函数对象);
- 方式 B 模块(routes_evaluation / routes_training / routes_tts 等)用
  Depends(get_db) 在定义期捕获原对象,setattr 无效;若全局替换会污染其
  命名空间,导致未迁移测试从模块重新 import 时拿到 lambda,从而 dependency
  overrides 的 key 失配。
- 仅替换函数体内存在 `get_db(...)` 直接调用的模块,精确且免维护。

方式 B 场景仍由 tests/conftest.py 的 make_client 通过 dependency_overrides
统一注入(overrides key 取 services.dependencies.get_db 原对象)。

注意:仅作用于本目录,unit 测试不受影响。
"""

from __future__ import annotations

import ast
import inspect
import sys

import pytest


def _calls_get_db_in_body(mod) -> bool:
    """模块源码中是否存在 `get_db(...)` 直接调用(非 Depends(get_db) 参数形式)。"""
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
    """默认屏蔽真实外部 LLM/大模型调用,保证集成测试不触网。

    覆盖:
    - services.evaluation_engine._call_judge(LLM Judge,评估回落 L1 规则)
    - services.evaluator._call_volcano(火山方舟大模型)

    测试需模拟模型返回时,在自身通过 monkeypatch.setattr 覆盖即可
    (测试体内执行晚于 fixture,优先级更高)。TTS / SOE-N / ASR 的真实
    Provider 由各测试显式注入 fake_providers 替身,不在此处处理。
    """
    monkeypatch.setattr(
        "services.evaluation_engine._call_judge", lambda *a, **k: None
    )
    monkeypatch.setattr("services.evaluator._call_volcano", lambda *a, **k: None)
