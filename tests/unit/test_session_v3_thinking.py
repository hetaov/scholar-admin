"""v3 会话路径的「思考」开关（2026-10-07 补齐）

背景：`POST /ai/session/v3` → `services/learning/session_task_v3.py::run_session_task`
→ `services/learning/dialogue_engine.generate_session_reply` → `services/providers/dialogue_gen.py`。
该 provider 的 `thinking_disabled` **默认 False**（其 docstring 记作「现行为」），而 v3 未显式透传
⇒ v3 一直是「思考开」，与已关思考的 v4 走同一 provider / 同一模型（`VOLCANO_CHAT_MODEL`，推理型）。

本文件锁定：run_session_task 确实把 `thinking_disabled` 透传给引擎，且取值跟随
`SESSION_V3_THINKING_DISABLED`（默认继承 v4 = 关）。
"""

import pytest

from services.learning import session_task_v3 as v3


class _Stub:
    def __init__(self):
        self.kwargs: dict | None = None
        self.finished: dict | None = None


def _install(monkeypatch, task) -> _Stub:
    """桩掉 db / 取任务 / 抢占 / 引擎 / 收尾，只观察传给引擎的 kwargs。"""
    stub = _Stub()

    monkeypatch.setattr(v3, "get_db", lambda: object())

    async def _get_task(_db, _task_id):
        return task

    async def _claim(_db, _task_id):
        return True

    async def _gen(**kwargs):
        stub.kwargs = kwargs
        return {"content_type": "text", "ai_text": "ok", "hint": None, "suggested_targets": []}

    async def _finish(_db, _task_id, result=None, error=None):
        stub.finished = {"result": result, "error": error}

    monkeypatch.setattr(v3, "get_task", _get_task)
    monkeypatch.setattr(v3, "claim_task", _claim)
    monkeypatch.setattr(v3.dialogue_engine, "generate_session_reply", _gen)
    monkeypatch.setattr(v3, "finish_task", _finish)
    return stub


# session_id 置空 → 跳过「回写历史 / 释放在途位」，把用例收窄到「透传参数」这一件事上
_TASK = {"session_id": None, "context": {}, "preferred_type": "auto", "mode": "start"}


@pytest.mark.asyncio
async def test_v3_passes_thinking_disabled_by_default(monkeypatch):
    """默认（SESSION_V3_THINKING_DISABLED 继承 v4 = 1）→ 引擎收到 thinking_disabled=True。"""
    stub = _install(monkeypatch, _TASK)
    monkeypatch.setattr(v3, "SESSION_V3_THINKING_DISABLED", 1)

    await v3.run_session_task("t_1")

    assert stub.kwargs is not None
    assert stub.kwargs["thinking_disabled"] is True
    # 既有入参零改动
    assert stub.kwargs["session_id"] is None
    assert stub.kwargs["context"] == {}
    assert stub.kwargs["preferred_type"] == "auto"
    assert stub.finished["error"] is None


@pytest.mark.asyncio
async def test_v3_thinking_flag_zero_reverts_to_current_behavior(monkeypatch):
    """置 0 → thinking_disabled=False（回到补齐前的「思考开」，可作对照实验）。"""
    stub = _install(monkeypatch, _TASK)
    monkeypatch.setattr(v3, "SESSION_V3_THINKING_DISABLED", 0)

    await v3.run_session_task("t_2")

    assert stub.kwargs["thinking_disabled"] is False


def test_v3_flag_default_follows_v4():
    """配置默认跟随 v4（写法沿用 LLM_JUDGE_DISABLE_THINKING 跟随 LLM_DISABLE_THINKING 的先例）。"""
    import config

    assert config.SESSION_V3_THINKING_DISABLED == config.SESSION_V4_THINKING_DISABLED
    assert config.SESSION_V3_THINKING_DISABLED == 1
