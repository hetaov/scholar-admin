"""单元测试：后台定时巡检 services/background_tasks（卡死恢复 + TTL 清理移出提交热路径）

覆盖：
- 一轮巡检：超时 processing → failed（LLM_TIMEOUT），过期任务删除，未超时不受影响
- 幂等启动：已有运行中的循环则不重复创建
- stop_all_loops：取消全部循环，无运行循环时安全
- dialogue_task 循环：与 translation 同构，各自独立启动/巡检
"""
from __future__ import annotations

import asyncio
import time

import services.background_tasks as bg
from services.dialogue_task import (
    STATUS_FAILED as DT_STATUS_FAILED,
    STATUS_PROCESSING as DT_STATUS_PROCESSING,
)
from services.translation_task import STATUS_FAILED, STATUS_PROCESSING
from tests.fakes.fake_db import FakeDB
from tests.fakes.seed_factory import seed_task, seed_translation_task


def test_cleanup_round_recovers_stale_and_deletes_expired(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(bg, "get_db", lambda: db)
    now = int(time.time() * 1000)
    seed_translation_task(
        db, task_id="tr_stale", status="processing", updated_at=now - 130_000
    )
    seed_translation_task(db, task_id="tr_fresh", status="processing", updated_at=now)
    seed_translation_task(db, task_id="tr_expired", expires_at=now - 1000)

    async def scenario():
        task = bg.start_translation_cleanup_loop(interval=0.01)
        # 幂等：再次启动返回同一运行中的循环任务
        assert bg.start_translation_cleanup_loop(interval=0.01) is task
        await asyncio.sleep(0.05)  # 让循环至少完整执行一轮
        await bg.stop_all_loops()

    asyncio.run(scenario())

    stored = {d["task_id"]: d for d in db.all("translation_task")}
    assert stored["tr_stale"]["status"] == STATUS_FAILED
    assert stored["tr_stale"]["error"]["error_code"] == "LLM_TIMEOUT"
    assert stored["tr_fresh"]["status"] == STATUS_PROCESSING  # 未超时不受影响
    assert "tr_expired" not in stored  # 过期任务已清理


def test_dialogue_loop_recovers_stale_and_deletes_expired(monkeypatch):
    """dialogue_task 循环与 translation 独立：各自巡检自己的集合。"""
    db = FakeDB()
    monkeypatch.setattr(bg, "get_db", lambda: db)
    now = int(time.time() * 1000)
    seed_task(db, task_id="dt_stale", status="processing", updated_at=now - 130_000)
    seed_task(db, task_id="dt_fresh", status="processing", updated_at=now)
    seed_task(db, task_id="dt_expired", expires_at=now - 1000)

    async def scenario():
        task = bg.start_dialogue_cleanup_loop(interval=0.01)
        # 幂等：再次启动返回同一运行中的循环任务
        assert bg.start_dialogue_cleanup_loop(interval=0.01) is task
        await asyncio.sleep(0.05)
        await bg.stop_all_loops()

    asyncio.run(scenario())

    stored = {d["task_id"]: d for d in db.all("dialogue_task")}
    assert stored["dt_stale"]["status"] == DT_STATUS_FAILED
    assert stored["dt_stale"]["error"] == "执行超时"
    assert stored["dt_fresh"]["status"] == DT_STATUS_PROCESSING
    assert "dt_expired" not in stored


def test_stop_all_loops_without_running_loops_is_safe():
    async def scenario():
        await bg.stop_all_loops()  # 无运行中循环也不应抛异常

    asyncio.run(scenario())


def test_loop_survives_transient_failure(monkeypatch):
    """单轮巡检抛异常不终止循环，下一轮恢复正常。"""
    db = FakeDB()
    monkeypatch.setattr(bg, "get_db", lambda: db)
    now = int(time.time() * 1000)
    seed_translation_task(
        db, task_id="tr_stale", status="processing", updated_at=now - 130_000
    )

    calls = {"n": 0}

    original_recover = bg.translation_task.recover_stale_tasks

    async def flaky_recover(db, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("临时故障")
        return await original_recover(db, **kwargs)

    monkeypatch.setattr(bg.translation_task, "recover_stale_tasks", flaky_recover)

    async def scenario():
        bg.start_translation_cleanup_loop(interval=0.01)
        await asyncio.sleep(0.05)  # 第一轮失败，后续轮成功
        await bg.stop_all_loops()

    asyncio.run(scenario())

    assert calls["n"] >= 2
    stored = {d["task_id"]: d for d in db.all("translation_task")}
    assert stored["tr_stale"]["status"] == STATUS_FAILED
