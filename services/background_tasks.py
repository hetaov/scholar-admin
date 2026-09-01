"""后台定时巡检任务（由 main.py lifespan 启动/取消）。

背景：翻译评估 / 对话匹配的提交接口原先内联 1/50 概率巡检
（recover_stale_tasks + cleanup_expired），命中时会拖慢提交响应
（两次全集合条件 update/delete，无索引时全表扫描数百 ms+）。
改为独立后台循环固定间隔执行，提交接口只保留查询热路径的定点自愈
（recover_task_if_stale）。

- start_translation_cleanup_loop : 启动 translation_task 巡检循环（每 60s 恢复卡死 + 清理过期）
- start_dialogue_cleanup_loop    : 启动 dialogue_task 巡检循环（同上）
- _loops 模块级强引用（按名字）  : 防止 asyncio 任务被 GC 回收（同 eval.py 后台任务模式）
- 单轮失败仅记日志               : 巡检是尽力而为，不影响后续轮与正常请求
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from services.dependencies import get_db
from services.learning import dialogue_task, translation_task

logger = logging.getLogger("scholar-admin.background")

# 巡检间隔：60s < 卡死判定默认 120s，保证卡死任务最多两个周期内被恢复；
# 也远小于任务 TTL（24h），过期清理足够及时。
CLEANUP_INTERVAL_SECONDS = 60

# 强引用集合（按循环名）：防止 asyncio 任务被 GC 回收导致循环中途取消
_loops: dict[str, asyncio.Task] = {}


async def _run_translation_cleanup_round() -> None:
    """执行一轮 translation_task 全集合巡检：恢复卡死 + 清理过期。"""
    db = get_db()
    await translation_task.recover_stale_tasks(db)
    await translation_task.cleanup_expired(db)


async def _run_dialogue_cleanup_round() -> None:
    """执行一轮 dialogue_task 全集合巡检：恢复卡死 + 清理过期。"""
    db = get_db()
    await dialogue_task.recover_stale_tasks(db)
    await dialogue_task.cleanup_expired(db)


async def _cleanup_loop(
    name: str,
    round_fn: Callable[[], Awaitable[None]],
    interval: float,
) -> None:
    while True:
        try:
            await round_fn()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — 单轮失败不影响后续轮与正常请求
            logger.warning("[bg] %s 巡检异常（忽略）", name, exc_info=True)
        await asyncio.sleep(interval)


def _start_loop(
    name: str,
    round_fn: Callable[[], Awaitable[None]],
    interval: float,
) -> asyncio.Task:
    """启动/复用指定巡检循环（幂等：同名循环运行中则直接返回）。"""
    existing = _loops.get(name)
    if existing is not None and not existing.done():
        return existing
    task = asyncio.create_task(
        _cleanup_loop(name, round_fn, interval), name=f"{name}-cleanup"
    )
    _loops[name] = task
    task.add_done_callback(lambda t: _loops.pop(name, None))
    logger.info("[bg] %s 巡检循环已启动（间隔 %ss）", name, interval)
    return task


def start_translation_cleanup_loop(
    interval: float = CLEANUP_INTERVAL_SECONDS,
) -> asyncio.Task:
    """启动 translation_task 巡检循环（幂等）。"""
    return _start_loop("translation", _run_translation_cleanup_round, interval)


def start_dialogue_cleanup_loop(
    interval: float = CLEANUP_INTERVAL_SECONDS,
) -> asyncio.Task:
    """启动 dialogue_task 巡检循环（幂等）。"""
    return _start_loop("dialogue", _run_dialogue_cleanup_round, interval)


async def stop_all_loops() -> None:
    """取消全部巡检循环（lifespan 关闭时调用）。"""
    tasks = [t for t in _loops.values() if not t.done()]
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _loops.clear()
    logger.info("[bg] 巡检循环已全部停止")
