"""对话匹配接口"""

from __future__ import annotations

import asyncio
import logging
import random
import time

from fastapi import APIRouter, HTTPException

from services.dependencies import get_db
from services.dialogue import load_learned_sentences, match_dialogue
from services.dialogue_task import (
    STATUS_FAILED,
    cleanup_expired,
    create_task,
    get_task,
    recover_stale_tasks,
    recover_task_if_stale,
    run_dialogue_task,
)

logger = logging.getLogger("scholar-admin.routes.dialogue")
router = APIRouter(tags=["对话匹配"])

# 后台任务强引用集合：防止 create_task 的协程被 GC 回收导致任务中途取消
_background_tasks: set[asyncio.Task] = set()
# 全集合 TTL 巡检概率（每次提交触发一次 1/50），避免查询热路径全表扫描
_CLEANUP_PROB = 0.02


@router.post("/match/dialogue")
async def match_dialogue_endpoint(data: dict):
    """对话匹配 — 根据输入英文句，从已学语句中匹配或生成问答对

    请求体：
    {
      \"scholarId\": \"6d758f346a6daee000859c332ed11089\",
      \"sentence\": \"I go to school by bus every day.\"
    }

    返回：
    {
      \"success\": true,
      \"data\": {
        \"type\": \"qa\",
        \"statement\": \"...\",
        \"question\": \"...\",
        \"source\": \"matched|generated\"
      },
      \"is_question\": false
    }
    """
    scholar_id = data.get("scholarId", "")
    input_sentence = data.get("sentence", "")

    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少参数 scholarId")
    if not input_sentence:
        raise HTTPException(status_code=400, detail="缺少参数 sentence")

    try:
        db = get_db()

        # 1. 加载该学者全部已学语句
        learned_sentences = await load_learned_sentences(db, scholar_id)
        if not learned_sentences:
            return {"success": False, "error": "该学者暂无已学语句", "data": None}

        logger.info(f'[match] scholar={scholar_id}, 输入="{input_sentence}"')

        # 2. 执行 LangGraph 工作流
        result = await match_dialogue(
            input_sentence=input_sentence,
            scholar_id=scholar_id,
            learned_sentences=learned_sentences,
        )
        return result

    except Exception as e:
        logger.error(
            f"[match] 对话匹配异常: scholar_id={scholar_id}, input={input_sentence!r}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"对话匹配失败: {str(e)}")


@router.post("/match/dialogue/task")
async def create_match_dialogue_task(data: dict):
    """提交异步对话匹配任务 — 毫秒级返回 taskId，匹配在后台执行

    请求体（与同步接口一致，P2/F10 新增可选 scenario/sessionId）：
    {
      "scholarId": "6d758f346a6daee000859c332ed11089",
      "sentence": "I go to school by bus every day.",
      "scenario": "daily",        // 可选：daily/travel/ordering/interview/free，缺省 free
      "sessionId": "ses_xxx"      // 可选：多轮会话标识，仅透传
    }

    返回：
    {
      "success": true,
      "data": { "taskId": "dt_xxx", "status": "pending" }
    }
    """
    scholar_id = data.get("scholarId", "")
    input_sentence = data.get("sentence", "")
    scenario = data.get("scenario")
    session_id = data.get("sessionId")

    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少参数 scholarId")
    if not input_sentence:
        raise HTTPException(status_code=400, detail="缺少参数 sentence")

    try:
        db = get_db()
        task = await create_task(
            db,
            scholar_id=scholar_id,
            sentence=input_sentence,
            scenario=scenario,
            session_id=session_id,
        )
        # 后台执行，不阻塞当前请求（毫秒级返回）。
        # 必须持有 task 引用并注册 done_callback，否则协程可能被 GC 回收而取消。
        bg = asyncio.create_task(
            run_dialogue_task(
                task["task_id"],
                scholar_id,
                input_sentence,
                scenario=scenario,
                session_id=session_id,
            )
        )
        _background_tasks.add(bg)
        bg.add_done_callback(_background_tasks.discard)

        # 概率性全集合 TTL 巡检（1/50）：恢复卡死任务 + 清理过期任务。
        # 刻意移出查询热路径——无索引时条件 update/delete 全表扫描会拖慢轮询。
        if random.random() < _CLEANUP_PROB:
            await recover_stale_tasks(db)
            await cleanup_expired(db)

        logger.info(
            f"[match] 异步任务已提交 → task_id={task['task_id']}, scholar={scholar_id}"
        )
        return {
            "success": True,
            "data": {
                "taskId": task["task_id"],
                "status": task["status"],
            },
        }
    except Exception as e:
        logger.error(
            f"[match] 任务提交异常: scholar_id={scholar_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"任务提交失败: {str(e)}")


@router.get("/match/dialogue/task/{task_id}")
async def get_match_dialogue_task(task_id: str):
    """查询异步对话匹配任务状态 — pending/processing/success/failed

    返回：
    {
      "success": true,
      "data": {
        "taskId": "dt_xxx",
        "status": "success",
        "result": { "type": "qa", "statement": "...", "question": "...", "source": "matched|generated" },
        "is_question": false,
        "error": null
      }
    }

    任务不存在或已过期返回 404。
    """
    try:
        db = get_db()
        task = await get_task(db, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        # 定点自愈：被查询任务若卡死（processing 且超时）→ 置 failed。
        # 只做单点更新，不做全集合巡检，保证轮询热路径毫秒级返回。
        if await recover_task_if_stale(db, task):
            task = {**task, "status": STATUS_FAILED, "error": "执行超时"}
        # TTL 过滤：expires_at 已过期的按不存在处理（物理清理由提交接口概率巡检执行）
        now_ms = int(time.time() * 1000)
        if task.get("expires_at", 0) <= now_ms:
            raise HTTPException(status_code=404, detail="任务已过期")
        return {
            "success": True,
            "data": {
                "taskId": task["task_id"],
                "status": task["status"],
                "result": task.get("result"),
                "is_question": task.get("is_question"),
                "error": task.get("error"),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[match] 任务查询异常: task_id={task_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"任务查询失败: {str(e)}")
