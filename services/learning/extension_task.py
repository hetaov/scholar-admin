"""英文语句扩展异步任务模型 — `extension_task` 集合 CRUD 与状态流转

契约：data-model-contract.md §4.25；api-contract.md §3.18（E1~E4）。
设计：docs_v1/扩展/第一期-scholar-admin接口与admin-web实验页-v1.md §3.2.2。

与 translation_task.py 同构，差异：
- COLLECTION = extension_task（可经 EXTENSION_TASK_COLLECTION 覆盖）；
- task_id 前缀 `ex_`；
- 新增 `kind` 字段：extract | evaluate（一个集合承载两类任务）；
- evaluate 任务额外有 task_type（l1_fill | l2_sentence）、selected_ids、points_snapshot；
- error 沿用 5 字段 { error_code, error_detail, failure_stage, llm_timeout_seconds, raw }。

状态机（单向，禁止回退）：
    pending ──(claim_task 原子抢占)──> processing ──┬──> success (+result)
                                                    └──> failed  (+error)
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import config
from services.dependencies import get_db

logger = logging.getLogger("scholar-admin.extension_task")

COLLECTION = config.EXTENSION_TASK_COLLECTION

# 任务默认保留时长：24h（与 translation_task 一致）
TASK_TTL_MS = 24 * 60 * 60 * 1000

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"

KIND_EXTRACT = "extract"
KIND_EVALUATE = "evaluate"

# 错误码（与 api-contract §3.18 业务码表对齐）
ERR_LLM_TIMEOUT = "LLM_TIMEOUT"
ERR_LLM_PARSE_ERROR = "LLM_PARSE_ERROR"
ERR_PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"

STAGE_LLM = "llm"
STAGE_RULE_FALLBACK = "rule_fallback"


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_task_id() -> str:
    """生成业务任务 ID：`ex_` + 32 位 uuid hex。"""
    return "ex_" + uuid.uuid4().hex


async def create_extension_task(
    db,
    *,
    kind: str,
    scholar_id: str | None = None,
    sentence_id: str | None = None,
    textbook_id: str | None = None,
    lesson_id: str | None = None,
    task_type: str | None = None,
    selected_ids: list[str] | None = None,
    points_snapshot: list[dict] | None = None,
    input_mode: str = "text",
    user_input: str | None = None,
    audio_base64: str | None = None,
    voice_format: str = "mp3",
    asr_engine: str = "16k_en",
    enable_fallback: bool = True,
) -> dict:
    """创建 pending 任务并落库，返回任务文档。

    kind: extract | evaluate。
    evaluate 任务需传 task_type / selected_ids / points_snapshot。
    audio_base64 不落库（CloudBase 单文档 1MB 上限），以 None 占位。
    """
    now = _now_ms()
    task_doc: dict[str, Any] = {
        "task_id": build_task_id(),
        "kind": kind,
        "scholar_id": scholar_id,
        "sentence_id": sentence_id,
        "textbook_id": textbook_id,
        "lesson_id": lesson_id,
        "task_type": task_type,
        "selected_ids": selected_ids or [],
        "points_snapshot": points_snapshot or [],
        "input_mode": input_mode,
        "user_input": user_input,
        "audio_base64": None,  # 不落库，仅透传执行器
        "voice_format": voice_format,
        "asr_engine": asr_engine,
        # 请求级「规则兜底」开关（实验页对照实验用）；与全局开关为「与」关系
        "enable_fallback": bool(enable_fallback),
        "status": STATUS_PENDING,
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    await db.insert(COLLECTION, task_doc)
    logger.info(
        f"[extension] create → task_id={task_doc['task_id']}, kind={kind}"
        + (f", task_type={task_type}" if task_type else "")
        + (f", sentence_id={sentence_id}" if sentence_id else "")
    )
    return task_doc


async def claim_task(db, task_id: str) -> bool:
    """原子抢占 pending → processing。"""
    res = await db.update(
        COLLECTION,
        where={"task_id": task_id, "status": STATUS_PENDING},
        data={"$set": {"status": STATUS_PROCESSING, "updated_at": _now_ms()}},
        multi=False,
    )
    return res.get("modified_count", 0) > 0


def _build_stale_error() -> dict:
    """卡死任务恢复的 error 对象（LLM_TIMEOUT）。"""
    return {
        "error_code": ERR_LLM_TIMEOUT,
        "error_detail": "执行超时",
        "failure_stage": STAGE_LLM,
        "llm_timeout_seconds": config.EXTENSION_LLM_TIMEOUT_SECONDS,
        "raw": None,
    }


async def finish_task(
    db,
    task_id: str,
    *,
    result: dict | list | None = None,
    error: dict | None = None,
) -> None:
    """写回执行结果：error 非空 → failed(+error, result 置 null)，否则 success(+result)。"""
    if error is not None:
        status = STATUS_FAILED
        result_value = None
        error_value = error
        logger.info(
            f"[extension] fail → task_id={task_id}, "
            f"error={error.get('error_code')}"
        )
    else:
        status = STATUS_SUCCESS
        result_value = result
        error_value = None
        logger.info(f"[extension] done → task_id={task_id}, status=success")
    await db.update(
        COLLECTION,
        where={"task_id": task_id},
        data={
            "$set": {
                "status": status,
                "result": result_value,
                "error": error_value,
                "updated_at": _now_ms(),
            }
        },
        multi=False,
    )


async def get_task(db, task_id: str) -> dict | None:
    """按 task_id 查询任务，未命中返回 None（不过滤 TTL）。"""
    res = await db.query(COLLECTION, where={"task_id": task_id}, limit=1)
    records = res.get("records", [])
    return records[0] if records else None


async def cleanup_expired(db, now_ms: int | None = None) -> int:
    """删除 expires_at <= now 的过期任务，返回删除数量。"""
    now = now_ms if now_ms is not None else _now_ms()
    res = await db.delete(COLLECTION, where={"expires_at": {"$lte": now}})
    count = res.get("deleted_count", 0)
    if count:
        logger.info(f"[extension] cleanup → 删除过期任务 {count} 条")
    return count


async def recover_stale_tasks(db, timeout_s: int = 120) -> int:
    """巡检卡死的 processing 任务：updated_at 超过 timeout_s → 置 failed。"""
    now = _now_ms()
    threshold = now - timeout_s * 1000
    res = await db.update(
        COLLECTION,
        where={
            "status": STATUS_PROCESSING,
            "updated_at": {"$lt": threshold},
        },
        data={
            "$set": {
                "status": STATUS_FAILED,
                "error": _build_stale_error(),
                "updated_at": now,
            }
        },
        multi=True,
    )
    count = res.get("modified_count", 0)
    if count:
        logger.info(f"[extension] recover → 卡死任务标记 failed {count} 条")
    return count


async def recover_task_if_stale(db, task: dict, timeout_s: int = 120) -> bool:
    """定点恢复：单条卡死 processing 任务（updated_at 超时）→ 置 failed。"""
    if task.get("status") != STATUS_PROCESSING:
        return False
    now = _now_ms()
    if task.get("updated_at", 0) > now - timeout_s * 1000:
        return False
    res = await db.update(
        COLLECTION,
        where={"task_id": task["task_id"], "status": STATUS_PROCESSING},
        data={
            "$set": {
                "status": STATUS_FAILED,
                "error": _build_stale_error(),
                "updated_at": now,
            }
        },
        multi=False,
    )
    if res.get("modified_count", 0):
        logger.info(
            f"[extension] recover → task_id={task['task_id']} 卡死任务标记 failed"
        )
        return True
    return False


async def run_extension_task(
    task_id: str,
    *,
    kind: str,
    sentence_id: str | None = None,
    textbook_id: str | None = None,
    lesson_id: str | None = None,
    task_type: str | None = None,
    selected_ids: list[str] | None = None,
    points_snapshot: list[dict] | None = None,
    input_mode: str = "text",
    user_input: str | None = None,
    audio_base64: str | None = None,
    voice_format: str = "mp3",
    scholar_id: str | None = None,
    original: str = "",
    translation: str = "",
    asr_engine: str = "16k_en",
    enable_fallback: bool = True,
) -> None:
    """后台执行扩展任务并写回结果。

    B04 骨架：实现 claim → 执行 → finish 结构。
    实际抽取/评测逻辑（B06 provider / B07 extension 核心）通过懒加载导入，
    在 B09/E1、B14/E2 接线时生效。
    """
    db = get_db()
    if not await claim_task(db, task_id):
        logger.info(
            f"[extension] run skip → task_id={task_id} 已被抢占或状态非 pending"
        )
        return

    result: dict | None = None
    error: dict | None = None
    try:
        if kind == KIND_EXTRACT:
            # B07 抽取核心（懒加载，避免 B04 单独导入时缺依赖）
            # 开关 EXTENSION_USE_LANGGRAPH：开 → 连续+非连续图并行；关 → 一期连续抽取。
            if config.EXTENSION_USE_LANGGRAPH:
                from services.english.extension import run_extract_pipeline_via_graph

                result = await run_extract_pipeline_via_graph(
                    db,
                    sentence_id=sentence_id,
                    textbook_id=textbook_id,
                    lesson_id=lesson_id,
                    original=original,
                    translation=translation,
                    scholar_id=scholar_id,
                    enable_fallback=enable_fallback,
                )
            else:
                from services.english.extension import run_extract_pipeline

                result = await run_extract_pipeline(
                    db,
                    sentence_id=sentence_id,
                    textbook_id=textbook_id,
                    lesson_id=lesson_id,
                    original=original,
                    translation=translation,
                    scholar_id=scholar_id,
                    enable_fallback=enable_fallback,
                )
        elif kind == KIND_EVALUATE:
            # B12/B13 评测核心
            from services.english.extension import run_evaluate_pipeline

            result = await run_evaluate_pipeline(
                db,
                task_type=task_type,
                selected_ids=selected_ids or [],
                points_snapshot=points_snapshot or [],
                input_mode=input_mode,
                user_input=user_input,
                audio_base64=audio_base64,
                voice_format=voice_format,
                scholar_id=scholar_id,
            )
        else:
            raise ValueError(f"unknown kind: {kind}")
    except NotImplementedError:
        # B07/B12 尚未实现时，占位 failed（开发期）
        error = {
            "error_code": ERR_PROVIDER_UNAVAILABLE,
            "error_detail": "extension pipeline not implemented yet",
            "failure_stage": STAGE_LLM,
            "llm_timeout_seconds": config.EXTENSION_LLM_TIMEOUT_SECONDS,
            "raw": None,
        }
    except Exception as e:  # noqa: BLE001
        logger.error(f"[extension] run error → task_id={task_id}: {e}", exc_info=True)
        error = {
            "error_code": ERR_PROVIDER_UNAVAILABLE,
            "error_detail": str(e)[:500],
            "failure_stage": STAGE_LLM,
            "llm_timeout_seconds": config.EXTENSION_LLM_TIMEOUT_SECONDS,
            "raw": None,
        }

    await finish_task(db, task_id, result=result, error=error)
