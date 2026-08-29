"""翻译评估异步任务模型 — `translation_task` 集合 CRUD 与状态流转

状态机（单向，禁止回退）：
    pending ──(claim_task 原子抢占)──> processing ──┬──> success (+result)
                                                    └──> failed  (+error)

- create_translation_task : 生成 `task_id`（`tr_` 前缀），插入 pending 任务，返回任务文档
- claim_task              : 原子抢占 pending → processing（multi=False + modified_count>0 判成功）
- finish_task             : processing → success(+result) | failed(+error)
- get_task                : 按 task_id 查询（不过滤 TTL，接口层自行过滤）
- cleanup_expired         : 删除 expires_at <= now 的任务（TTL 清理，提交接口概率巡检用）
- recover_stale_tasks     : 巡检卡死 processing 任务（全集合，提交接口概率触发 1/50）
- recover_task_if_stale   : 定点恢复单条卡死任务（查询热路径，避免全表扫描）
- run_translation_task    : 后台执行器：claim → ASR（语音路径）→ LLM 评分（wait_for 超时）
                           → 终态证据落库 evaluation（成功/失败均写）→ finish_task

与 `dialogue_task`（services/learning/dialogue_task.py）的差异：
- task_id 前缀 `tr_`（data-model-contract §4.16）；
- `error` 为对象：{ error_code, error_detail, failure_stage, llm_timeout_seconds, raw }
  （ADR-0022 决策 C / §4.16），失败全量留痕；
- `audio_base64` 不落库（CloudBase 单文档 1MB 上限，5MB 音频 base64 超限），
  仅由提交接口透传后台执行器，任务文档以 None 占位（§4.16 允许只存引用）；
- 终态（success/failed）无条件写 `evaluation` 证据（docs_v1 §5.2 / §4.11.2 翻译扩展），
  不因任何原因跳过落库（进程崩溃前尽力写盘）。
"""
from __future__ import annotations

import base64
import binascii
import logging
import time
import uuid
from typing import Any

from config import TRANSLATION_LLM_TIMEOUT_SECONDS
from services.asr import get_asr_service
from services.dependencies import get_db
from services.providers.evaluation_engine import EVALUATION_COLLECTION
from services.providers.translation_eval import (
    ERR_EVAL_UNAVAILABLE,
    ERR_LLM_TIMEOUT,
    STAGE_LLM,
    STAGE_PARSE,
    TranslationEvalError,
    evaluate_translation_v2,
)
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("scholar-admin.translation_task")

COLLECTION = "translation_task"

# 任务默认保留时长：24h（与 dialogue_task 一致，保证客户端轮询窗口 + 容错重试绰绰有余）
TASK_TTL_MS = 24 * 60 * 60 * 1000

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_task_id() -> str:
    """生成业务任务 ID：`tr_` + 32 位 uuid hex。"""
    return "tr_" + uuid.uuid4().hex


async def create_translation_task(
    db,
    *,
    original_text: str,
    input_mode: str,
    mode: str,
    scholar_id: str | None = None,
    sentence_id: str | None = None,
    user_input: str | None = None,
    audio_base64: str | None = None,
    voice_format: str = "mp3",
) -> dict:
    """创建 pending 任务并落库，返回任务文档。

    不做任何 LLM/ASR 调用，保证调用方（提交接口）耗时毫秒级（ADR-0022 决策 A）。

    字段对齐 data-model-contract §4.16；`audio_base64` 不落库
    （CloudBase 单文档 1MB 上限，5MB 音频 base64 超限），以 None 占位。
    """
    now = _now_ms()
    task_doc: dict[str, Any] = {
        "task_id": build_task_id(),
        "scholar_id": scholar_id,
        "sentence_id": sentence_id,
        "original_text": original_text,
        "user_input": user_input,
        "audio_base64": None,  # P0 只存引用（§4.16），音频本体不落库
        "voice_format": voice_format,
        "input_mode": input_mode,
        "mode": mode,
        "status": STATUS_PENDING,
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    await db.insert(COLLECTION, task_doc)
    logger.info(
        f"[translation] create → task_id={task_doc['task_id']}, "
        f"mode={mode}, input_mode={input_mode}"
        + (f", scholar={scholar_id}" if scholar_id else "")
        + (f", sentence_id={sentence_id}" if sentence_id else "")
    )
    return task_doc


async def claim_task(db, task_id: str) -> bool:
    """原子抢占 pending → processing。

    并发安全：where 限定 status=pending，multi=False 只更新一条；
    `modified_count>0` 说明本实例抢占成功，否则已被其他实例抢占/状态已变。
    """
    res = await db.update(
        COLLECTION,
        where={"task_id": task_id, "status": STATUS_PENDING},
        data={"$set": {"status": STATUS_PROCESSING, "updated_at": _now_ms()}},
        multi=False,
    )
    return res.get("modified_count", 0) > 0


def _build_stale_error() -> dict:
    """卡死任务恢复的 error 对象（api-contract §3.4：卡死自愈 → LLM_TIMEOUT）。"""
    return {
        "error_code": ERR_LLM_TIMEOUT,
        "error_detail": "执行超时",
        "failure_stage": STAGE_LLM,
        "llm_timeout_seconds": TRANSLATION_LLM_TIMEOUT_SECONDS,
        "raw": None,
    }


async def finish_task(
    db,
    task_id: str,
    *,
    result: dict | list | None = None,
    error: dict | None = None,
) -> None:
    """写回执行结果：error 非空 → failed(+error, result 置 null)，否则 success(+result)。

    error 为对象 { error_code, error_detail, failure_stage, llm_timeout_seconds, raw }。
    """
    if error is not None:
        status = STATUS_FAILED
        result_value = None
        error_value = error
        logger.info(
            f"[translation] fail → task_id={task_id}, error={error.get('error_code')}"
        )
    else:
        status = STATUS_SUCCESS
        result_value = result
        error_value = None
        logger.info(f"[translation] done → task_id={task_id}, status=success")
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
        logger.info(f"[translation] cleanup → 删除过期任务 {count} 条")
    return count


async def recover_stale_tasks(db, timeout_s: int = 120) -> int:
    """巡检卡死的 processing 任务：updated_at 超过 timeout_s 未更新 → 置为 failed。

    兜底容器回收 / 进程崩溃导致的 processing 卡死（任务记录是唯一状态源，
    实例挂了没有 else 分支写 failed，必须靠巡检恢复）。

    Returns:
        修复（置为 failed）的任务数量
    """
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
        logger.info(f"[translation] recover → 卡死任务标记 failed {count} 条")
    return count


async def recover_task_if_stale(db, task: dict, timeout_s: int = 120) -> bool:
    """定点恢复：单条卡死 processing 任务（updated_at 超时）→ 置 failed。

    与 `recover_stale_tasks` 的区别：只针对传入任务的 task_id 做单点更新，
    避免查询热路径触发全集合条件更新（无索引时全表扫描会拖慢轮询）。

    Args:
        task: get_task 返回的任务文档

    Returns:
        是否恢复成功（该任务被置为 failed）
    """
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
            f"[translation] recover → task_id={task['task_id']} 卡死任务标记 failed"
        )
        return True
    return False


async def _write_evaluation(
    db,
    *,
    task_id: str,
    scholar_id: str | None,
    sentence_id: str | None,
    original_text: str,
    input_mode: str,
    mode: str,
    user_input: str | None,
    result: dict | None,
    error: dict | None,
) -> None:
    """终态双写：任务进入终态时无条件写 `evaluation` 证据（docs_v1 §5.2 / §4.11.2）。

    - 成功记录：succeeded=true + status/feedback/confidence/raw；
    - 失败记录：succeeded=false + error_code/error_detail/failure_stage/
      llm_timeout_seconds/raw。
    **失败即记录**：落库失败仅记日志，不阻断任务收尾（进程崩溃前尽力写盘）。
    """
    try:
        base = {
            "scholar_id": scholar_id,
            "sentence_id": sentence_id,
            "task_id": task_id,
            "mode": mode,
            "input_mode": input_mode,
            "original_text": original_text,
            "user_input": user_input,
            "provider": "volcano",
            "type": "translation",
            "created_at": int(time.time() * 1000),
        }
        if error is not None:
            doc = {
                **base,
                "succeeded": False,
                "status": None,
                "feedback": None,
                "confidence": None,
                "error_code": error.get("error_code"),
                "error_detail": error.get("error_detail"),
                "failure_stage": error.get("failure_stage"),
                "llm_timeout_seconds": error.get("llm_timeout_seconds"),
                "raw": error.get("raw"),
            }
        else:
            doc = {
                **base,
                "succeeded": True,
                "status": (result or {}).get("status"),
                "feedback": (result or {}).get("feedback"),
                "confidence": (result or {}).get("confidence"),
                "raw": (result or {}).get("raw_model_output"),
            }
        await db.insert(EVALUATION_COLLECTION, doc)
    except Exception as e:  # noqa: BLE001 — 留痕失败不影响任务收尾
        logger.error(f"[translation] evaluation 留痕失败 task_id={task_id}: {e}")


async def run_translation_task(
    task_id: str,
    *,
    original_text: str,
    mode: str,
    input_mode: str,
    user_input: str | None = None,
    audio_base64: str | None = None,
    voice_format: str = "mp3",
    scholar_id: str | None = None,
    sentence_id: str | None = None,
) -> None:
    """后台执行翻译评估任务并写回结果（ADR-0022 决策 A/B/C/D）。

    由提交接口 `asyncio.create_task(...)` 调度，与请求解耦：
    - claim_task 原子抢占：被其他实例抢占则直接返回，避免重复执行
    - 语音路径：ASR 转写（失败 → failed + ASR_UNAVAILABLE，stage=asr）
    - 评分：LLM 调用 `asyncio.wait_for` 包裹（超时默认 300s）→ failed + LLM_TIMEOUT
    - **不降级**：LLM 不可用 / 输出非法 → failed（不回退 levenshtein / 混元 / 云函数）
    - 终态无条件写 evaluation 证据（成功/失败均写），随后 finish_task
    """
    db = get_db()
    if not await claim_task(db, task_id):
        logger.info(f"[translation] run skip → task_id={task_id} 已被抢占或状态非 pending")
        return
    result: dict | None = None
    error: dict | None = None
    transcription = user_input or ""
    try:
        if input_mode == "voice":
            try:
                audio_bytes = base64.b64decode(audio_base64 or "", validate=True)
            except (binascii.Error, ValueError):
                raise TranslationEvalError(
                    "ASR_UNAVAILABLE", "asr", "audio_base64 不是合法 base64"
                )
            if not audio_bytes:
                raise TranslationEvalError("ASR_UNAVAILABLE", "asr", "音频内容为空")
            asr = get_asr_service()
            if not asr.available:
                raise TranslationEvalError(
                    "ASR_UNAVAILABLE", "asr", "ASR 服务未配置凭据"
                )
            # recognize 为同步阻塞调用，丢线程池避免卡事件循环
            transcription = await run_in_threadpool(
                asr.recognize, audio_bytes, voice_format
            )
            if not transcription:
                raise TranslationEvalError("ASR_UNAVAILABLE", "asr", "语音识别失败")
        if not transcription.strip():
            raise TranslationEvalError("EVAL_UNAVAILABLE", "parse", "用户输入为空")
        eval_result = await evaluate_translation_v2(mode, original_text, transcription)
        result = {
            "transcription": transcription,
            "status": eval_result["status"],
            "feedback": eval_result["feedback"],
            "confidence": eval_result["confidence"],
            "raw_model_output": eval_result.get("raw"),
        }
    except TranslationEvalError as e:
        error = e.to_dict(llm_timeout_seconds=TRANSLATION_LLM_TIMEOUT_SECONDS)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[translation] run error → task_id={task_id}: {e}", exc_info=True)
        error = {
            "error_code": "NETWORK_ERROR",
            "error_detail": str(e)[:500],
            "failure_stage": STAGE_LLM,
            "llm_timeout_seconds": TRANSLATION_LLM_TIMEOUT_SECONDS,
            "raw": None,
        }
    await finish_task(db, task_id, result=result, error=error)
    await _write_evaluation(
        db,
        task_id=task_id,
        scholar_id=scholar_id,
        sentence_id=sentence_id,
        original_text=original_text,
        input_mode=input_mode,
        mode=mode,
        user_input=transcription or user_input,
        result=result,
        error=error,
    )
