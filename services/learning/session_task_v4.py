"""沉浸式 AI 会话 v4 异步生成任务模型 — `ai_session_v4_task` 集合 CRUD 与状态流转（api-contract §3.17）

同构复制 `services/learning/session_task_v3.py`（v3）：状态机、并发口径、失败回写规则与
v3 逐条一致，**仅集合名参数化**（`ai_session_v4_task` / `ai_session_v4`），生成核同为
`services.learning.dialogue_engine`，**不写 v2 / v3 集合**。

状态机（单向，禁止回退）：
    pending ──(claim_task 原子抢占)──> processing ──┬──> success (+result)
                                                    └──> failed  (+error)

与 v3 的差异（仅 v4 追加，不改动 v3 语义）：
- 任务文档追加**可选字段**：`stream` / `hint_mode` / `model_tier`（请求透传）、
  `partial_text` / `partial_seq` / `partial_updated_at`（伪流式增量）、`timings`（最小埋点）；
- 新增 `write_partial`：节流写增量（伪流式落库，§5/§6）；
- **双开关**：请求级 `stream=true` **且** 服务级 `SESSION_V4_STREAM_ENABLED=1` 才产增量，
  否则行为与 v3 完全一致（S5 判据：「v4 ≡ v3」可二分验证）。

字段与口径见契约 §3.17；`result` 结构 `{ content_type, ai_text, hint, suggested_targets }` 与 v3 一致。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Awaitable, Callable

from config import (
    SESSION_LLM_TIMEOUT_SECONDS,
    SESSION_V4_CHECKPOINT_COLLECTION,
    SESSION_V4_PARTIAL_MAX_WRITES,
    SESSION_V4_PARTIAL_MIN_CHARS,
    SESSION_V4_PARTIAL_THROTTLE_MS,
    SESSION_V4_STATE_COLLECTION,
    SESSION_V4_STREAM_ENABLED,
    SESSION_V4_TASK_COLLECTION,
    SESSION_V4_THINKING_DISABLED,
)
from services.dependencies import get_db
from services.learning import session_state_v4 as session_state
from services.learning import dialogue_engine
from services.providers.session_gen import (
    ERR_LLM_TIMEOUT,
    ERR_NETWORK_ERROR,
    STAGE_LLM,
    SessionGenError,
)

logger = logging.getLogger("scholar-admin.session_task_v4")

COLLECTION = SESSION_V4_TASK_COLLECTION
SESSION_COLLECTION = SESSION_V4_STATE_COLLECTION

# 任务默认保留时长：24h（与 session_state_v4 同口径）
TASK_TTL_MS = 24 * 60 * 60 * 1000

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"

# 增量回调签名：`(delta_text) -> None | Awaitable[None]`
OnDelta = Callable[[str], Any]

# 内存节流态：task_id → { last_ms, length, writes }
# 单实例生效；多实例部署时退化为「每次回调都写」，仍受 MAX_WRITES 上限保护。
_partial_state: dict[str, dict[str, Any]] = {}


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_task_id() -> str:
    """生成业务任务 ID：`st_` + 32 位 uuid hex（与 v2/v3 同前缀）。"""
    return "st_" + uuid.uuid4().hex


def stream_enabled_for(request_stream: bool | None) -> bool:
    """是否产出伪流式增量：请求 stream=true **且** 服务级开关开启。"""
    return bool(request_stream) and bool(SESSION_V4_STREAM_ENABLED)


async def create_session_task(
    db,
    *,
    task_id: str | None = None,
    scholar_id: str,
    session_id: str,
    mode: str,
    preferred_type: str,
    context: dict,
    stream: bool = False,
    hint_mode: str = "sync",
    model_tier: str = "standard",
) -> dict:
    """创建 pending 任务并落库，返回任务文档（不做 LLM 调用，毫秒级返回）。

    追加字段（契约 §3.17，全为可选，缺省即 v3 行为）：
    - `stream` / `hint_mode` / `model_tier`：请求透传，供执行器决策；
    - `partial_text` / `partial_seq` / `partial_updated_at`：伪流式增量（终态清空）；
    - `timings`：最小埋点（`submitted_at` 起，终态补 `claimed_at` / `llm_total_ms` / `total_ms`）。
    """
    now = _now_ms()
    task_doc: dict[str, Any] = {
        "task_id": task_id or build_task_id(),
        "scholar_id": scholar_id,
        "session_id": session_id,
        "mode": mode,
        "preferred_type": preferred_type,
        "status": STATUS_PENDING,
        "result": None,
        "error": None,
        "context": context,
        # ---- v4 追加 ----
        "stream": bool(stream),
        "hint_mode": hint_mode,
        "model_tier": model_tier,
        "partial_text": None,
        "partial_seq": 0,
        "partial_updated_at": None,
        "timings": {"submitted_at": now},
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    await db.insert(COLLECTION, task_doc)
    logger.info(
        f"[session_v4] create → task_id={task_doc['task_id']}, session_id={session_id}, "
        f"mode={mode}, preferred_type={preferred_type}, stream={bool(stream)}, "
        f"scholar={scholar_id}"
    )
    return task_doc


async def claim_task(db, task_id: str) -> bool:
    """原子抢占 pending → processing（并发安全，同 v3）。"""
    res = await db.update(
        COLLECTION,
        where={"task_id": task_id, "status": STATUS_PENDING},
        data={"$set": {"status": STATUS_PROCESSING, "updated_at": _now_ms()}},
        multi=False,
    )
    return res.get("modified_count", 0) > 0


def _build_stale_error() -> dict:
    """卡死任务恢复的 error 对象（卡死自愈 → LLM_TIMEOUT，同 v3）。"""
    return {
        "error_code": ERR_LLM_TIMEOUT,
        "error_detail": "执行超时",
        "failure_stage": STAGE_LLM,
        "llm_timeout_seconds": SESSION_LLM_TIMEOUT_SECONDS,
        "raw": None,
    }


async def write_partial(
    db,
    task_id: str,
    text: str,
    *,
    force: bool = False,
) -> bool:
    """节流写伪流式增量（契约 §3.17：partial_text / partial_seq / partial_updated_at）。

    节流：距上次写入 ≥ `SESSION_V4_PARTIAL_THROTTLE_MS` **或** 新增字符 ≥ `SESSION_V4_PARTIAL_MIN_CHARS`；
    单轮写入次数达 `SESSION_V4_PARTIAL_MAX_WRITES` 后停止（写放大保护）。
    `force=True` 忽略节流（终态前最后一次补齐用）。
    """
    if not text:
        return False
    now = _now_ms()
    state = _partial_state.setdefault(task_id, {"last_ms": 0, "length": 0, "writes": 0})
    if state["writes"] >= SESSION_V4_PARTIAL_MAX_WRITES:
        return False
    if not force:
        throttled = (
            now - state["last_ms"] < SESSION_V4_PARTIAL_THROTTLE_MS
            and len(text) - state["length"] < SESSION_V4_PARTIAL_MIN_CHARS
        )
        if throttled:
            return False

    try:
        res = await db.update(
            COLLECTION,
            where={"task_id": task_id},
            data={
                "$set": {
                    "partial_text": text,
                    "partial_seq": state["writes"] + 1,
                    "partial_updated_at": now,
                    "updated_at": now,
                }
            },
            multi=False,
        )
    except Exception as e:  # noqa: BLE001 — 增量写失败不阻断生成主流程
        logger.warning(f"[session_v4] write_partial 失败 → task_id={task_id}: {e}")
        return False

    state["last_ms"] = now
    state["length"] = len(text)
    state["writes"] += 1
    return res.get("modified_count", 0) > 0


def _clear_partial_state(task_id: str) -> None:
    _partial_state.pop(task_id, None)


async def finish_task(
    db,
    task_id: str,
    *,
    result: dict | list | None = None,
    error: dict | None = None,
    timings_extra: dict | None = None,
) -> None:
    """写回执行结果：error 非空 → failed(+error)，否则 success(+result)。

    终态约定（契约 §3.17）：
    - 清空 `partial_text`（增量脏字段不得与 `result` 共存）；
    - 合并 `timings`（补 `total_ms` 等），保留终态耗时用于验收。
    """
    now = _now_ms()
    prev = await get_task(db, task_id)
    timings = dict((prev or {}).get("timings") or {})
    if prev and prev.get("created_at"):
        timings["total_ms"] = now - int(prev["created_at"])
    for key, value in (timings_extra or {}).items():
        if value is not None:
            timings[key] = value

    if error is not None:
        status = STATUS_FAILED
        result_value = None
        error_value = error
        logger.info(
            f"[session_v4] fail → task_id={task_id}, error={error.get('error_code')}"
        )
    else:
        status = STATUS_SUCCESS
        result_value = result
        error_value = None
        logger.info(f"[session_v4] done → task_id={task_id}, status=success")

    await db.update(
        COLLECTION,
        where={"task_id": task_id},
        data={
            "$set": {
                "status": status,
                "result": result_value,
                "error": error_value,
                "partial_text": None,
                "timings": timings,
                "updated_at": now,
            }
        },
        multi=False,
    )
    _clear_partial_state(task_id)


async def get_task(db, task_id: str) -> dict | None:
    """按 task_id 查询任务，未命中返回 None（不过滤 TTL，接口层自行过滤）。"""
    res = await db.query(COLLECTION, where={"task_id": task_id}, limit=1)
    records = res.get("records", [])
    return records[0] if records else None


async def cleanup_expired(db, now_ms: int | None = None) -> int:
    """删除 expires_at <= now 的过期任务，返回删除数量（TTL 口径同 v3）。"""
    now = now_ms if now_ms is not None else _now_ms()
    res = await db.delete(COLLECTION, where={"expires_at": {"$lte": now}})
    count = res.get("deleted_count", 0)
    if count:
        logger.info(f"[session_v4] cleanup → 删除过期任务 {count} 条")
    return count


async def _release_session_slot(db, session_id: str | None, task_id: str | None) -> None:
    """释放会话在途位：`ai_session_v4.pending_task == task_id` 时置 null。"""
    if not session_id or not task_id:
        return
    try:
        await db.update(
            SESSION_COLLECTION,
            where={"session_id": session_id, "pending_task": task_id},
            data={"$set": {"pending_task": None, "updated_at": _now_ms()}},
            multi=False,
        )
    except Exception as e:  # noqa: BLE001 — 释放失败不影响任务收尾
        logger.error(f"[session_v4] 释放会话在途位失败 session={session_id}: {e}")


async def recover_stale_tasks(db, timeout_s: int | None = None) -> int:
    """巡检卡死的 processing 任务：超时未更新 → 置 failed，并同步释放会话在途位。"""
    timeout = timeout_s or SESSION_LLM_TIMEOUT_SECONDS
    now = _now_ms()
    threshold = now - timeout * 1000
    stale_where = {
        "status": STATUS_PROCESSING,
        "updated_at": {"$lt": threshold},
    }
    res = await db.query(COLLECTION, where=stale_where, limit=1000)
    stale_tasks = res.get("records", [])
    if not stale_tasks:
        return 0
    upd = await db.update(
        COLLECTION,
        where=stale_where,
        data={
            "$set": {
                "status": STATUS_FAILED,
                "error": _build_stale_error(),
                "updated_at": now,
            }
        },
        multi=True,
    )
    count = upd.get("modified_count", 0)
    if count:
        logger.info(f"[session_v4] recover → 卡死任务标记 failed {count} 条")
    for t in stale_tasks:
        await _release_session_slot(db, t.get("session_id"), t.get("task_id"))
        _clear_partial_state(t.get("task_id") or "")
    return count


async def recover_task_if_stale(db, task: dict, timeout_s: int | None = None) -> bool:
    """定点恢复：单条卡死 processing 任务 → 置 failed，并同步释放会话在途位。"""
    if task.get("status") != STATUS_PROCESSING:
        return False
    timeout = timeout_s or SESSION_LLM_TIMEOUT_SECONDS
    now = _now_ms()
    if task.get("updated_at", 0) > now - timeout * 1000:
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
            f"[session_v4] recover → task_id={task['task_id']} 卡死任务标记 failed"
        )
        await _release_session_slot(db, task.get("session_id"), task["task_id"])
        _clear_partial_state(task["task_id"])
        return True
    return False


def _session_gen_error_to_dict(e: Exception) -> dict | None:
    """把生成域业务异常映射为任务 error 对象；非生成域异常返回 None（通用兜底）。"""
    if isinstance(e, SessionGenError):
        return e.to_dict(llm_timeout_seconds=SESSION_LLM_TIMEOUT_SECONDS)
    return None


def build_on_delta(db, task_id: str, task: dict) -> OnDelta | None:
    """构造伪流式增量回调（已接线：`run_session_task` → `dialogue_engine`）。

    仅在「请求 stream=true 且 SESSION_V4_STREAM_ENABLED=1」时返回回调；否则返回 None，
    执行路径与 v3 完全一致（S5 判据）。

    回调为 **async**（由 `_default_stream_generator` await），内部累积文本并节流落库
    （见 `write_partial`）；累积缓冲挂在回调的 `buffer` 属性上，供终态取
    `partial_first_ms`（S6 埋点）。
    """
    if not stream_enabled_for(task.get("stream")):
        return None

    buffer: dict[str, Any] = {"text": "", "first_at": None}

    async def _on_delta(delta: str) -> None:
        if not delta:
            return
        buffer["text"] += delta
        if buffer["first_at"] is None:
            buffer["first_at"] = _now_ms()
        await write_partial(db, task_id, buffer["text"])

    _on_delta.buffer = buffer  # type: ignore[attr-defined] — 供 timings 取首帧时间
    return _on_delta


def partial_first_ms_of(on_delta: OnDelta | None) -> int | None:
    """取增量回调的首帧时间戳（ms）；无回调/未产出增量时返回 None。"""
    buffer = getattr(on_delta, "buffer", None) if on_delta is not None else None
    return (buffer or {}).get("first_at")


async def run_session_task(task_id: str) -> None:
    """后台执行会话 v4 生成任务并写回结果。

    由提交接口 `asyncio.create_task(...)` 调度，与请求解耦：
    - claim_task 原子抢占：被其他实例抢占则直接返回，避免重复执行
    - 生成：任务 `context` 自包含快照即执行唯一依据，走 v3 同款引擎
      （services.learning.dialogue_engine，超时 SESSION_LLM_TIMEOUT_SECONDS）
    - 伪流式：`build_on_delta` 返回回调时透传给引擎（S6 已接线；回调为空则路径 ≡ v3）
    - 成功：回写 history 并释放 pending_task（session_state_v4.complete_turn），finish success
    - 失败：释放 pending_task（不污染 history），finish failed（不降级、不静默）
    """
    db = get_db()
    task = await get_task(db, task_id)
    if task is None:
        logger.info(f"[session_v4] run skip → task_id={task_id} 不存在（已 TTL 清理）")
        return
    if not await claim_task(db, task_id):
        logger.info(f"[session_v4] run skip → task_id={task_id} 已被抢占或状态非 pending")
        return

    session_id = task.get("session_id")
    result: dict | None = None
    error: dict | None = None
    timings_extra: dict = {"claimed_at": _now_ms()}
    llm_started = _now_ms()
    # 伪流式增量回调（stream=true + 服务级开关开启时非空；否则 None → 路径 ≡ v3）
    on_delta = build_on_delta(db, task_id, task)

    try:
        payload = await dialogue_engine.generate_session_reply(
            db=db,
            session_id=session_id,
            context=task.get("context") or {},
            preferred_type=task.get("preferred_type", "auto"),
            on_delta=on_delta,
            thinking_disabled=bool(SESSION_V4_THINKING_DISABLED),
            checkpoint_collection=SESSION_V4_CHECKPOINT_COLLECTION,
        )
        timings_extra["llm_total_ms"] = _now_ms() - llm_started
        # 伪流式首帧埋点：首个增量到达时刻 - 任务提交时刻（契约 §3.17 timings）
        submitted_at = int((task.get("timings") or {}).get("submitted_at") or 0)
        first_at = partial_first_ms_of(on_delta)
        if first_at and submitted_at:
            timings_extra["partial_first_ms"] = first_at - submitted_at
        result = {
            "session_id": session_id,
            "content_type": payload["content_type"],
            "ai_text": payload["ai_text"],
            "hint": payload.get("hint"),
            "suggested_targets": payload.get("suggested_targets") or [],
        }
        # 仅 success 回写会话历史并释放在途位
        if session_id:
            ctx = task.get("context") or {}
            await session_state.complete_turn(
                db,
                session_id=session_id,
                task_id=task_id,
                user_text=(
                    ctx.get("user_input") if task.get("mode") == "turn" else None
                ),
                assisted=bool(ctx.get("assisted")),
                ai_text=result["ai_text"],
                content_type=result["content_type"],
                suggested_targets=result.get("suggested_targets"),
            )
    except Exception as e:  # noqa: BLE001
        error = _session_gen_error_to_dict(e)
        if error is None:
            logger.error(f"[session_v4] run error → task_id={task_id}: {e}", exc_info=True)
            error = {
                "error_code": ERR_NETWORK_ERROR,
                "error_detail": str(e)[:500],
                "failure_stage": STAGE_LLM,
                "llm_timeout_seconds": SESSION_LLM_TIMEOUT_SECONDS,
                "raw": None,
            }
        # 失败：释放在途位（不污染 history）；自身异常仅记日志不阻断任务收尾
        if session_id:
            try:
                await session_state.release_pending(
                    db, session_id=session_id, task_id=task_id
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"[session_v4] 释放在途位失败 session_id={session_id}: {exc}",
                    exc_info=True,
                )
    await finish_task(db, task_id, result=result, error=error, timings_extra=timings_extra)
