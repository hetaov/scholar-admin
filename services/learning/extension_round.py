"""语言点造句「多轮」会话仓储 — `extension_round` 集合 CRUD 与状态流转

契约：data-model-contract.md §4.28；api-contract.md §3.18（E7 / E8 / E9）
设计：docs_v1/扩展/第三期-语言点造句多轮-v1.md §3.1（模块划分）/ §3.6（集合与索引）
账本：docs_v1/扩展/第三期-语言点造句多轮-任务拆分与断点-v1.md（B02）

定位：**多轮会话的服务端状态**（ADR-0031）。一期 L2 是单轮（题面为整句 `translation`、
服务端无状态），做不了「下一轮换情景」「针对上一轮错误给提示」。本集合持有
「第几轮 / 上一轮错在哪 / 出过哪些情景」——这三条**在客户端不可信**（切句即丢、
可伪造、跨端不一致），因此必须在服务端。

状态机（单向，禁止回退）：
    active ──┬──> passed     达标（E8 写）
             ├──> exhausted  轮次用尽未达标（E8 写）
             └──> abandoned  学习者放弃

`turn_index` **只在 `append_turn`（下发新一轮情景）时前进**；判分写回结果不前进，
出题 LLM 失败亦不前进（红线 R13）。

红线（契约 §4.28）：
- **R10 不写 mastery status**：本模块只写 `extension_round`，不调用任何既有写状态接口，
  不进 `status>=3` 达标 / `status>=5` 出队双口径；
- **R11 不回写 `english_extension_point`**：`points_snapshot` 只在会话内，反向写回会污染
  AI 层缓存与跨句去重；
- **R12 轮次数据仅本人可见**：`get_round` 强制校验 `scholar_id` 归属，不匹配 → None
  （E9 → 404 `ROUND_NOT_FOUND`，防 `round_id` 枚举读 `turns[]`）；
- **R14 文档体积**：`max_turns ≤ EXTENSION_ROUND_MAX_TURNS_HARD`、`user_input` 长度封顶。
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

import config
from services.providers.extension_llm import ExtensionError

logger = logging.getLogger("scholar-admin.learning.extension_round")

COLLECTION = config.EXTENSION_ROUND_COLLECTION

STATUS_ACTIVE = "active"
STATUS_PASSED = "passed"
STATUS_EXHAUSTED = "exhausted"
STATUS_ABANDONED = "abandoned"

# 终态：closed 之后不可再写轮次（状态机单向）
TERMINAL_STATUSES = (STATUS_PASSED, STATUS_EXHAUSTED, STATUS_ABANDONED)

# 失败阶段（与 ExtensionError 的 failure_stage 对齐）
STAGE_INPUT = "input"
STAGE_ROUND = "round"

# 业务码（api-contract §3.18）
ERR_INVALID_INPUT = "INVALID_INPUT"
ERR_ROUND_NOT_FOUND = "ROUND_NOT_FOUND"
ERR_ROUND_CLOSED = "ROUND_CLOSED"
ERR_ROUND_TURN_MISMATCH = "ROUND_TURN_MISMATCH"

_WS_RE = re.compile(r"\s+")


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_round_id() -> str:
    """生成会话 ID：`rnd_` + 32 位 uuid hex（契约 §4.28）。"""
    return "rnd_" + uuid.uuid4().hex


def build_prompt_fingerprint(prompt_zh: str, scene_tag: list[str] | None = None) -> str:
    """情景句去重指纹（**纯函数**）：`prompt_zh` 归一化 + `scene_tag` 排序归一。

    `used_prompts[]` 存的就是它：出题侧比对新情景与历史是否重复（场景 / 人物 / 动作
    至少换一项），B05 的 `is_repeat_prompt` 直接用本函数的输出做相等/相似判定。
    """
    text = _WS_RE.sub("", str(prompt_zh or "")).lower()
    tags = sorted(
        t.strip().lower() for t in (scene_tag or []) if str(t or "").strip()
    )
    return f"{text}#{','.join(tags)}"


# ===========================================================================
# 读
# ===========================================================================


async def _load(db, round_id: str) -> dict | None:
    """按 round_id 读文档（**不做归属与 TTL 过滤**，内部用）。"""
    res = await db.query(COLLECTION, where={"round_id": round_id}, limit=1)
    records = res.get("records", [])
    return records[0] if records else None


async def get_round(
    db,
    round_id: str,
    *,
    scholar_id: str | None = None,
    now_ms: int | None = None,
) -> dict | None:
    """E9：读会话详情。未命中 / 归属不匹配 / 已过期 → **一律 None**。

    调用方（E9 路由）把 None 映射为 **404 `ROUND_NOT_FOUND`**：
    - 归属不匹配也不区分「不存在」与「不属于你」（R12：防 `round_id` 枚举读 `turns[]`）；
    - `expires_at <= now` 视为不存在（TTL 24h，契约 §4.28）。
    """
    doc = await _load(db, round_id)
    if doc is None:
        return None
    if scholar_id is not None and doc.get("scholar_id") != scholar_id:
        logger.info(f"[extension_round] get deny → round_id={round_id} 归属不匹配")
        return None
    now = now_ms if now_ms is not None else _now_ms()
    expires_at = doc.get("expires_at")
    if isinstance(expires_at, int) and expires_at <= now:
        return None
    return doc


def _require_active(doc: dict | None) -> dict:
    """会话必须存在且处于 `active`（终态不可再写轮次）。"""
    if doc is None:
        raise ExtensionError(
            ERR_ROUND_NOT_FOUND, STAGE_ROUND, "会话不存在或已过期"
        )
    if doc.get("status") != STATUS_ACTIVE:
        raise ExtensionError(
            ERR_ROUND_CLOSED,
            STAGE_ROUND,
            f"会话已结束（status={doc.get('status')}），不可继续作答",
        )
    return doc


async def _save(db, doc: dict, fields: dict) -> dict:
    """写回字段（只 $set 标量 / 整体数组，不用 $push —— 兼容 FakeDB 与真实客户端）。"""
    now = _now_ms()
    payload = {**fields, "updated_at": now}
    await db.update(
        COLLECTION,
        where={"round_id": doc["round_id"]},
        data={"$set": payload},
        multi=False,
    )
    return {**doc, **payload}


# ===========================================================================
# 写
# ===========================================================================


async def create_round(
    db,
    *,
    scholar_id: str,
    sentence_id: str,
    selected_ids: list[str],
    points_snapshot: list[dict],
    original: str = "",
    translation: str = "",
    textbook_id: str | None = None,
    lesson_id: str | None = None,
    max_turns: int | None = None,
    register: str = "auto",
    difficulty: str = "same",
) -> dict:
    """E7：建会话（**不含第 1 轮情景**，情景由出题链路 `append_turn` 写入）。

    `register` / `difficulty`（**2026-10-05 F9 补订**，data-model §4.28）：E7 传入的
    **出题偏好**，落会话供**第 2 轮及以后**出题取用（E8 入参不含二者 → 服务端自取）。
    枚举校验放在路由入参侧；老文档无此字段 → 读侧 `.get()` 按 `auto` / `same` 回落
    （既有行零迁移，同 `extension_task.round_action` 范式）。

    入参校验（红线 R14 / 契约 §4.28）：
    - `scholar_id` / `sentence_id` 空 → INVALID_INPUT；
    - `selected_ids` 空或超过 `EXTENSION_ROUND_MAX_SELECTED` → INVALID_INPUT；
    - `max_turns` 缺省取 `EXTENSION_ROUND_MAX_TURNS`，越界（<1 或 > 硬上限）→ INVALID_INPUT。

    `turn_index` 起始为 **0**：第 1 轮情景由 `append_turn` 写入后变为 1。
    """
    if not scholar_id or not str(scholar_id).strip():
        raise ExtensionError(ERR_INVALID_INPUT, STAGE_INPUT, "scholar_id 不能为空")
    if not sentence_id or not str(sentence_id).strip():
        raise ExtensionError(ERR_INVALID_INPUT, STAGE_INPUT, "sentence_id 不能为空")

    selected = list(selected_ids or [])
    if not selected:
        raise ExtensionError(ERR_INVALID_INPUT, STAGE_INPUT, "selected_ids 不能为空")
    if len(selected) > config.EXTENSION_ROUND_MAX_SELECTED:
        raise ExtensionError(
            ERR_INVALID_INPUT,
            STAGE_INPUT,
            f"单次会话最多勾选 {config.EXTENSION_ROUND_MAX_SELECTED} 个语言点，"
            f"当前 {len(selected)} 个",
        )

    turns_cap = (
        max_turns if max_turns is not None else config.EXTENSION_ROUND_MAX_TURNS
    )
    try:
        turns_cap = int(turns_cap)
    except (TypeError, ValueError):
        raise ExtensionError(ERR_INVALID_INPUT, STAGE_INPUT, "max_turns 必须为整数")
    if turns_cap < 1 or turns_cap > config.EXTENSION_ROUND_MAX_TURNS_HARD:
        raise ExtensionError(
            ERR_INVALID_INPUT,
            STAGE_INPUT,
            f"max_turns 必须在 1~{config.EXTENSION_ROUND_MAX_TURNS_HARD} 之间，"
            f"当前 {turns_cap}",
        )

    now = _now_ms()
    doc: dict[str, Any] = {
        "round_id": build_round_id(),
        "scholar_id": scholar_id,
        "sentence_id": sentence_id,
        "textbook_id": textbook_id,
        "lesson_id": lesson_id,
        # 会话内锁定：中途不可换（换 = 新开 round）
        "selected_ids": selected,
        # 落快照：判分不回查 english_extension_point（R11）
        "points_snapshot": list(points_snapshot or []),
        "original": original or "",
        "translation": translation or "",
        # F9（2026-10-05）：出题偏好落库，供第 2 轮及以后出题取用
        "register": register or "auto",
        "difficulty": difficulty or "same",
        "max_turns": turns_cap,
        "turn_index": 0,
        "status": STATUS_ACTIVE,
        "turns": [],
        "used_prompts": [],
        "summary": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + config.EXTENSION_ROUND_TTL_HOURS * 60 * 60 * 1000,
    }
    await db.insert(COLLECTION, doc)
    logger.info(
        f"[extension_round] create → round_id={doc['round_id']}, "
        f"sentence_id={sentence_id}, max_turns={turns_cap}"
    )
    return doc


async def append_turn(
    db,
    round_id: str,
    *,
    turn: dict,
    prompt_fingerprint: str | None = None,
) -> dict:
    """下发新一轮情景：`turns[]` append 一条 + `used_prompts[]` 记指纹 + `turn_index` 前进。

    - `turn_index` **由服务端按 `len(turns)+1` 计算**，不信任入参（防伪造轮次）；
    - 已到 `max_turns` 仍调用 → INVALID_INPUT（轮次封顶，R14）；
    - 会话非 active → ROUND_CLOSED；不存在 → ROUND_NOT_FOUND；
    - `user_input` / `result` 恒初始化为 None / None：本轮刚下发，尚未作答。
    - `prompt_fingerprint` 为空时按 `prompt_zh` + `scene_tag` 现算（B05 出题侧可显式传入）。
    """
    doc = _require_active(await _load(db, round_id))
    turns = list(doc.get("turns") or [])
    if len(turns) >= int(doc.get("max_turns") or 0):
        raise ExtensionError(
            ERR_INVALID_INPUT,
            STAGE_ROUND,
            f"轮次已达上限 {doc.get('max_turns')}",
        )

    src = dict(turn or {})
    fingerprint = prompt_fingerprint or build_prompt_fingerprint(
        src.get("prompt_zh", ""), src.get("scene_tag") or []
    )
    new_turn = {
        "turn_index": len(turns) + 1,
        "prompt_zh": src.get("prompt_zh", ""),
        "register": src.get("register", ""),
        # 第 1 轮为空串；后续针对上一轮 errors[0]
        "focus_hint": src.get("focus_hint", "") or "",
        "scene_tag": list(src.get("scene_tag") or []),
        "must_use": list(src.get("must_use") or []),
        # 出题时落库但**不在本轮下发**（防泄题，决策 D5）
        "reference_en": src.get("reference_en", "") or "",
        "user_input": None,
        "result": None,
        "repeat_risk": bool(src.get("repeat_risk", False)),
        "at": _now_ms(),
    }

    used = list(doc.get("used_prompts") or [])
    if fingerprint:
        used.append(fingerprint)

    return await _save(
        db,
        doc,
        {
            "turns": turns + [new_turn],
            "used_prompts": used,
            "turn_index": new_turn["turn_index"],
        },
    )


async def replace_last_turn_prompt(
    db,
    round_id: str,
    *,
    turn: dict,
) -> dict:
    """把**末轮**（尚未作答的那一轮）的题面改指为 `turn` —— 供「同题重提（改一次）」使用。

    语义（2026-10-07，真机走查裁定）：客户端在未达标后仍显示**上一轮**题面并重提时，服务端把
    那一轮题面**重新下发为当前轮**，随后照常判分写回 ⇒ 该次作答**同样消耗一轮**（`turn_index` 不变、
    **不新增轮次**）。与 `append_turn` 的差别：**不改 `turn_index`、不 append `turns`、不追加 `used_prompts`**
    （同一题面的指纹已在其中，避免重复记账）。

    - `turn_index`（服务端当前轮）不变；`result` / `user_input` 复位为 None（替换后由 `append_turn_result` 写入）；
    - 仅允许在**末轮尚未作答**时替换（`result is None`）→ 否则 ROUND_TURN_MISMATCH（防覆盖已判分的轮）；
    - 会话非 active → ROUND_CLOSED；不存在 → ROUND_NOT_FOUND。
    """
    doc = _require_active(await _load(db, round_id))
    turns = [dict(t) for t in (doc.get("turns") or [])]
    if not turns:
        raise ExtensionError(
            ERR_ROUND_TURN_MISMATCH, STAGE_ROUND, "会话尚无轮次，无法改写题面"
        )
    idx = len(turns) - 1
    if turns[idx].get("result") is not None:
        raise ExtensionError(
            ERR_ROUND_TURN_MISMATCH,
            STAGE_ROUND,
            "当前轮已判分，不能改写题面",
        )
    src = dict(turn or {})
    replaced = {
        **turns[idx],
        "prompt_zh": src.get("prompt_zh", ""),
        "register": src.get("register", ""),
        "focus_hint": src.get("focus_hint", "") or "",
        "scene_tag": list(src.get("scene_tag") or []),
        "must_use": list(src.get("must_use") or []),
        "reference_en": src.get("reference_en", "") or "",
        "repeat_risk": bool(src.get("repeat_risk", False)),
        "user_input": None,
        "result": None,
        "at": _now_ms(),
    }
    turns[idx] = replaced
    return await _save(db, doc, {"turns": turns})


async def append_turn_result(
    db,
    round_id: str,
    *,
    turn_index: int,
    result: dict,
    user_input: str | None = None,
    repeat_risk: bool | None = None,
) -> dict:
    """写回本轮判分结果（**不前进 `turn_index`** —— 推进由 `append_turn` 负责）。

    - `turn_index` 与服务端当前轮不一致 → ROUND_TURN_MISMATCH（防重复 / 乱序提交）；
    - `user_input` 长度超过 `EXTENSION_ROUND_MAX_INPUT_LEN` → INVALID_INPUT（R14）；
    - LLM 失败不该调用本函数：失败时本轮保持 `result=None`、`turn_index` 不变（R13）。
    """
    doc = _require_active(await _load(db, round_id))
    turns = [dict(t) for t in (doc.get("turns") or [])]
    current = int(doc.get("turn_index") or 0)
    if int(turn_index) != current:
        raise ExtensionError(
            ERR_ROUND_TURN_MISMATCH,
            STAGE_ROUND,
            f"轮次不匹配：服务端当前第 {current} 轮，提交的是第 {turn_index} 轮",
        )
    if not turns:
        raise ExtensionError(
            ERR_ROUND_TURN_MISMATCH, STAGE_ROUND, "会话尚无轮次，无法写入结果"
        )

    idx = len(turns) - 1
    if turns[idx].get("turn_index") != current:
        raise ExtensionError(
            ERR_ROUND_TURN_MISMATCH,
            STAGE_ROUND,
            f"轮次数据不一致：turns 末位为第 {turns[idx].get('turn_index')} 轮",
        )

    if user_input is not None:
        if len(str(user_input)) > config.EXTENSION_ROUND_MAX_INPUT_LEN:
            raise ExtensionError(
                ERR_INVALID_INPUT,
                STAGE_INPUT,
                f"user_input 超过 {config.EXTENSION_ROUND_MAX_INPUT_LEN} 字符",
            )
        turns[idx]["user_input"] = user_input
    turns[idx]["result"] = dict(result or {})
    if repeat_risk is not None:
        turns[idx]["repeat_risk"] = bool(repeat_risk)

    return await _save(db, doc, {"turns": turns})


async def close_round(
    db,
    round_id: str,
    *,
    status: str,
    summary: dict,
) -> dict:
    """结束会话并写小结（**只接受 `passed` / `exhausted`**；放弃走 `abandon_round`）。

    `summary` = `{ turns_used, passed, best_score, top_errors[], model_sentences[] }`。
    """
    if status not in (STATUS_PASSED, STATUS_EXHAUSTED):
        raise ExtensionError(
            ERR_INVALID_INPUT,
            STAGE_ROUND,
            f"close_round 只接受 {STATUS_PASSED} / {STATUS_EXHAUSTED}，当前 {status!r}",
        )
    doc = _require_active(await _load(db, round_id))
    updated = await _save(
        db, doc, {"status": status, "summary": dict(summary or {})}
    )
    logger.info(
        f"[extension_round] close → round_id={round_id}, status={status}, "
        f"turns_used={(summary or {}).get('turns_used')}"
    )
    return updated


async def abandon_round(
    db,
    round_id: str,
    *,
    scholar_id: str | None = None,
    summary: dict | None = None,
) -> dict:
    """学习者放弃本轮会话 → `abandoned`（终态，不可回退）。

    `scholar_id` 传入时做归属校验（R12），不匹配 → None（路由侧映射 404）。
    """
    doc = await _load(db, round_id)
    if doc is None:
        raise ExtensionError(ERR_ROUND_NOT_FOUND, STAGE_ROUND, "会话不存在或已过期")
    if scholar_id is not None and doc.get("scholar_id") != scholar_id:
        raise ExtensionError(ERR_ROUND_NOT_FOUND, STAGE_ROUND, "会话不存在或已过期")
    _require_active(doc)
    payload: dict[str, Any] = {"status": STATUS_ABANDONED}
    if summary is not None:
        payload["summary"] = dict(summary)
    updated = await _save(db, doc, payload)
    logger.info(f"[extension_round] abandon → round_id={round_id}")
    return updated


async def cleanup_expired(db, now_ms: int | None = None) -> int:
    """删除 `expires_at <= now` 的过期会话，返回删除条数。

    会话服务端有状态（ADR-0031），TTL 必须有界，避免孤儿文档无限堆积。
    调度方见账本 §5（评审 F7：运维面由 B03 之后的运维脚本 / 定时任务挂载）。
    """
    now = now_ms if now_ms is not None else _now_ms()
    res = await db.delete(COLLECTION, where={"expires_at": {"$lte": now}})
    count = res.get("deleted_count", 0)
    if count:
        logger.info(f"[extension_round] cleanup → 删除过期会话 {count} 条")
    return count
