"""学习者校对 overlay（2026-10-03 修订 2 新增）

契约：api-contract.md §3.18（E1 读侧合并 / E4' 提交补丁 / E6 我的校对与历史）
      data-model-contract.md §4.26 `extension_review` / §4.27 `extension_review_log`
      service-contract.md §8.7（`extension_review.py` 6 函数）

定位：**校对是学习者的学习行为，不是系统的统一判断**。
`source=manual` 全局人工覆盖层已下线（B28），本模块承载 per-scholar 的 overlay：

- `english_extension_point` **永远只存 AI 产出**（`llm` / `rule`），本模块**不反向写回**；
- 学习者判断只写 `extension_review`（补丁）与 `extension_review_log`（append-only 历史）；
- **零 mastery 写入**（红线 R9）：不进 `status>=3` 达标 / `status>=5` 出队双口径，
  不调用任何既有写状态接口。

删除主键用**归一化 `text`** 而非 `id`：`id` 在 `rank_points` 之后重算，prompt 升版
或排序变化会让 `id` 漂移，按 `id` 删会误删。
"""
from __future__ import annotations

import logging
import re
import string
import time
import uuid

import config
from services.providers.extension_llm import ExtensionError

logger = logging.getLogger("scholar-admin.english.extension_review")

# 自建点允许的类型（契约 §4.26：只做「标记 + 复习素材」，不做判分，故类型收敛）
_MINE_TYPES = ("word", "phrase")

# 自建点文本长度上限（契约 §4.26）
_MINE_TEXT_MAX_LEN = 60

# 历史返回条数（契约 §3.18 E6：最近 50 条）
_HISTORY_LIMIT = 50

# 首尾标点：ASCII + 常用中文/全角。仅用于 strip 两端，不动内部字符。
_EDGE_PUNCT = string.punctuation + "。，、；：？！“”‘’（）《》【】〈〉「」『』…—·～"


def normalize_point_text(text: str) -> str:
    """归一化语言点文本：删除比对的主键。

    规则（契约 §3.18）：`lower()` → 去首尾标点 → 空白折叠。
    覆盖大小写 / 首尾标点 / 多空格三类差异；纯函数，无副作用。
    """
    if not text:
        return ""
    s = str(text).lower().strip()
    s = s.strip(_EDGE_PUNCT).strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _mine_point(item: dict, index: int) -> dict:
    """把 `extension_review.added` 的一项整形为 `points[]` 元素（schema 子集）。

    契约 §3.18 `origin` 行：`{ id: "mine_<n>", type, text, meaning_zh, note, origin }`，
    **恒 `l1 = null`** —— 自建点没有后端生成的题面与答案键，不进判分（D2）。
    """
    return {
        "id": item.get("id") or f"mine_{index}",
        "type": item.get("type", ""),
        "text": item.get("text", ""),
        "meaning_zh": item.get("meaning_zh", ""),
        "note": item.get("note", ""),
        "origin": "mine",
        "l1": None,
    }


def merge_review_points(
    ai_points: list[dict],
    review: dict | None,
    content_hash: str,
) -> tuple[list[dict], dict]:
    """读侧合并 AI 点与学习者 overlay（**纯函数，不写库、不写回 AI 集合**）。

    顺序（契约 §3.18「学习者 overlay 合并规则」六步）：
      1. base = ai_points（含 `l1` 题面）
      2. `content_hash` 不一致 → **不应用补丁**，仅回 `review.stale = true`
      3. AI 点按归一化 text 命中 `removed_texts` → 剔除
      4. 追加 `added`（`origin="mine"`、`l1=None`）
      5. 自建点不占配额、不计入 `count_by_type`（由调用方保持既有语义）
      6. 每点补 `origin`；出参补 `review` 摘要

    Returns:
        (points, review_summary)；`review_summary` =
        `{ has_review, added_count, removed_count, stale, updated_at }`
    """
    if not review:
        # 未传 scholar_id 或该学习者未校对 → 纯 AI 结果
        return _with_origin(ai_points, "ai"), {
            "has_review": False,
            "added_count": 0,
            "removed_count": 0,
            "stale": False,
            "updated_at": None,
        }

    stale = review.get("content_hash") != content_hash
    removed_texts = review.get("removed_texts") or []
    added = review.get("added") or []

    if stale:
        # 步骤 2：原句已更新 → 不应用补丁（不自动删除学习者标注，红线 ⑦）
        return _with_origin(ai_points, "ai"), {
            "has_review": True,
            "added_count": len(added),
            "removed_count": len(removed_texts),
            "stale": True,
            "updated_at": review.get("updated_at"),
        }

    removed_keys = {normalize_point_text(t) for t in removed_texts}
    kept = [p for p in ai_points if normalize_point_text(p.get("text", "")) not in removed_keys]

    merged = _with_origin(kept, "ai")
    merged.extend(_mine_point(a, i) for i, a in enumerate(added))

    return merged, {
        "has_review": True,
        "added_count": len(added),
        "removed_count": len(ai_points) - len(kept),
        "stale": False,
        "updated_at": review.get("updated_at"),
    }


def _with_origin(points: list[dict], origin: str) -> list[dict]:
    """给每个点补 `origin`（浅拷贝，不改入参 —— 纯函数）。"""
    return [{**p, "origin": origin} for p in points]


# ===========================================================================
# E4' / E6：提交「我的补丁」+ 我的校对与学习历史
# ===========================================================================


def _gate() -> None:
    """校对开关门控（与 EXTENSION_ENABLED 相互独立）。"""
    if not config.EXTENSION_REVIEW_ENABLED:
        raise ExtensionError(
            "EXTENSION_DISABLED",
            "gate",
            "学习者校对功能未开启（请配置 EXTENSION_REVIEW_ENABLED=1）",
        )


def _validate(scholar_id: str, added: list[dict]) -> None:
    """入参强校验：`scholar_id` 空 / `type` 越界 / 条数或长度超限 → INVALID_INPUT。"""
    if not scholar_id or not str(scholar_id).strip():
        raise ExtensionError("INVALID_INPUT", "input", "scholar_id 不能为空")

    if len(added) > config.EXTENSION_REVIEW_MAX_ADDED:
        raise ExtensionError(
            "INVALID_INPUT",
            "input",
            f"自建语言点最多 {config.EXTENSION_REVIEW_MAX_ADDED} 条，当前 {len(added)} 条",
        )

    for i, item in enumerate(added):
        ptype = item.get("type", "")
        if ptype not in _MINE_TYPES:
            raise ExtensionError(
                "INVALID_INPUT", "input", f"added[{i}].type 必须为 word 或 phrase，当前 {ptype!r}"
            )
        text = (item.get("text") or "").strip()
        if not text:
            raise ExtensionError("INVALID_INPUT", "input", f"added[{i}].text 不能为空")
        if len(text) > _MINE_TEXT_MAX_LEN:
            raise ExtensionError(
                "INVALID_INPUT",
                "input",
                f"added[{i}].text 超过 {_MINE_TEXT_MAX_LEN} 字符",
            )


def diff_review(old_review: dict | None, new_review: dict) -> list[dict]:
    """全量 overlay 与旧值 diff → append-only 历史事件（**纯函数**）。

    三类动作（契约 §3.18 E4'）：
      - `remove`  ：学习者剔除某点（新进 `removed_texts`；或自建点从 `added` 中移除）
      - `restore` ：学习者恢复某点（从 `removed_texts` 中去掉）
      - `add`     ：学习者新建点（新进 `added`）

    Returns:
        `[{action, target, payload}]`，`target` 为归一化文本；`add` 的 `payload` 含
        自建点全量，其余为 `{}`。顺序按 (action, target) 稳定排序，便于断言与展示。
    """
    old = old_review or {}
    old_removed = {normalize_point_text(t) for t in old.get("removed_texts") or []}
    new_removed = {normalize_point_text(t) for t in new_review.get("removed_texts") or []}
    old_added = {normalize_point_text(a.get("text", "")): a for a in old.get("added") or []}
    new_added = {normalize_point_text(a.get("text", "")): a for a in new_review.get("added") or []}

    events: list[dict] = []
    for t in sorted(new_removed - old_removed):
        events.append({"action": "remove", "target": t, "payload": {}})
    for t in sorted(old_added.keys() - new_added.keys()):
        events.append({"action": "remove", "target": t, "payload": {}})
    for t in sorted(old_removed - new_removed):
        events.append({"action": "restore", "target": t, "payload": {}})
    for t in sorted(new_added.keys() - old_added.keys()):
        events.append({"action": "add", "target": t, "payload": new_added[t]})

    return events


async def append_review_log(
    db,
    *,
    scholar_id: str,
    sentence_id: str,
    content_hash: str,
    action: str,
    target: str,
    payload: dict | None = None,
) -> str:
    """追加一条学习历史事件（**只 append，不 update、不 delete** —— 红线⑦）。

    Returns:
        `log_id`（`erv_<32hex>`）
    """
    log_id = "erv_" + uuid.uuid4().hex
    doc = {
        "log_id": log_id,
        "scholar_id": scholar_id,
        "sentence_id": sentence_id,
        "content_hash": content_hash,
        "action": action,
        "target": target,
        "payload": payload or {},
        "at": int(time.time() * 1000),
    }
    await db.insert(config.EXTENSION_REVIEW_LOG_COLLECTION, doc)
    return log_id


async def _load_review(db, scholar_id: str, sentence_id: str) -> dict | None:
    res = await db.query(
        config.EXTENSION_REVIEW_COLLECTION,
        where={"scholar_id": scholar_id, "sentence_id": sentence_id},
        limit=1,
    )
    records = res.get("records", [])
    return records[0] if records else None


async def save_review(
    db,
    *,
    scholar_id: str,
    sentence_id: str,
    content_hash: str,
    removed_texts: list[str],
    added: list[dict],
    note: str = "",
) -> dict:
    """E4'：提交「我的补丁」→ 全量替换 overlay → diff → 逐条 append 历史。

    出参（契约 §3.18 E4'）：`{ status, review, points, changes }`，
    `changes = { added, removed, restored }`。

    异常：`ExtensionError("EXTENSION_DISABLED", ...)` / `ExtensionError("INVALID_INPUT", ...)`
    """
    _gate()
    added = list(added or [])
    _validate(scholar_id, added)

    # 删除主键一律存归一化文本（删除比对也走归一化 —— 见 normalize_point_text）
    normalized_removed = [normalize_point_text(t) for t in (removed_texts or [])]
    normalized_removed = [t for t in normalized_removed if t]

    old = await _load_review(db, scholar_id, sentence_id)
    new_review_body = {
        "scholar_id": scholar_id,
        "sentence_id": sentence_id,
        "content_hash": content_hash,
        "removed_texts": normalized_removed,
        "added": added,
    }

    events = diff_review(old, new_review_body)
    for e in events:
        await append_review_log(
            db,
            scholar_id=scholar_id,
            sentence_id=sentence_id,
            content_hash=content_hash,
            action=e["action"],
            target=e["target"],
            payload=e["payload"],
        )

    now = int(time.time() * 1000)
    doc = {**new_review_body, "note": note or "", "updated_at": now}

    if old is not None:
        doc["created_at"] = old.get("created_at", now)
        await db.update(
            config.EXTENSION_REVIEW_COLLECTION,
            where={"_id": old["_id"]},
            data={"$set": doc},
            multi=False,
        )
    else:
        doc["created_at"] = now
        await db.insert(config.EXTENSION_REVIEW_COLLECTION, doc)

    # 读侧合并（不写回 AI 集合 —— §4.26 红线②）
    from services.english.extension import resolve_effective_point

    rec = await resolve_effective_point(db, sentence_id)
    ai_points = (rec or {}).get("points", []) or []
    points, summary = merge_review_points(ai_points, doc, content_hash)

    changes = {
        "added": sum(1 for e in events if e["action"] == "add"),
        "removed": sum(1 for e in events if e["action"] == "remove"),
        "restored": sum(1 for e in events if e["action"] == "restore"),
    }
    return {
        "status": "success",
        "review": {**doc, "stale": summary["stale"]},
        "points": points,
        "changes": changes,
    }


async def get_review(db, *, scholar_id: str, sentence_id: str) -> dict:
    """E6：我的校对 + 学习历史时间线（**只读**）。

    出参：`{ review: {...} | null, history: [{ log_id, action, target, payload, at }] }`，
    `history` 按 `at` 倒序、默认最近 50 条。
    """
    _gate()
    if not scholar_id or not str(scholar_id).strip():
        raise ExtensionError("INVALID_INPUT", "input", "scholar_id 不能为空")

    review = await _load_review(db, scholar_id, sentence_id)

    res = await db.query(
        config.EXTENSION_REVIEW_LOG_COLLECTION,
        where={"scholar_id": scholar_id, "sentence_id": sentence_id},
        order=[{"field": "at", "direction": "desc"}],
        limit=_HISTORY_LIMIT,
    )
    history = [
        {
            "log_id": r.get("log_id"),
            "action": r.get("action"),
            "target": r.get("target"),
            "payload": r.get("payload") or {},
            "at": r.get("at"),
        }
        for r in res.get("records", [])
    ]

    if review is None:
        return {"review": None, "history": history}

    # stale 以**当前 AI 记录的 content_hash** 为准（原句已更新 → 标注可能失效）
    from services.english.extension import resolve_effective_point

    rec = await resolve_effective_point(db, sentence_id)
    current_hash = (rec or {}).get("content_hash")
    stale = bool(current_hash) and current_hash != review.get("content_hash")

    return {
        "review": {
            "scholar_id": review.get("scholar_id"),
            "sentence_id": review.get("sentence_id"),
            "content_hash": review.get("content_hash"),
            "removed_texts": review.get("removed_texts") or [],
            "added": review.get("added") or [],
            "stale": stale,
            "updated_at": review.get("updated_at"),
        },
        "history": history,
    }
