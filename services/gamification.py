"""P2 扩展功能服务层 — 每日目标 / 徽章 / 排行榜 / 错题本

对应提案 `docs_v2/03-change/proposals/2026-08-16-P2-后续扩展功能完整设计.md`（2026-08-17 定稿）。

集合：
- `badge`         ：徽章定义（管理端可维护，`enabled` 控制是否启用）
- `scholar_badge` ：学者已获得徽章，复合键 `{scholar_id}_{badge_code}`（幂等发放）
- `scholars`      ：学者档案（排行榜昵称来源，`_id` = scholar_id、`name` = 昵称）

设计原则（与后端既有服务一致）：
- 一次学者级查询 + 内容 `$in` 批量加载，避免 N+1；
- 全部纯服务函数，不持有 FastAPI 依赖，路由层负责入参校验；
- 无数据回落空态（`success:true`），不抛业务错误。
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone

from services.events import (
    ATTEMPT_STATUS_INCORRECT,
    ERROR_TYPE_FALLBACK,
    VALID_ERROR_TYPES,
    STUDY_ATTEMPT,
)
from services.models_learning import (
    SKILL_STATE,
    STATUS_LEARNED,
    STATUS_MASTERED,
)
from services.models_content import (
    SENTENCE_V2,
    get_sentences_by_ids,
    query_all_pages,
)

logger = logging.getLogger("scholar-admin.gamification")

# ---------------------------------------------------------------------------
# 集合名（顶层常量，供 check_schema.py 扫描）
# ---------------------------------------------------------------------------

BADGE = "badge"
SCHOLAR_BADGE = "scholar_badge"
SCHOLARS = "scholars"

# 徽章条件类型
CONDITION_LEARNED_COUNT = "learned_count"  # 已学句子数（learned/mastered 去重）
CONDITION_STUDY_MINUTES = "study_minutes"  # 累计学习分钟数
CONDITION_ATTEMPT_COUNT = "attempt_count"  # 累计学习次数
CONDITION_STREAK_DAYS = "streak_days"      # 连续打卡天数
CONDITION_WRONG_CLEARED = "wrong_cleared"  # 错题清零数（同句既有 incorrect 又有 correct）

VALID_CONDITIONS = {
    CONDITION_LEARNED_COUNT,
    CONDITION_STUDY_MINUTES,
    CONDITION_ATTEMPT_COUNT,
    CONDITION_STREAK_DAYS,
    CONDITION_WRONG_CLEARED,
}

# ---------------------------------------------------------------------------
# F6 每日目标 — 规则引擎
# ---------------------------------------------------------------------------

# 目标上下限（缺失 7 天记录时回落 floor；达标过冲时封顶 cap）
DAILY_GOAL_FLOOR = {"new_sentences": 3, "minutes": 5, "attempts": 10}
DAILY_GOAL_CAP = {"new_sentences": 15, "minutes": 30, "attempts": 50}
DAILY_GOAL_GROWTH = 1.2  # goal = clamp(round(avg7 × 1.2), floor, cap)
DAILY_GOAL_WINDOW_DAYS = 7  # 近 7 天（不含今日）


def _day_bounds(date_obj: datetime.date) -> tuple[int, int]:
    """某自然日（UTC）的 [start_ts, end_ts] 秒级窗口。"""
    start_ts = int(datetime.combine(date_obj, time.min, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.combine(date_obj, time.max, tzinfo=timezone.utc).timestamp())
    return start_ts, end_ts


async def _daily_new_sentence_ids(db, scholar_id: str, start_ts: int, end_ts: int) -> set[str]:
    """今日新学句数：当日 study_attempt 中出现、且 skill_state 达到 learned/mastered 的去重句子。

    一次学者级事件查询 + 一次 $in 状态查询，避免逐句 N+1。
    """
    attempts = await query_all_pages(
        db,
        collection=STUDY_ATTEMPT,
        where={"scholar_id": scholar_id, "created_at": {"$gte": start_ts, "$lte": end_ts}},
        select={"sentence_id": 1},
    )
    sids = {a.get("sentence_id") for a in attempts if a.get("sentence_id")}
    if not sids:
        return set()
    learned: set[str] = set()
    for i in range(0, len(sids), 200):
        batch = list(sids)[i:i + 200]
        states = await query_all_pages(
            db,
            collection=SKILL_STATE,
            where={
                "scholar_id": scholar_id,
                "sentence_id": {"$in": batch},
                "status": {"$in": [STATUS_LEARNED, STATUS_MASTERED]},
            },
            select={"sentence_id": 1},
        )
        learned |= {s.get("sentence_id") for s in states if s.get("sentence_id")}
    return learned & sids


async def _daily_stats(db, scholar_id: str, day: datetime.date) -> dict:
    """某自然日（UTC）的 {new_sentences, minutes, attempts} 口径统计。"""
    start_ts, end_ts = _day_bounds(day)
    attempts = await query_all_pages(
        db,
        collection=STUDY_ATTEMPT,
        where={"scholar_id": scholar_id, "created_at": {"$gte": start_ts, "$lte": end_ts}},
        select={"sentence_id": 1, "time_spent": 1},
    )
    total_seconds = sum(int(a.get("time_spent") or 0) for a in attempts)
    new_ids = await _daily_new_sentence_ids(db, scholar_id, start_ts, end_ts)
    return {
        "new_sentences": len(new_ids),
        "minutes": round(total_seconds / 60),
        "attempts": len(attempts),
    }


def _clamp_goal(value: float, key: str) -> int:
    lo = DAILY_GOAL_FLOOR[key]
    hi = DAILY_GOAL_CAP[key]
    return max(lo, min(hi, round(value)))


async def build_daily_goal(db, *, scholar_id: str, date_str: str | None = None) -> dict:
    """F6 每日学习目标（契约 §3.7 POST /tracking/daily-goal 的服务口径）。

    返回：`{ date, goal, progress, completed, percent }`。
    - goal 生成：`clamp(round(avg7 × 1.2), floor, cap)`；近 7 天无记录 → floor；
    - progress：今日实时统计；percent = 三指标达标率均值；completed = percent ≥ 100。
    """
    if not date_str:
        today = datetime.now(timezone.utc).date()
    else:
        today = datetime.strptime(date_str, "%Y-%m-%d").date()
    date_str = today.strftime("%Y-%m-%d")

    # 1. 近 7 天（不含今日）均值 → 目标
    avg = {"new_sentences": 0.0, "minutes": 0.0, "attempts": 0.0}
    window_days = DAILY_GOAL_WINDOW_DAYS
    for i in range(1, window_days + 1):
        day = today - timedelta(days=i)
        stats = await _daily_stats(db, scholar_id, day)
        avg["new_sentences"] += stats["new_sentences"] / window_days
        avg["minutes"] += stats["minutes"] / window_days
        avg["attempts"] += stats["attempts"] / window_days

    # 2. 目标：近 7 天无任何记录 → 直接 floor
    has_history = any(avg[k] > 0 for k in avg)
    if has_history:
        goal = {k: _clamp_goal(avg[k] * DAILY_GOAL_GROWTH, k) for k in avg}
    else:
        goal = dict(DAILY_GOAL_FLOOR)

    # 3. 今日进度
    progress = await _daily_stats(db, scholar_id, today)

    # 4. 达标率：分母为 0 的项跳过
    ratios = []
    for k in goal:
        if goal[k] > 0:
            ratios.append(min(1.0, progress[k] / goal[k]))
    percent = round(sum(ratios) / len(ratios) * 100) if ratios else 0
    completed = percent >= 100

    logger.info(
        f"[daily-goal] scholar={scholar_id}, date={date_str}, "
        f"goal={goal}, progress={progress}, percent={percent}"
    )
    return {
        "date": date_str,
        "goal": goal,
        "progress": progress,
        "completed": completed,
        "percent": percent,
    }


# ---------------------------------------------------------------------------
# F8 徽章 — 判定 + 幂等发放
# ---------------------------------------------------------------------------


def _iso(ts: int | None) -> str | None:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


async def _aggregate_badge_metrics(db, scholar_id: str) -> dict:
    """一次学者级数据聚合徽章所需指标（减少重复查询）。"""
    states = await query_all_pages(
        db,
        collection=SKILL_STATE,
        where={"scholar_id": scholar_id},
        select={"sentence_id": 1, "status": 1},
    )
    learned_sids: set[str] = set()
    for s in states:
        if s.get("status") in (STATUS_LEARNED, STATUS_MASTERED) and s.get("sentence_id"):
            learned_sids.add(s["sentence_id"])

    attempts = await query_all_pages(
        db,
        collection=STUDY_ATTEMPT,
        where={"scholar_id": scholar_id},
        select={"sentence_id": 1, "status": 1, "time_spent": 1, "created_at": 1},
    )
    total_seconds = 0
    incorrect_sids: set[str] = set()
    correct_sids: set[str] = set()
    day_counts: dict[str, int] = {}
    for a in attempts:
        total_seconds += int(a.get("time_spent") or 0)
        status = a.get("status")
        sid = a.get("sentence_id")
        if status == ATTEMPT_STATUS_INCORRECT and sid:
            incorrect_sids.add(sid)
        elif status == "correct" and sid:
            correct_sids.add(sid)
        ts = a.get("created_at")
        if ts:
            try:
                day = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
                day_counts[day] = day_counts.get(day, 0) + 1
            except (TypeError, ValueError):
                continue

    # 连续打卡：从今天起往前连续 ≥1 条 attempt；今天无记录 → 0（与 calendar 一致）
    streak_days = 0
    today = datetime.now(timezone.utc).date()
    if day_counts.get(today.strftime("%Y-%m-%d"), 0) > 0:
        cur = today
        while day_counts.get(cur.strftime("%Y-%m-%d"), 0) > 0:
            streak_days += 1
            cur -= timedelta(days=1)

    return {
        CONDITION_LEARNED_COUNT: len(learned_sids),
        CONDITION_STUDY_MINUTES: round(total_seconds / 60),
        CONDITION_ATTEMPT_COUNT: len(attempts),
        CONDITION_STREAK_DAYS: streak_days,
        CONDITION_WRONG_CLEARED: len(incorrect_sids & correct_sids),
    }


async def evaluate_badges(db, *, scholar_id: str) -> dict:
    """F8 徽章墙（契约 §3.7 GET /tracking/{id}/badges 的服务口径）。

    按 `badge.condition_type` 聚合 current；`current ≥ target_value` 且未发放 →
    幂等插入 `scholar_badge`（复合键 `{scholar_id}_{badge_code}`，记录 `first_awarded_at`）。
    返回 `{ earned: [...], locked: [...] }`。
    """
    badges = await query_all_pages(
        db,
        collection=BADGE,
        where={"enabled": True},
    )
    if not badges:
        return {"earned": [], "locked": []}

    metrics = await _aggregate_badge_metrics(db, scholar_id)

    # 已发放集合
    granted_rows = await query_all_pages(
        db,
        collection=SCHOLAR_BADGE,
        where={"scholar_id": scholar_id},
        select={"badge_code": 1, "awarded_at": 1},
    )
    granted: dict[str, int] = {g.get("badge_code"): g.get("awarded_at") for g in granted_rows}

    now = int(datetime.now(timezone.utc).timestamp())
    earned: list[dict] = []
    locked: list[dict] = []
    for b in badges:
        code = b.get("badge_code", "")
        cond = b.get("condition_type", "")
        if cond not in VALID_CONDITIONS:
            continue
        current = int(metrics.get(cond, 0) or 0)
        target = int(b.get("target_value") or 0)
        base = {
            "badge_code": code,
            "name": b.get("name", ""),
            "icon": b.get("icon", ""),
            "description": b.get("description", ""),
        }
        if current >= target and code not in granted:
            await db.update(
                collection=SCHOLAR_BADGE,
                where={"scholar_id": scholar_id, "badge_code": code},
                data={
                    "$set": {
                        "scholar_id": scholar_id,
                        "badge_code": code,
                        "awarded_at": now,
                        "first_awarded_at": now,
                        "updated_at": now,
                    }
                },
                multi=False,
                upsert=True,
            )
            granted[code] = now
        if code in granted:
            earned.append({**base, "awarded_at": _iso(granted[code])})
        else:
            locked.append({**base, "progress": {"current": current, "target": target}})

    logger.info(
        f"[badges] scholar={scholar_id}, earned={len(earned)}, locked={len(locked)}"
    )
    return {"earned": earned, "locked": locked}


# ---------------------------------------------------------------------------
# F8 排行榜
# ---------------------------------------------------------------------------

PERIOD_DAYS = {"week": 7, "month": 30, "all": None}
MAX_LEADERBOARD_LIMIT = 50


async def build_leaderboard(
    db,
    *,
    period: str = "week",
    metric: str = "minutes",
    limit: int = 10,
    scholar_id: str | None = None,
) -> dict:
    """F8 排行榜（契约 §3.7 GET /tracking/leaderboard 的服务口径）。

    按周期过滤 `study_attempt.created_at` 窗口 → 按 scholar_id 分组聚合
    （`minutes` = time_spent 求和秒→分钟；`sentences` = 去重句子数）→ 内存排序取 TopN。
    `name` 批量加载自 `scholars` 集合；不返回 openid。
    """
    limit = max(1, min(int(limit), MAX_LEADERBOARD_LIMIT))
    days = PERIOD_DAYS.get(period, 7)
    where: dict = {}
    if days is not None:
        start_ts = int(
            datetime.combine(
                datetime.now(timezone.utc).date() - timedelta(days=days),
                time.min,
                tzinfo=timezone.utc,
            ).timestamp()
        )
        where["created_at"] = {"$gte": start_ts}

    attempts = await query_all_pages(
        db,
        collection=STUDY_ATTEMPT,
        where=where,
        select={"scholar_id": 1, "sentence_id": 1, "time_spent": 1},
    )

    # 按学者分组聚合
    totals: dict[str, dict] = {}
    for a in attempts:
        sid = a.get("scholar_id")
        if not sid:
            continue
        bucket = totals.setdefault(sid, {"minutes": 0, "sentences": set(), "seconds": 0})
        bucket["seconds"] += int(a.get("time_spent") or 0)
        if a.get("sentence_id"):
            bucket["sentences"].add(a["sentence_id"])

    rows = []
    for sid, bucket in totals.items():
        rows.append({
            "scholar_id": sid,
            "minutes": round(bucket["seconds"] / 60),
            "sentences": len(bucket["sentences"]),
        })

    rows.sort(key=lambda r: r.get(metric, 0), reverse=True)
    top = rows[:limit]

    # 昵称批量加载（scholars._id = scholar_id, name = 昵称）
    names: dict[str, str] = {}
    ids = [r["scholar_id"] for r in top]
    for i in range(0, len(ids), 200):
        batch = ids[i:i + 200]
        recs = await query_all_pages(
            db,
            collection=SCHOLARS,
            where={"_id": {"$in": batch}},
            select={"_id": 1, "name": 1},
        )
        for r in recs:
            names[r.get("_id")] = r.get("name") or ""

    items = [
        {
            "rank": idx + 1,
            "scholar_id": r["scholar_id"],
            "name": names.get(r["scholar_id"], ""),
            "value": r.get(metric, 0),
            "is_me": bool(scholar_id) and r["scholar_id"] == scholar_id,
        }
        for idx, r in enumerate(top)
    ]

    my_rank: int | None = None
    if scholar_id:
        for idx, r in enumerate(rows):
            if r["scholar_id"] == scholar_id:
                my_rank = idx + 1
                break

    logger.info(
        f"[leaderboard] period={period}, metric={metric}, limit={limit}, "
        f"items={len(items)}, my_rank={my_rank}"
    )
    return {
        "period": period,
        "metric": metric,
        "items": items,
        "my_rank": my_rank,
    }


# ---------------------------------------------------------------------------
# F11 错题本
# ---------------------------------------------------------------------------

MAX_WRONG_BOOK_LIMIT = 200
DEFAULT_WRONG_BOOK_LIMIT = 50


async def aggregate_wrong_book(
    db,
    *,
    scholar_id: str,
    limit: int = DEFAULT_WRONG_BOOK_LIMIT,
    error_type: str | None = None,
) -> dict:
    """F11 错题本（契约 §3.7 GET /tracking/{id}/wrong-book 的服务口径）。

    一次学者级 `study_attempt`（status=incorrect）查询 → 按句子内存分组聚合：
    - `error_count`：错误次数（传 `error_type` 时仅统计该类型）；
    - `last_error_at`：最近一次错误时间（ISO）；
    - `error_types`：全量错误类型分布（缺省回落 `other`）；
    内容 `$in` 批量加载 `sentence_v2`，按 `last_error_at` 降序。
    """
    limit = max(1, min(int(limit), MAX_WRONG_BOOK_LIMIT))
    attempts = await query_all_pages(
        db,
        collection=STUDY_ATTEMPT,
        where={"scholar_id": scholar_id, "status": ATTEMPT_STATUS_INCORRECT},
        select={
            "sentence_id": 1,
            "lesson_id": 1,
            "skill_code": 1,
            "error_type": 1,
            "created_at": 1,
        },
    )

    # 按句子聚合
    by_sentence: dict[str, dict] = {}
    for a in attempts:
        sid = a.get("sentence_id")
        if not sid:
            continue
        bucket = by_sentence.setdefault(sid, {
            "sentence_id": sid,
            "lesson_id": a.get("lesson_id"),
            "skill_code": a.get("skill_code"),
            "error_count": 0,
            "last_error_at": 0,
            "error_types": {},
        })
        et = a.get("error_type") or ERROR_TYPE_FALLBACK
        if et not in VALID_ERROR_TYPES:
            et = ERROR_TYPE_FALLBACK
        bucket["error_types"][et] = bucket["error_types"].get(et, 0) + 1
        if not error_type or et == error_type:
            bucket["error_count"] += 1
        ts = a.get("created_at")
        if ts:
            try:
                ts = int(ts)
                if ts > int(bucket["last_error_at"]):
                    bucket["last_error_at"] = ts
            except (TypeError, ValueError):
                continue

    if not by_sentence:
        return {"total": 0, "items": []}

    # 内容批量加载
    content_by_id: dict[str, dict] = {}
    ids = list(by_sentence.keys())
    for i in range(0, len(ids), 200):
        sentences = await get_sentences_by_ids(db, ids[i:i + 200])
        for s in sentences:
            content_by_id[s.get("sentence_id")] = s

    items = []
    for sid, bucket in by_sentence.items():
        content = content_by_id.get(sid, {})
        items.append({
            "sentence_id": sid,
            "content": content.get("text", ""),
            "translation": content.get("translation", ""),
            "lesson_id": bucket.get("lesson_id") or content.get("lesson_id"),
            "chapter_id": content.get("chapter_id"),
            "skill_code": bucket.get("skill_code") or "",
            "error_count": bucket["error_count"],
            "last_error_at": _iso(bucket["last_error_at"]),
            "error_types": [
                {"type": t, "count": c}
                for t, c in sorted(bucket["error_types"].items(), key=lambda kv: -kv[1])
            ],
        })

    items.sort(key=lambda it: it["last_error_at"] or "", reverse=True)
    items = items[:limit]

    logger.info(
        f"[wrong-book] scholar={scholar_id}, total={len(by_sentence)}, "
        f"returned={len(items)}, error_type={error_type}"
    )
    return {"total": len(by_sentence), "items": items}
