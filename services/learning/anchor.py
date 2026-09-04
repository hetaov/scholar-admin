"""证据化锚点解析器（纯函数，无 DB/IO）— §7 阶段 1 产物

背景（docs_v1《沉浸式锚点与learning写入证据化重构方案》§5.5.2 + §10.3 修订 + §10.7 数据落地）：
  幽灵 learning 清洗（A/B/C 区）后 skill_state 只剩真实证据行，锚点无需再按百分比猜真伪。
  本解析器按「课级证据标签」做确定性锚点决策，供 L4 repair_book_anchor.py 校准存量锚点，
  与前端 L2 简单规则（§5.3）共用同一口径：
    零真实证据 → 第 1 课冷启动；有历史 → 信任锚点；最后真实课为进行中 → 续学；
    全部真实课已通关 → 完成态 / 推进边界（下一课）。

本模块只做决策，不采集证据。调用方职责：
  - lessons_sorted：教材课列表，按课序升序（课序口径同 scripts/repair_lesson_order.py：
    先显式 order 唯一则用之，否则按标题课号；与前端 sortChaptersByLessonNumber 对齐）；
  - tags：{lesson_id: "none" | "started" | "done"} 课级证据标签，由调用方按 §5.5.1
    证据模型标注：
      none    = 该课无任何真实证据（真实 = 结果性状态 / mastery≥阈值 / 带分 attempt / 长会话）；
      started = 有真实证据但课未通关（真实已开始 / 进行中）；
      done    = 课已通关（全部句子达结果态，读侧 progress=100 同义）。
  - stored：存量 scholar_book 锚点文档（可空），决策只读它、不信任其真实性。

锚点写回契约：current_chapter_id / current_lesson_id 同写为课 id（与小程序
  _enterNextLesson、repair_ghost_c2.py 归一化同一契约，兼容旧读取），current_group_id=null。

孤立尾部（§5.5.2 orphan_tail）保留语义，按 §10.3 数据现实收紧为「只针对存量脏锚点」：
  - 仅当存量锚点课本身就是孤立尾部（它是最后真实课、未通关，且与首个真实课之间存在
    顺序缺口）、且存在更早的真实进行中课时 → 回退第一个真实未通关课（幽灵推进把锚点
    钉到跳学课/末课的脏锚钉尾形态）；
  - 无更早进行中课（如全书仅一课有证据在末课）不回退，正常按存量锚点续学；
  - **无存量锚点时不做任何回退**，与读侧简单规则（§5.3：无锚点 → 最后一个进行中课）
    同一口径，避免把真实跳学/选择性练习误判成脏锚点。
"""
from __future__ import annotations

TAG_NONE = "none"
TAG_STARTED = "started"
TAG_DONE = "done"
_VALID_TAGS = frozenset({TAG_NONE, TAG_STARTED, TAG_DONE})

REASON_NEW_START = "new_start"
REASON_RESUME = "resume"
REASON_ORPHAN_TAIL = "orphan_tail"
REASON_FRONTIER = "frontier"
REASON_COMPLETED = "completed"


def lesson_key(lesson: dict) -> str:
    """课的主键（lesson_id 优先，兼容 _id 旧数据）。"""
    return str(lesson.get("lesson_id") or lesson.get("_id") or "")


def normalize_lesson_tags(lessons: list[dict], tags: dict | None) -> list[str]:
    """把调用方 tags（key=lesson_id）转成与 lessons 同序的标签数组；缺省/非法 → none。"""
    by_id = {lesson_key(l): l for l in lessons}
    tags = tags or {}
    out: list[str] = []
    for lesson in lessons:
        t = tags.get(lesson_key(lesson))
        out.append(t if t in _VALID_TAGS else TAG_NONE)
    return out


def _last_index(seq: list, pred) -> int | None:
    """从右向左找首个满足 pred 的下标；无则 None。"""
    for i in range(len(seq) - 1, -1, -1):
        if pred(seq[i]):
            return i
    return None


def resolve_book_anchor(
    *,
    lessons_sorted: list[dict],
    tags: dict | None = None,
    stored: dict | None = None,
) -> dict | None:
    """按课级证据标签解析教材锚点（确定性，幂等）。

    参数：
      lessons_sorted：课列表，按课序升序（必须非空且每课含 lesson_id/_id）；
      tags：{lesson_id: "none"|"started"|"done"}，缺省全部按 none；
      stored：存量锚点 {current_lesson_id?, current_chapter_id?, ...}（可空）。

    返回（无课教材返回 None）：
      {lesson_id, chapter_id(同课 id), group_id: None, order, title,
       reason, note, book_finished}。reason ∈ {new_start, resume, orphan_tail,
       frontier, completed}。
    """
    lessons = list(lessons_sorted or [])
    if not lessons:
        return None
    n = len(lessons)
    keys = [lesson_key(l) for l in lessons]
    index_of = {k: i for i, k in enumerate(keys)}
    tag_list = normalize_lesson_tags(lessons, tags)

    real_idx = [i for i, t in enumerate(tag_list) if t != TAG_NONE]

    def _mk(idx: int, reason: str, note: str, *, finished: bool = False) -> dict:
        lesson = lessons[idx]
        lid = keys[idx]
        return {
            "lesson_id": lid,
            "chapter_id": lid,  # 锚点契约：chapter/lesson 同写课 id（兼容旧读取）
            "group_id": None,
            "order": lesson.get("order"),
            "title": lesson.get("title", ""),
            "reason": reason,
            "note": note,
            "book_finished": finished,
        }

    # 1) 零真实证据 → 新开始（覆盖存量脏锚点）
    if not real_idx:
        return _mk(0, REASON_NEW_START, "全书无任何真实证据 → 第 1 课冷启动")

    first_real, last_real = real_idx[0], real_idx[-1]

    stored_lesson = (
        (stored or {}).get("current_lesson_id")
        or (stored or {}).get("current_chapter_id")
        or ""
    )
    stored_idx = index_of.get(stored_lesson) if stored_lesson else None

    def _is_isolated_tail(idx: int) -> bool:
        """idx 课是否为孤立尾部：最后真实课、未通关、且与首个真实课间有顺序缺口。"""
        if idx != last_real or tag_list[idx] != TAG_STARTED:
            return False
        if idx <= first_real:
            return False
        return any(t == TAG_NONE for t in tag_list[first_real + 1:idx])

    # 2) 孤立尾部回退（仅针对存量脏锚点钉尾）：stored 锚点课是孤立尾部、且存在更早的
    #    真实进行中课 → 回退第一个真实未通关课。
    if stored_idx is not None and _is_isolated_tail(stored_idx):
        earliest_started = next(
            (i for i in real_idx if tag_list[i] == TAG_STARTED), None
        )
        if earliest_started is not None and earliest_started != stored_idx:
            return _mk(
                earliest_started,
                REASON_ORPHAN_TAIL,
                "存量锚点为孤立尾部（顺序缺口后）→ 回退第一个真实未通关课",
            )

    # 3) 存量锚点信任：锚点课真实进行中（started）→ 采纳（断点续学不误伤）
    if stored_idx is not None and tag_list[stored_idx] == TAG_STARTED:
        return _mk(stored_idx, REASON_RESUME, "存量锚点课真实进行中 → 信任采纳（断点续学）")

    # 4) 按证据计算（与读侧简单规则同口径：无锚点 → 最后一个真实进行中课）
    last_started = _last_index(tag_list, lambda t: t == TAG_STARTED)
    if last_started is not None:
        return _mk(last_started, REASON_RESUME, "最后真实进行中课 → 续学")

    # 全部真实课均已通关（done）
    last_done = _last_index(tag_list, lambda t: t == TAG_DONE)
    if last_done is None:  # 理论不可达（real_idx 非空且无 started → 必有 done）
        return _mk(real_idx[-1], REASON_RESUME, "无通关态信息，按最后真实课续学")
    if last_done == n - 1:
        return _mk(n - 1, REASON_COMPLETED, "末课已通关 → 停留完成态", finished=True)
    return _mk(
        last_done + 1,
        REASON_FRONTIER,
        "已通关课的下一课（推进边界，与 _enterNextLesson 语义一致）",
    )
