"""学习追踪统计 — learning_mastery_tracking 统计服务（纯函数，便于单元测试）

统计维度：
- 用户级别学习时间：record_list 中 time_spent 之和（单位：秒）
- sentence 学习进度：每条句子的掌握情况（score / mastery / status → 0~1）
- unit 学习进度：单元内已学句子数 / 单元总句子数
- textbook 学习进度：整本教材已学句子数 / 教材总句子数
"""

from __future__ import annotations

import math
from typing import Any

# 判定为“已学/已掌握”的状态（支持中英文）
_LEARNED_STATUSES = {
    "learned",
    "mastered",
    "complete",
    "completed",
    "done",
    "已学",
    "已学会",
    "已学完",
    "已完成",
    "已掌握",
    "掌握",
}

# 判定为“未学”的状态
_UNLEARNED_STATUSES = {
    "new",
    "todo",
    "not_started",
    "unlearned",
    "未学",
    "未开始",
    "未学习",
}

# 及格分数（0-100）与掌握阈值（0-1）
_DEFAULT_PASS_SCORE = 60.0
_DEFAULT_MASTERY_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def parse_time_spent(value: Any, default: float = 0.0) -> float:
    """把 time_spent 等数值字段解析为 float。

    支持 int / float / 数字字符串（含小数、负数处理），
    无效值（None、空串、非数字、NaN、负值）返回 default。
    """
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result) or result < 0:
        return default
    return result


def is_learned(record: dict) -> bool:
    """根据 status / score / mastery 判断该条记录对应的句子是否已学。

    判定优先级：
    1. status 命中已学/未学关键字 → 直接返回
    2. score 给出 → score >= 60 视为已学（给出但不及格视为未学）
    3. mastery 给出 → mastery >= 0.6 视为已学
    4. 均未给出 → 未学
    """
    status = str(record.get("status") or "").strip().lower()
    if status:
        if status in _LEARNED_STATUSES:
            return True
        if status in _UNLEARNED_STATUSES:
            return False

    score = parse_time_spent(record.get("score"), default=float("nan"))
    if not math.isnan(score):
        return score >= _DEFAULT_PASS_SCORE

    mastery = parse_time_spent(record.get("mastery"), default=float("nan"))
    if not math.isnan(mastery):
        return mastery >= _DEFAULT_MASTERY_THRESHOLD

    return False


def sentence_progress(record: dict) -> float:
    """计算单条记录对应句子的学习进度 0~1。

    优先级：score/100 > mastery > 是否已学（已学=1，未学=0）。
    """
    score = parse_time_spent(record.get("score"), default=float("nan"))
    if not math.isnan(score):
        return max(0.0, min(1.0, score / 100.0))

    mastery = parse_time_spent(record.get("mastery"), default=float("nan"))
    if not math.isnan(mastery):
        return max(0.0, min(1.0, mastery))

    return 1.0 if is_learned(record) else 0.0


def merge_records(record_list: list[dict]) -> dict[str, dict]:
    """将同一 sentence_id 的多条记录合并为一条。

    合并规则：
    - time_spent 累加（同一句子的多次学习时长合计）
    - score / mastery 取最大值
    - status 任一记录为已学状态则视为已学
    - 忽略缺少 sentence_id 或非 dict 的脏数据

    Returns:
        {sentence_id: {sentence_id, time_spent, score, mastery, status, ...}}
    """
    merged: dict[str, dict] = {}
    for rec in record_list:
        if not isinstance(rec, dict):
            continue
        sent_id = str(rec.get("sentence_id") or "").strip()
        if not sent_id:
            continue

        item = merged.setdefault(
            sent_id,
            {
                "sentence_id": sent_id,
                "time_spent": 0.0,
                "score": None,
                "mastery": None,
                "status": "",
                "competency_id": rec.get("competency_id", ""),
            },
        )
        item["time_spent"] += parse_time_spent(rec.get("time_spent"))

        score = parse_time_spent(rec.get("score"), default=float("nan"))
        if not math.isnan(score) and (item["score"] is None or score > item["score"]):
            item["score"] = score

        mastery = parse_time_spent(rec.get("mastery"), default=float("nan"))
        if not math.isnan(mastery) and (item["mastery"] is None or mastery > item["mastery"]):
            item["mastery"] = mastery

        status = str(rec.get("status") or "").strip()
        if status and status.lower() in _LEARNED_STATUSES:
            item["status"] = status
        elif status and not item["status"]:
            item["status"] = status
    return merged


def format_duration(seconds: float) -> str:
    """把秒数格式化为可读字符串，如 3725 → "1小时2分5秒"、"45分"、"30秒"。"""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}小时{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


# ---------------------------------------------------------------------------
# 主统计函数
# ---------------------------------------------------------------------------


def compute_tracking_stats(
    scholar_id: str,
    text_book_id: str,
    record_list: list[dict],
    sentences: list[dict],
    units: list[dict],
) -> dict:
    """统计用户学习时间与 textbook / unit / sentence 三级学习进度。

    Args:
        scholar_id: 学者 ID
        text_book_id: 教材 ID
        record_list: 客户端上报的学习记录列表，每条记录至少含 sentence_id，
            可含 time_spent（秒）、status、score（0-100）、mastery（0-1）
        sentences: 该教材下全部句子，来自 sentence 集合
            （含 sentence_id / unit_id / index / text 字段）
        units: 该教材下全部单元，来自 unit 集合
            （含 unit_id / title 字段）

    Returns:
        统计结果 dict，包含 summary / units / sentences 三个层级。
    """
    # 1. 建立索引
    unit_title_map: dict[str, str] = {}
    for u in units:
        if isinstance(u, dict) and u.get("unit_id"):
            unit_title_map[str(u["unit_id"])] = str(u.get("title") or "")

    sentence_index: dict[str, dict] = {}
    for s in sentences:
        if isinstance(s, dict) and s.get("sentence_id"):
            sentence_index[str(s["sentence_id"])] = s

    # 2. 合并记录（同一句子多条记录 → 一条）
    merged = merge_records(record_list)

    # 3. 句子级统计
    learned = 0
    total_time = 0.0
    unit_sentence_map: dict[str, list[dict]] = {}
    sentence_stats: list[dict] = []

    # 保持数据库顺序（unit 分组 + index 升序）
    ordered = sorted(
        sentence_index.values(),
        key=lambda s: (str(s.get("unit_id") or ""), int(s.get("index") or 0)),
    )
    for s in ordered:
        sent_id = str(s["sentence_id"])
        unit_id = str(s.get("unit_id") or "")
        rec = merged.get(sent_id, {})
        learned_flag = is_learned(rec) if rec else False
        progress = sentence_progress(rec) if rec else 0.0
        spent = rec.get("time_spent", 0.0) if rec else 0.0

        total_time += spent
        if learned_flag:
            learned += 1

        unit_sentence_map.setdefault(unit_id, []).append(
            {
                "sentence_id": sent_id,
                "unit_id": unit_id,
                "index": int(s.get("index") or 0),
                "text": str(s.get("text") or ""),
                "learned": learned_flag,
                "status": rec.get("status", "") if rec else "",
                "score": rec.get("score") if rec else None,
                "time_spent": round(spent, 1),
                "progress": round(progress, 4),
            }
        )

        sentence_stats.append(
            {
                "sentence_id": sent_id,
                "unit_id": unit_id,
                "index": int(s.get("index") or 0),
                "text": str(s.get("text") or ""),
                "learned": learned_flag,
                "status": rec.get("status", "") if rec else "",
                "score": rec.get("score") if rec else None,
                "time_spent": round(spent, 1),
                "progress": round(progress, 4),
            }
        )

    # 4. 单元级统计
    unit_stats: list[dict] = []
    for unit_id, items in unit_sentence_map.items():
        total_in_unit = len(items)
        learned_in_unit = sum(1 for it in items if it["learned"])
        time_in_unit = sum(it["time_spent"] for it in items)
        unit_stats.append(
            {
                "unit_id": unit_id,
                "unit_title": unit_title_map.get(unit_id, ""),
                "total_sentence_count": total_in_unit,
                "learned_sentence_count": learned_in_unit,
                "progress": round(learned_in_unit / total_in_unit, 4) if total_in_unit else 0.0,
                "time_spent": round(time_in_unit, 1),
            }
        )
    unit_stats.sort(key=lambda u: u["unit_id"])

    total_sentence_count = len(sentence_index)
    completed_units = sum(1 for u in unit_stats if u["progress"] >= 1.0)
    active_units = sum(1 for u in unit_stats if u["learned_sentence_count"] > 0)
    avg_unit_progress = (
        sum(u["progress"] for u in unit_stats) / len(unit_stats) if unit_stats else 0.0
    )

    # 5. 汇总
    summary = {
        "scholar_id": scholar_id,
        "text_book_id": text_book_id,
        "record_count": len(record_list),
        "matched_record_count": sum(1 for r in record_list if isinstance(r, dict) and str(r.get("sentence_id") or "").strip() in sentence_index),
        "total_time_spent": round(total_time, 1),
        "total_time_spent_display": format_duration(total_time),
        "total_sentence_count": total_sentence_count,
        "learned_sentence_count": learned,
        "textbook_progress": round(learned / total_sentence_count, 4) if total_sentence_count else 0.0,
        "unit_count": len(unit_stats),
        "learned_unit_count": active_units,
        "completed_unit_count": completed_units,
        "avg_unit_progress": round(avg_unit_progress, 4),
    }

    return {
        "scholar_id": scholar_id,
        "text_book_id": text_book_id,
        "summary": summary,
        "units": unit_stats,
        "sentences": sentence_stats,
    }
