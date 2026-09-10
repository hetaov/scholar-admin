"""学者维度技能条统计（M11 任务② · 方案 §3.3 / §4.3）

契约（本次登记，进 M12 前落 docs_v2）：
- skill_progress[] 挂在 GET /math/error-stats 出参 data.skill_progress（缺省空数组兼容存量）；
- 每条：{ display_group, kp_ids[], mastery_avg, weakness_signal }：
  - display_group：单元技能条名（取自 unit 级 ai_summary.display_groups[].display_group）；
  - kp_ids[]：该技能条下细点 source_node_id 集合（display_groups[].kp_ids 直取）；
  - mastery_avg：掌握度 0.0~1.0（保留两位小数），1 = 无负向证据；
  - weakness_signal：bool，薄弱信号 = error_type 分布 + 重复错 occurrence + drill 未过 综合。

口径（确定性、可测、缺省兼容存量）：
1. 学者错题记录按 (textbook_id, unit_title) 归到单元；EXTRA_AI / 未归链
   （textbook_id 为空或 EXTRA_AI）不参与（单元技能条仅在正式教材链上）。
2. 记录须有 knowledge_point_name 才能映射技能条（display_groups[].kp_names 匹配）；
   存量记录无该字段 / 单元尚无 display_groups（老总结）→ 该单元贡献为空。
3. 每条 display_group 只在该学者有 ≥1 条可映射错题时才产出（无错误证据 → 不入列）；
   无任何可映射数据时整体返回 []（前端 M12 走缺省降级）。
4. mastery_avg：
   - 每条记录的负向当量 loss = occurrence（≥1，缺省 1；重复错次数累加）；
   - 已巩固通过（drill_stats.pass_count ≥ 1 或 last_drill_result.correct=true）
     视为掌握过半，当量折半（一次通过抵消一半错题当量，仍保留 0.5 弱残留证据）；
   - 组容量 capacity = 该技能条成员细点数（kp_names 长度，兜底 1）；
   - mastery = 1 - Σloss / (capacity + Σloss)，clamp [0,1] 四舍五入两位。
5. weakness_signal（任一命中即 True）：
   - mastery_avg < SKILL_PROGRESS_WEAK_MASTERY_MAX（0.5，即单条首错 0.5 不算薄弱，
     与 M3 能量文案 50% = 「需要充能」对齐）；
   - 存在 occurrence ≥ SKILL_PROGRESS_REPEAT_OCCURRENCE_MIN（2）的重复错；
   - 存在「巩固过但未通过」（drill 尝试 ≥1 且 pass_count=0 且 last correct≠true）；
   - 同错因（primary_error）记录 ≥2 条且占比 ≥ SKILL_PROGRESS_DOMINANT_TYPE_RATIO（0.5）
     → 错因分布形成集中趋势。

与既有错题统计一致：聚合数据源 = error-stats 同一批 scholar 记录（items 同源），
故 items 截断不影响列表页/技能条统计一致性（沿用 MVP 口径，超大错题量后续再单独扫描）。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from services.database import CURRICULUM_NODE_COLLECTION

# ---------------------------------------------------------------------------
# 常量（weakness 综合阈值，进 M12 契约登记）
# ---------------------------------------------------------------------------

# 薄弱掌握度上限：mastery_avg < 0.5 → 薄弱（M3 能量口径 50% = 「需要充能」，0.5 单独首错不标薄弱）
SKILL_PROGRESS_WEAK_MASTERY_MAX = 0.5
# 重复错阈值：occurrence ≥ 2 → 连续错/反复错
SKILL_PROGRESS_REPEAT_OCCURRENCE_MIN = 2
# 错因集中占比：单一 primary_error 记录 ≥2 条且占比 ≥0.5 → 错因分布信号
SKILL_PROGRESS_DOMINANT_TYPE_RATIO = 0.5
# 已巩固通过的当量折半系数（一次通过抵消一半错题当量）
SKILL_PROGRESS_PASSED_LOSS_FACTOR = 0.5


# ---------------------------------------------------------------------------
# 单条记录证据提取（缺省兼容存量：occurrence 缺失 / drill 字段为空）
# ---------------------------------------------------------------------------


def record_occurrence(record: dict) -> int:
    """记录错题次数：occurrence ≥1 整数化；缺省/非法按 1（存量一次错）"""
    n = record.get("occurrence")
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 1
    return n if n >= 1 else 1


def _drill_stats_of(record: dict) -> dict:
    d = record.get("drill_stats")
    return d if isinstance(d, dict) else {}


def _last_drill_result_of(record: dict) -> dict:
    d = record.get("last_drill_result")
    return d if isinstance(d, dict) else {}


def record_drill_attempted(record: dict) -> bool:
    """是否做过巩固（drill_count ≥1 或 last_drill_result 有内容）"""
    stats = _drill_stats_of(record)
    try:
        drill_count = int(stats.get("drill_count") or 0)
    except (TypeError, ValueError):
        drill_count = 0
    return drill_count >= 1 or bool(_last_drill_result_of(record))


def record_drill_passed(record: dict) -> bool:
    """是否已巩固通过：drill_stats.pass_count ≥1 或 last_drill_result.correct=true"""
    stats = _drill_stats_of(record)
    try:
        pass_count = int(stats.get("pass_count") or 0)
    except (TypeError, ValueError):
        pass_count = 0
    last = _last_drill_result_of(record)
    return pass_count >= 1 or last.get("correct") is True


def record_drill_failed(record: dict) -> bool:
    """做过巩固但未通过（drill 尝试后仍未 pass_count）"""
    return record_drill_attempted(record) and not record_drill_passed(record)


# ---------------------------------------------------------------------------
# 技能条组聚合（纯函数，可单测）
# ---------------------------------------------------------------------------


def group_capacity(display_group: dict) -> int:
    """组容量 = 技能条成员细点数（kp_names 长度；兜底 1 防除零）"""
    names = display_group.get("kp_names") or []
    return len(names) if names else 1


def aggregate_group_mastery(records: list[dict], capacity: int) -> float:
    """组掌握度：1 - Σloss / (capacity + Σloss)，clamp [0,1] 四舍五入两位。

    loss = occurrence（缺省 1）；已巩固通过折半（SKILL_PROGRESS_PASSED_LOSS_FACTOR）。
    """
    total_loss = 0.0
    for rec in records or []:
        occ = record_occurrence(rec)
        loss = occ * (
            SKILL_PROGRESS_PASSED_LOSS_FACTOR if record_drill_passed(rec) else 1.0
        )
        total_loss += loss
    denom = max(1, capacity) + total_loss
    mastery = 1.0 - total_loss / denom
    return round(max(0.0, min(1.0, mastery)), 2)


def _dominant_error_type_ratio(records: list[dict]) -> float:
    """组内最高频错因占比：≥2 条同错因才有意义，否则 0（单条首错不构成分布信号）"""
    counts: dict[str, int] = defaultdict(int)
    for rec in records or []:
        etype = (rec.get("primary_error") or "").strip()
        if etype:
            counts[etype] += 1
    if not counts:
        return 0.0
    dominant = max(counts.values())
    if dominant < 2:
        return 0.0
    return dominant / sum(counts.values())


def weakness_signal_for(records: list[dict], mastery_avg: float) -> bool:
    """薄弱信号综合判定（error_type 分布 + occurrence + drill 未过 + 掌握度）"""
    recs = records or []
    if mastery_avg < SKILL_PROGRESS_WEAK_MASTERY_MAX:
        return True
    if any(record_occurrence(r) >= SKILL_PROGRESS_REPEAT_OCCURRENCE_MIN for r in recs):
        return True
    if any(record_drill_failed(r) for r in recs):
        return True
    if _dominant_error_type_ratio(recs) >= SKILL_PROGRESS_DOMINANT_TYPE_RATIO:
        return True
    return False


# ---------------------------------------------------------------------------
# 记录 ↔ display_group join（纯函数，可单测）
# ---------------------------------------------------------------------------


def _unit_key(record: dict) -> tuple[str, str]:
    """记录所属单元键：(textbook_id, unit_title)"""
    return (record.get("textbook_id") or "", record.get("unit_title") or "")


def _is_formal_textbook_key(key: tuple[str, str]) -> bool:
    """正式教材链才参与技能条：textbook_id 非空且非 EXTRA_AI"""
    textbook_id, unit_title = key
    if not textbook_id or textbook_id == "EXTRA_AI":
        return False
    return bool(unit_title)


def join_records_to_display_groups(
    records: list[dict],
    unit_nodes: dict[tuple[str, str], dict],
) -> list[dict]:
    """把学者错题映射到单元技能条，产出 skill_progress[]（确定性顺序）。

    入参：
      records     - error_record 原始列表（含 textbook_id/unit_title/knowledge_point_name/
                    occurrence/drill_stats/last_drill_result/primary_error）
      unit_nodes  - {(textbook_id, unit_title): unit 节点}（节点带 ai_summary.display_groups）
    出参：
      [{display_group, kp_ids[], mastery_avg, weakness_signal}]，
      仅含「有 ≥1 条可映射错题」的技能条；无映射 → []（存量兼容）。
    """
    # 1) 按单元键分组（只留正式教材链 + 有 knowledge_point_name 的记录）
    records_by_unit: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records or []:
        key = _unit_key(rec)
        if not _is_formal_textbook_key(key):
            continue
        if not (rec.get("knowledge_point_name") or "").strip():
            continue  # 无 kp 名（存量回退 node_code）无法映射技能条，跳过
        records_by_unit[key].append(rec)

    # 2) 逐单元：kp 名 → display_group 映射（先名字→组，保 F1 组序）
    out: list[dict] = []
    for key in sorted(records_by_unit):
        unit_node = unit_nodes.get(key)
        if not unit_node:
            continue
        display_groups = (unit_node.get("ai_summary") or {}).get("display_groups") or []
        if not display_groups:
            continue  # 老总结无 display_groups → 该单元贡献空（存量兼容）
        name_to_group: dict[str, dict] = {}
        for g in display_groups:
            for name in g.get("kp_names") or []:
                name_to_group.setdefault(str(name), g)

        records_by_group: dict[str, list[dict]] = defaultdict(list)
        for rec in records_by_unit[key]:
            g = name_to_group.get(str(rec.get("knowledge_point_name") or ""))
            if g is not None:
                records_by_group[g.get("display_group") or ""].append(rec)

        # 3) 仅产出有错误证据的组（保 display_groups 原始顺序 → 确定性）
        for g in display_groups:
            label = g.get("display_group") or ""
            recs = records_by_group.get(label)
            if not recs:
                continue
            mastery = aggregate_group_mastery(recs, group_capacity(g))
            out.append(
                {
                    "display_group": label,
                    "kp_ids": list(g.get("kp_ids") or []),
                    "mastery_avg": mastery,
                    "weakness_signal": weakness_signal_for(recs, mastery),
                }
            )
    return out


# ---------------------------------------------------------------------------
# DB 加载入口（路由调用：GET /math/error-stats 侧新增响应字段）
# ---------------------------------------------------------------------------


async def build_skill_progress(db, records: list[dict]) -> list[dict]:
    """加载学者错题对应单元技能条并聚合 skill_progress[]。

    仅查询出现过的正式教材单元（textbook_id+unit_title 去重），无节点/无
    display_groups 的单元自动跳过；无任何可映射数据 → []。
    """
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for rec in records or []:
        key = _unit_key(rec)
        if not _is_formal_textbook_key(key):
            continue
        if not (rec.get("knowledge_point_name") or "").strip():
            continue
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)

    unit_nodes: dict[tuple[str, str], dict] = {}
    for textbook_id, unit_title in keys:
        res = await db.query(
            CURRICULUM_NODE_COLLECTION,
            where={
                "node_type": "unit",
                "textbook_id": textbook_id,
                "unit_title": unit_title,
            },
            limit=1,
        )
        nodes = (res or {}).get("records") or []
        if nodes:
            unit_nodes[(textbook_id, unit_title)] = nodes[0]

    return join_records_to_display_groups(records, unit_nodes)


# 便捷类型标注（供路由/测试引用）
SkillProgressItem = dict[str, Any]
