"""单元测试：学者维度技能条统计 services/math/skill_progress（M11 任务②）

覆盖：
- mastery_avg 公式（单条首错 0.5 / 重复错压低 / 已巩固通过折半 / 组容量稀释）；
- weakness_signal 综合信号（重复错 occurrence / drill 未过 / 错因集中分布 / 正向对照）；
- record ↔ display_group join（kp 名映射、仅产出有错误证据的组、顺序确定、kp_ids 透传）；
- build_skill_progress db 入口（正式教材单元命中 / 老总结无 display_groups 兼容 / 空学者空数组）。
"""
from __future__ import annotations

import pytest

from services.database import CURRICULUM_NODE_COLLECTION
from services.math.skill_progress import (
    SKILL_PROGRESS_DOMINANT_TYPE_RATIO,
    SKILL_PROGRESS_REPEAT_OCCURRENCE_MIN,
    SKILL_PROGRESS_WEAK_MASTERY_MAX,
    aggregate_group_mastery,
    build_skill_progress,
    join_records_to_display_groups,
    record_drill_failed,
    record_drill_passed,
    record_occurrence,
    weakness_signal_for,
)
from tests.fakes.fake_db import FakeDB

UNIT_A_KEY = ("TB-3A", "第1单元 万以内的加法和减法")


def _record(**overrides) -> dict:
    doc = {
        "record_id": "er_1",
        "scholar_id": "s1",
        "knowledge_point_name": "整数加法",
        "primary_error": "computation",
        "source": "auto_scan",
        "created_at": 1000,
        "textbook_id": UNIT_A_KEY[0],
        "unit_title": UNIT_A_KEY[1],
        "occurrence": 1,
        "drill_stats": {},
        "last_drill_result": {},
    }
    doc.update(overrides)
    return doc


def _display_groups() -> list[dict]:
    return [
        {
            "display_group": "进位加法",
            "kp_names": ["整数加法", "整数减法"],
            "kp_ids": ["kp_jf", "kp_jt"],
            "count": 2,
        },
        {
            "display_group": "应用与建模",
            "kp_names": ["生活应用"],
            "kp_ids": ["kp_yy"],
            "count": 1,
        },
        {
            "display_group": "逻辑推理",
            "kp_names": ["图形推理"],
            "kp_ids": ["kp_tl"],
            "count": 1,
        },
    ]


def _unit_node(display_groups: list | None = None) -> dict:
    node = {
        "node_id": "unit_a",
        "node_type": "unit",
        "textbook_id": UNIT_A_KEY[0],
        "unit_title": UNIT_A_KEY[1],
        "title": UNIT_A_KEY[1],
    }
    if display_groups is not None:
        node["ai_summary"] = {
            "status": "success",
            "knowledge_points": [],
            "extended_points": [],
            "display_groups": display_groups,
        }
    return node


# ---------------------------------------------------------------------------
# 记录证据提取（存量缺省兼容）
# ---------------------------------------------------------------------------


def test_record_occurrence_defaults_to_one_for_legacy_and_clamps():
    assert record_occurrence(_record()) == 1
    assert record_occurrence(_record(occurrence=3)) == 3
    assert record_occurrence(_record(occurrence=0)) == 1
    assert record_occurrence(_record(occurrence="abc")) == 1
    assert record_occurrence(_record(occurrence=None)) == 1


def test_drill_evidence_passed_and_failed():
    passed = _record(drill_stats={"drill_count": 1, "pass_count": 1},
                     last_drill_result={"correct": True, "at": 5000})
    assert record_drill_passed(passed) is True
    assert record_drill_failed(passed) is False
    failed = _record(drill_stats={"drill_count": 2, "pass_count": 0},
                     last_drill_result={"correct": False, "at": 6000})
    assert record_drill_passed(failed) is False
    assert record_drill_failed(failed) is True
    legacy = _record()  # 存量：drill 字段全空 → 不算「巩固过」
    assert record_drill_passed(legacy) is False
    assert record_drill_failed(legacy) is False


# ---------------------------------------------------------------------------
# mastery_avg 公式
# ---------------------------------------------------------------------------


def test_mastery_single_fresh_error_is_0_5():
    """单条首错（occ=1 未巩固，组容量 1）→ 0.5（50% = 需要充能，非薄弱）"""
    assert aggregate_group_mastery([_record()], capacity=1) == 0.5


def test_mastery_repeated_error_lowers_to_0_25():
    """同组重复错（occ=3 未巩固）→ 0.25"""
    assert aggregate_group_mastery([_record(occurrence=3)], capacity=1) == 0.25


def test_mastery_drill_passed_halves_loss():
    """已巩固通过 → 当量折半：occ=1 通过 → 0.67（高于 0.5 未过）"""
    rec = _record(drill_stats={"drill_count": 1, "pass_count": 1},
                  last_drill_result={"correct": True})
    assert aggregate_group_mastery([rec], capacity=1) == 0.67


def test_mastery_larger_group_capacity_dilutes_loss():
    """组容量大（3 细点 1 错）→ 0.75；容量兜底 ≥1"""
    assert aggregate_group_mastery([_record()], capacity=3) == 0.75
    assert aggregate_group_mastery([_record()], capacity=0) == 0.5


# ---------------------------------------------------------------------------
# weakness_signal 综合判定
# ---------------------------------------------------------------------------


def test_weakness_signal_false_for_single_fresh_error():
    """正向对照：单条首错未重复未巩固 → 不标薄弱（0.5 非 <0.5，单条不成分布）"""
    mastery = aggregate_group_mastery([_record()], capacity=1)
    assert weakness_signal_for([_record()], mastery) is False


def test_weakness_signal_repeat_occurrence_triggers():
    """重复错（occ ≥ 2）→ 薄弱（即使 mastery ≥ 阈值也命中）"""
    rec = _record(occurrence=SKILL_PROGRESS_REPEAT_OCCURRENCE_MIN)
    assert weakness_signal_for([rec], 0.6) is True


def test_weakness_signal_drill_attempted_not_passed_triggers():
    """巩固过但未通过 → 薄弱"""
    rec = _record(drill_stats={"drill_count": 1, "pass_count": 0},
                  last_drill_result={"correct": False})
    assert weakness_signal_for([rec], 0.5) is True


def test_weakness_signal_dominant_error_type_triggers():
    """同错因 ≥2 条且占比 ≥0.5 → 错因分布薄弱"""
    recs = [
        _record(knowledge_point_name="整数加法", primary_error="method"),
        _record(knowledge_point_name="整数加法", primary_error="method"),
    ]
    assert weakness_signal_for(recs, 0.6) is True


def test_weakness_signal_low_mastery_triggers():
    """mastery_avg < 阈值 → 薄弱"""
    assert weakness_signal_for([_record()], SKILL_PROGRESS_WEAK_MASTERY_MAX - 0.01) is True


# ---------------------------------------------------------------------------
# record ↔ display_group join
# ---------------------------------------------------------------------------


def test_join_only_emits_groups_with_evidence_and_preserves_order():
    """仅「学者有错题」的技能条入列；组序保 display_groups 原始顺序"""
    records = [
        _record(knowledge_point_name="整数加法", occurrence=2),   # 进位加法组
        _record(knowledge_point_name="生活应用"),                 # 应用与建模组
        _record(knowledge_point_name="整数减法"),                 # 也属进位加法组
    ]
    unit_nodes = {UNIT_A_KEY: _unit_node(_display_groups())}
    items = join_records_to_display_groups(records, unit_nodes)
    # 「逻辑推理」无错误证据 → 不入列
    assert [it["display_group"] for it in items] == ["进位加法", "应用与建模"]
    first = items[0]
    assert first["kp_ids"] == ["kp_jf", "kp_jt"]  # kp_ids 直取技能条
    # 进位加法组：2 条记录 occ 2+1，容量 2 → loss=3 → mastery=1-3/5=0.4
    assert first["mastery_avg"] == 0.4
    assert first["weakness_signal"] is True  # 重复错 occ=2
    second = items[1]
    assert second["display_group"] == "应用与建模"
    assert second["mastery_avg"] == 0.5  # 单条首错
    assert second["weakness_signal"] is False


def test_join_skips_unlinked_extra_ai_and_nameless_legacy_records():
    """未归链 / EXTRA_AI / 无 kp 名（存量回退 node_code）→ 不参与聚合"""
    records = [
        _record(textbook_id="", unit_title=""),
        _record(textbook_id="EXTRA_AI", unit_title="课外补充"),
        _record(textbook_id=UNIT_A_KEY[0], unit_title=UNIT_A_KEY[1],
                knowledge_point_name=""),
    ]
    unit_nodes = {UNIT_A_KEY: _unit_node(_display_groups())}
    assert join_records_to_display_groups(records, unit_nodes) == []


def test_join_no_unit_summary_or_no_display_groups_returns_empty():
    """老总结无 display_groups / 单元节点缺失 → []（存量兼容）"""
    records = [_record(knowledge_point_name="整数加法")]
    legacy_node = _unit_node(display_groups=None)  # ai_summary 无 display_groups
    assert join_records_to_display_groups(records, {UNIT_A_KEY: legacy_node}) == []
    assert join_records_to_display_groups(records, {}) == []


# ---------------------------------------------------------------------------
# build_skill_progress db 入口
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_skill_progress_hits_formal_textbook_unit():
    """db 入口：正式教材单元命中 display_groups → 产出聚合"""
    db = FakeDB()
    db.add("error_record", _record(knowledge_point_name="整数加法", occurrence=2))
    db.add("error_record", _record(knowledge_point_name="生活应用"))
    db.add(CURRICULUM_NODE_COLLECTION, _unit_node(_display_groups()))
    items = await build_skill_progress(db, db.all("error_record"))
    assert [it["display_group"] for it in items] == ["进位加法", "应用与建模"]
    # 进位加法组：1 条记录 occ2（重复错）容量 2 → loss2 → 1-2/4=0.5
    assert items[0]["mastery_avg"] == 0.5
    assert items[0]["weakness_signal"] is True  # 重复错 occ=2


@pytest.mark.asyncio
async def test_build_skill_progress_empty_for_legacy_and_empty_scholar():
    """学者无记录 / 单元无 display_groups → [] 兼容存量"""
    db = FakeDB()
    assert await build_skill_progress(db, []) == []
    db.add("error_record", _record(knowledge_point_name="整数加法"))
    # 教材内没有任何已总结单元节点
    assert await build_skill_progress(db, db.all("error_record")) == []
    # 单元节点存在但为老总结（无 display_groups）
    db2 = FakeDB()
    db2.add("error_record", _record(knowledge_point_name="整数加法"))
    db2.add(CURRICULUM_NODE_COLLECTION, _unit_node(display_groups=None))
    assert await build_skill_progress(db2, db2.all("error_record")) == []
