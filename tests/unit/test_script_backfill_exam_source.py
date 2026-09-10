"""M13 迁移脚本 build_plan 纯函数单测（backfill_error_record_exam_source）。

覆盖：
- 缺 kp_source 的 EXTRA_AI 记录 → 进回填计划（kp_source='exam_paper'）；
- 已标记 / 正式教材链 / 无链记录不进计划（存量零改动）；
- 计划示例与计数口径（dry-run 可安全预览）。
"""
from __future__ import annotations

from scripts.backfill_error_record_exam_source import build_plan
from services.math.error_scanner import (
    EXTRA_AI_TEXTBOOK_ID,
    KP_SOURCE_EXAM_PAPER,
)


def _rec(record_id: str, *, textbook_id: str = "", kp_source: str = "") -> dict:
    doc = {"record_id": record_id, "textbook_id": textbook_id}
    if kp_source:
        doc["kp_source"] = kp_source
    return doc


def test_build_plan_only_extra_ai_missing_kp_source():
    records = [
        # EXTRA_AI 锚定但缺 kp_source → 进计划
        _rec("er_x1", textbook_id=EXTRA_AI_TEXTBOOK_ID),
        # EXTRA_AI 已标记 → 不进计划（计数 already）
        _rec("er_x2", textbook_id=EXTRA_AI_TEXTBOOK_ID, kp_source=KP_SOURCE_EXAM_PAPER),
        # 正式教材链 / 无链 → 一律不进计划
        _rec("er_tb", textbook_id="TB-3A"),
        _rec("er_unlinked"),
    ]
    to_fill, already = build_plan(records)
    assert to_fill == [("er_x1", KP_SOURCE_EXAM_PAPER)]
    assert already == 1


def test_build_plan_empty_and_all_marked():
    assert build_plan([]) == ([], 0)
    all_marked = [
        _rec("er_1", textbook_id=EXTRA_AI_TEXTBOOK_ID, kp_source=KP_SOURCE_EXAM_PAPER),
        _rec("er_2", textbook_id="TB-1", kp_source="textbook"),
    ]
    to_fill, already = build_plan(all_marked)
    assert to_fill == []
    assert already == 1  # 仅 EXTRA_AI 已标记计入
