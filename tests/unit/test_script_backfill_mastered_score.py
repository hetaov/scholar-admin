"""存量 skill_state.mastery_score 回填脚本 build_plan 纯函数单测（短板消灭战 D5/P1）。

覆盖：
- status=mastered 且 mastery_score 缺失/< FLOOR → 进计划，抬到 FLOOR，并同步重算 progress/next_review_at；
- 已达标 / 非 mastered → 不进计划（存量零改动，I3）；
- --scholar 灰度过滤；
- 幂等：对回填后记录再跑一次 → 0 变更；
- 缺主键记录 → skipped 计数。
"""
from __future__ import annotations

import argparse
import asyncio

from scripts import backfill_mastered_score as mod
from scripts.backfill_mastered_score import build_plan
from services.models_learning import (
    MASTERED_SCORE_FLOOR,
    SKILL_STATE,
    compute_next_review_at,
    derive_progress,
)
from tests.fakes.fake_db import FakeDB

NOW = 1_700_000_000


def _rec(
    state_id: str | None,
    *,
    scholar_id: str = "s1",
    status: str = "mastered",
    mastery_score=None,
    last_studied_at: int = 1000,
    attempt_count: int = 1,
) -> dict:
    doc = {
        "scholar_id": scholar_id,
        "status": status,
        "last_studied_at": last_studied_at,
        "attempt_count": attempt_count,
    }
    if state_id is not None:
        doc["state_id"] = state_id
    if mastery_score is not None:
        doc["mastery_score"] = mastery_score
    return doc


def test_mastered_low_score_goes_into_plan():
    records = [_rec("s1_x_translation", mastery_score=40.0)]
    to_fill, already, skipped = build_plan(records, now=NOW)
    assert already == 0
    assert skipped == 0
    assert len(to_fill) == 1
    state_id, fields = to_fill[0]
    assert state_id == "s1_x_translation"
    assert fields["mastery_score"] == float(MASTERED_SCORE_FLOOR)
    assert fields["progress"] == derive_progress("mastered", MASTERED_SCORE_FLOOR)
    assert fields["next_review_at"] == compute_next_review_at(
        1000, 1, float(MASTERED_SCORE_FLOOR)
    )
    assert fields["updated_at"] == NOW


def test_mastered_missing_score_goes_into_plan():
    to_fill, already, skipped = build_plan([_rec("s1_y_speaking")], now=NOW)
    assert [sid for sid, _ in to_fill] == ["s1_y_speaking"]
    assert to_fill[0][1]["mastery_score"] == float(MASTERED_SCORE_FLOOR)


def test_already_above_floor_no_change():
    """已达标（>=FLOOR）不进计划 → 幂等；重跑 0 变更。"""
    records = [
        _rec("s1_a", mastery_score=80.0),
        _rec("s1_b", mastery_score=100.0),
    ]
    to_fill, already, skipped = build_plan(records, now=NOW)
    assert to_fill == []
    assert already == 2
    assert skipped == 0


def test_non_mastered_untouched():
    """I3：learning / learned 一律不进计划。"""
    records = [
        _rec("s1_l", status="learning", mastery_score=30.0),
        _rec("s1_d", status="learned", mastery_score=65.0),
        _rec("s1_m", status="mastered", mastery_score=30.0),
    ]
    to_fill, already, skipped = build_plan(records, now=NOW)
    assert [sid for sid, _ in to_fill] == ["s1_m"]


def test_scholar_grayscale_filter():
    records = [
        _rec("s1_a", scholar_id="s1", mastery_score=30.0),
        _rec("s2_a", scholar_id="s2", mastery_score=30.0),
    ]
    to_fill, _, _ = build_plan(records, scholars=["s2"], now=NOW)
    assert [sid for sid, _ in to_fill] == ["s2_a"]


def test_missing_primary_key_skipped():
    records = [_rec(None, mastery_score=30.0)]
    to_fill, already, skipped = build_plan(records, now=NOW)
    assert to_fill == []
    assert skipped == 1


def test_idempotent_rerun_via_plan_output():
    """用首轮 $set 结果构造「已回填」记录，再跑一次 → 0 变更（幂等）。"""
    first, _, _ = build_plan([_rec("s1_x", mastery_score=40.0)], now=NOW)
    _, fields = first[0]
    replayed = _rec("s1_x", mastery_score=fields["mastery_score"])
    second, already, _ = build_plan([replayed], now=NOW)
    assert second == []
    assert already == 1


# ---------------------------------------------------------------------------
# DB 层接线：run() 的查询 / 写入 / dry-run / 幂等 / 灰度 / 分页（FakeDB）
# ---------------------------------------------------------------------------


def _db_doc(
    state_id: str,
    *,
    scholar_id: str = "s1",
    status: str = "mastered",
    mastery_score=None,
    last_studied_at: int = 1000,
    attempt_count: int = 1,
) -> dict:
    doc = {
        "state_id": state_id,
        "scholar_id": scholar_id,
        "status": status,
        "last_studied_at": last_studied_at,
        "attempt_count": attempt_count,
    }
    if mastery_score is not None:
        doc["mastery_score"] = mastery_score
    return doc


def _patch_db(monkeypatch, db: FakeDB) -> None:
    monkeypatch.setattr(mod, "CloudBaseNoSQLClient", lambda: db)


def _args(*, commit: bool, scholar=None) -> argparse.Namespace:
    return argparse.Namespace(commit=commit, scholar=scholar)


def test_run_dry_run_does_not_write(monkeypatch):
    db = FakeDB()
    db.add(SKILL_STATE, _db_doc("s1_x", mastery_score=40.0))
    _patch_db(monkeypatch, db)
    code = asyncio.run(mod.run(_args(commit=False)))
    assert code == 0
    assert db.all(SKILL_STATE)[0]["mastery_score"] == 40.0  # dry-run 零写入


def test_run_commit_raises_mastered_only(monkeypatch):
    db = FakeDB()
    db.add(SKILL_STATE, _db_doc("s1_x", mastery_score=40.0))
    db.add(SKILL_STATE, _db_doc("s1_y", status="learning", mastery_score=30.0))
    _patch_db(monkeypatch, db)
    code = asyncio.run(mod.run(_args(commit=True)))
    assert code == 0
    docs = {d["state_id"]: d for d in db.all(SKILL_STATE)}
    assert docs["s1_x"]["mastery_score"] == float(MASTERED_SCORE_FLOOR)
    assert docs["s1_x"]["progress"] == derive_progress("mastered", MASTERED_SCORE_FLOOR)
    assert docs["s1_y"]["mastery_score"] == 30.0  # I3：非 mastered 未动


def test_run_commit_is_idempotent(monkeypatch):
    db = FakeDB()
    db.add(SKILL_STATE, _db_doc("s1_x", mastery_score=40.0))
    calls = {"modified": 0}
    orig_update = db.update

    async def counting_update(collection, where, data, upsert=False, multi=True):
        res = await orig_update(collection, where, data, upsert=upsert, multi=multi)
        calls["modified"] += res.get("modified_count", 0)
        return res

    monkeypatch.setattr(db, "update", counting_update)
    _patch_db(monkeypatch, db)

    asyncio.run(mod.run(_args(commit=True)))
    assert calls["modified"] == 1

    calls["modified"] = 0
    asyncio.run(mod.run(_args(commit=True)))
    assert calls["modified"] == 0  # 重跑幂等：0 变更


def test_run_scholar_grayscale(monkeypatch):
    db = FakeDB()
    db.add(SKILL_STATE, _db_doc("s1_x", scholar_id="s1", mastery_score=40.0))
    db.add(SKILL_STATE, _db_doc("s2_x", scholar_id="s2", mastery_score=40.0))
    _patch_db(monkeypatch, db)
    asyncio.run(mod.run(_args(commit=True, scholar=["s2"])))
    docs = {d["state_id"]: d for d in db.all(SKILL_STATE)}
    assert docs["s1_x"]["mastery_score"] == 40.0  # 非灰度学者零改动
    assert docs["s2_x"]["mastery_score"] == float(MASTERED_SCORE_FLOOR)


def test_run_paginated_fetch(monkeypatch):
    db = FakeDB()
    for i in range(5):
        db.add(SKILL_STATE, _db_doc(f"s1_{i}", mastery_score=40.0))
    monkeypatch.setattr(mod, "PAGE", 2)  # 强制多页拉取
    _patch_db(monkeypatch, db)
    asyncio.run(mod.run(_args(commit=True)))
    assert all(
        d["mastery_score"] == float(MASTERED_SCORE_FLOOR)
        for d in db.all(SKILL_STATE)
    )
