"""存量 speech_evaluation.parsed.completion 量纲回填脚本单测（2026-09-22 缺陷修复配套）。

覆盖：
- 旧口径（0~1）→ 进计划，补 ×100 的 0~100 值；
- 已新口径 / 本就为 0 → 不进计划（幂等：重跑 0 变更）；
- `parsed` 缺 completion（None）→ 进计划补值；
- `expected < stored`（本缺陷解释不了的形态）→ **不动**，计入 anomalies；
- `raw` 缺失/非 dict/JSON 文本 三种形态的判定与兜底；
- `parsed` 非 dict / 缺 `_id` → skipped；
- `--limit` 金丝雀截断；
- DB 层接线：dry-run 零写入、commit 只改 completion、其余 parsed 字段与 raw 原样保留、重跑幂等。
"""
from __future__ import annotations

import argparse
import asyncio
import json

from scripts import backfill_speech_completion as mod
from scripts.backfill_speech_completion import build_plan
from services.providers.speech_eval import SPEECH_EVALUATION_COLLECTION
from tests.fakes.fake_db import FakeDB

# 真机/定标口径的原始响应：PronCompletion 官方量纲 0~1
RAW_CLEAN = {"PronAccuracy": 98.0, "PronFluency": 0.98, "PronCompletion": 1.0, "SuggestedScore": 97.0}
RAW_PARTIAL = {"PronAccuracy": 40.0, "PronFluency": 0.9, "PronCompletion": 0.444, "SuggestedScore": 35.0}


def _doc(doc_id, *, completion, raw=RAW_CLEAN, parsed_extra=None):
    parsed = {"accuracy": 90.0, "fluency": 98.0, "suggested_score": 97.0, "words": []}
    if completion is not None:
        parsed["completion"] = completion
    parsed.update(parsed_extra or {})
    return {"_id": doc_id, "parsed": parsed, "raw": raw}


# ---------------------------------------------------------------------------
# 纯函数 build_plan
# ---------------------------------------------------------------------------


def test_old_scale_goes_into_plan():
    """旧口径 1.0（实指 100 分）→ 回填为 100.0。"""
    to_fill, already, skipped, anomalies = build_plan([_doc("d1", completion=1.0)])
    assert to_fill == [("d1", 1.0, 100.0)]
    assert (already, skipped, anomalies) == (0, 0, 0)


def test_partial_completion_scaled():
    """0.444（旧口径，实指 44.4）→ 44.4。"""
    to_fill, *_ = build_plan([_doc("d2", completion=0.444, raw=RAW_PARTIAL)])
    assert to_fill == [("d2", 0.444, 44.4)]


def test_new_scale_is_noop():
    """新口径已是 0~100 → 不进计划（幂等）。"""
    to_fill, already, skipped, anomalies = build_plan([_doc("d3", completion=100.0)])
    assert to_fill == []
    assert (already, skipped, anomalies) == (1, 0, 0)


def test_zero_stays_zero():
    """无有效发音（0~1 口径的 0.0）→ expected 也是 0.0 → 不进计划，不误判。"""
    to_fill, already, _, _ = build_plan([_doc("d4", completion=0.0, raw={"PronCompletion": 0.0})])
    assert to_fill == []
    assert already == 1


def test_missing_completion_gets_filled():
    """parsed 缺 completion → 补值（stored=None 走 expected > -1 分支）。"""
    to_fill, *_ = build_plan([_doc("d5", completion=None)])
    assert to_fill == [("d5", None, 100.0)]


def test_expected_lower_than_stored_is_anomaly_untouched():
    """expected < stored（本缺陷解释不了）→ 不动，计入 anomalies（留人工判断）。"""
    to_fill, already, skipped, anomalies = build_plan(
        [_doc("d6", completion=500.0, raw={"PronCompletion": 1.0})]
    )
    assert to_fill == []
    assert (already, skipped, anomalies) == (0, 0, 1)


def test_unfixable_shapes_skipped():
    """raw 缺失 / 非 dict / parsed 非 dict / 缺 _id → skipped，不猜。"""
    records = [
        {"_id": "x1", "parsed": {"completion": 1.0}},                    # 无 raw
        {"_id": "x2", "parsed": {"completion": 1.0}, "raw": None},
        {"_id": "x3", "parsed": {"completion": 1.0}, "raw": "not-json"},
        {"_id": "x4", "parsed": "oops", "raw": RAW_CLEAN},                # parsed 非 dict
        {"parsed": {"completion": 1.0}, "raw": RAW_CLEAN},                # 缺 _id
    ]
    to_fill, already, skipped, anomalies = build_plan(records)
    assert to_fill == []
    assert (already, skipped, anomalies) == (0, 5, 0)


def test_raw_as_json_text_is_parsed():
    """存量 raw 若被存成 JSON 文本 → 解析后照常判定（不静默跳过）。"""
    to_fill, *_ = build_plan([_doc("d7", completion=1.0, raw=json.dumps(RAW_CLEAN))])
    assert to_fill == [("d7", 1.0, 100.0)]


def test_limit_truncates_plan():
    records = [_doc(f"d{i}", completion=1.0) for i in range(5)]
    to_fill, *_ = build_plan(records, limit=2)
    assert [sid for sid, _, _ in to_fill] == ["d0", "d1"]


def test_idempotent_rerun_via_plan_output():
    """用首轮结果构造「已回填」记录，再跑一次 → 0 变更（幂等）。"""
    first, *_ = build_plan([_doc("d8", completion=0.444, raw=RAW_PARTIAL)])
    _, _stored, new_completion = first[0]
    second, already, _, _ = build_plan([_doc("d8", completion=new_completion, raw=RAW_PARTIAL)])
    assert second == []
    assert already == 1


# ---------------------------------------------------------------------------
# DB 层接线（FakeDB）
# ---------------------------------------------------------------------------


def _patch_db(monkeypatch, db: FakeDB) -> None:
    monkeypatch.setattr(mod, "CloudBaseNoSQLClient", lambda: db)


def _args(*, commit: bool, limit=None) -> argparse.Namespace:
    return argparse.Namespace(commit=commit, limit=limit)


def test_run_dry_run_does_not_write(monkeypatch):
    db = FakeDB()
    db.add(SPEECH_EVALUATION_COLLECTION, _doc("d1", completion=1.0))
    _patch_db(monkeypatch, db)
    code = asyncio.run(mod.run(_args(commit=False)))
    assert code == 0
    assert db.all(SPEECH_EVALUATION_COLLECTION)[0]["parsed"]["completion"] == 1.0  # dry-run 零写入


def test_run_commit_fixes_completion_only(monkeypatch):
    """只改 completion 一个字段：其余 parsed 字段与 raw 原样保留（红线）。"""
    db = FakeDB()
    db.add(
        SPEECH_EVALUATION_COLLECTION,
        _doc("d1", completion=1.0, parsed_extra={"accuracy": 12.5, "words": [{"word": "the", "match_tag": 0}]}),
    )
    _patch_db(monkeypatch, db)
    code = asyncio.run(mod.run(_args(commit=True)))
    assert code == 0
    doc = db.all(SPEECH_EVALUATION_COLLECTION)[0]
    assert doc["parsed"]["completion"] == 100.0
    assert doc["parsed"]["accuracy"] == 12.5  # 未动
    assert doc["parsed"]["words"] == [{"word": "the", "match_tag": 0}]  # 未动
    assert doc["raw"] == RAW_CLEAN  # raw 存档原样（定标/复核依据）
    assert "updated_at" not in doc  # 不扩 schema（§4.9 未定义）


def test_run_commit_is_idempotent(monkeypatch):
    db = FakeDB()
    db.add(SPEECH_EVALUATION_COLLECTION, _doc("d1", completion=0.444, raw=RAW_PARTIAL))
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


def test_run_commit_skips_anomalies(monkeypatch):
    """anomaly 记录不得被写入。"""
    db = FakeDB()
    db.add(SPEECH_EVALUATION_COLLECTION, _doc("a1", completion=500.0, raw={"PronCompletion": 1.0}))
    _patch_db(monkeypatch, db)
    asyncio.run(mod.run(_args(commit=True)))
    assert db.all(SPEECH_EVALUATION_COLLECTION)[0]["parsed"]["completion"] == 500.0


def test_run_limit_canary(monkeypatch):
    """--limit 只写前 N 条（灰度）。"""
    db = FakeDB()
    for i in range(4):
        db.add(SPEECH_EVALUATION_COLLECTION, _doc(f"d{i}", completion=1.0))
    _patch_db(monkeypatch, db)
    asyncio.run(mod.run(_args(commit=True, limit=2)))
    docs = {d["_id"]: d["parsed"]["completion"] for d in db.all(SPEECH_EVALUATION_COLLECTION)}
    assert docs == {"d0": 100.0, "d1": 100.0, "d2": 1.0, "d3": 1.0}


def test_run_paginated_fetch(monkeypatch):
    db = FakeDB()
    for i in range(5):
        db.add(SPEECH_EVALUATION_COLLECTION, _doc(f"d{i}", completion=1.0))
    monkeypatch.setattr(mod, "PAGE", 2)  # 强制多页拉取
    _patch_db(monkeypatch, db)
    asyncio.run(mod.run(_args(commit=True)))
    assert all(
        d["parsed"]["completion"] == 100.0
        for d in db.all(SPEECH_EVALUATION_COLLECTION)
    )
