"""单元测试：学习者校对 overlay（2026-10-03 修订 2 · service-contract §8.7）

覆盖 B30（归一化 / 读侧合并纯函数）；B35 会在此文件续补 diff 三类事件 / 超限 /
**R9 零 mastery 写入**断言（契约要求 R9 与 R1 **同文件放置**的断言在
`test_extension_evaluate.py`，见 B35）。

红线对照：
- §4.26 红线②：合并**只发生在读侧**，不得反向写回 `english_extension_point`；
- §3.18 红线⑦：`content_hash` 变更只标 `stale`，**不自动删除**学习者标注。
"""
from __future__ import annotations

import asyncio

import pytest

from tests.fakes.fake_db import FakeDB

import config
from services.english.extension_review import (
    _MINE_TEXT_MAX_LEN,
    append_review_log,
    diff_review,
    get_review,
    merge_review_points,
    normalize_point_text,
    save_review,
)
from services.providers.extension_llm import ExtensionError


def _run(coro):
    return asyncio.run(coro)


def _ai_point(text: str, i: int = 0) -> dict:
    return {
        "id": f"s1#{i}",
        "type": "word",
        "text": text,
        "meaning_zh": "释义",
        "l1": {"stem": f"____ {text}", "options": [text, "a", "b", "c"]},
    }


# ---------------------------------------------------------------------------
# normalize_point_text
# ---------------------------------------------------------------------------


def test_normalize_lowercase():
    assert normalize_point_text("Take Part In") == "take part in"


def test_normalize_strips_edge_punctuation():
    assert normalize_point_text("take part in.") == "take part in"
    assert normalize_point_text("...take part in!") == "take part in"
    # 中文/全角标点同样只去首尾
    assert normalize_point_text("参加，") == "参加"


def test_normalize_collapses_whitespace():
    assert normalize_point_text("take   part\n\tin") == "take part in"


def test_normalize_empty():
    assert normalize_point_text("") == ""
    assert normalize_point_text(None) == ""


def test_normalize_keeps_inner_punctuation():
    """内部标点必须保留：它是语言点文本的一部分（如 look-up / mother-in-law）。"""
    assert normalize_point_text("mother-in-law") == "mother-in-law"


# ---------------------------------------------------------------------------
# merge_review_points — 无 overlay
# ---------------------------------------------------------------------------


def test_merge_without_review_returns_pure_ai():
    ai = [_ai_point("coat", 0), _ai_point("shirt", 1)]
    points, summary = merge_review_points(ai, None, "h1")

    assert [p["text"] for p in points] == ["coat", "shirt"]
    assert all(p["origin"] == "ai" for p in points)
    # l1 题面必须原样保留
    assert points[0]["l1"]["stem"] == "____ coat"
    assert summary == {
        "has_review": False,
        "added_count": 0,
        "removed_count": 0,
        "stale": False,
        "updated_at": None,
    }


def test_merge_does_not_mutate_input():
    """纯函数：不得就地修改入参（AI 点来自缓存记录，改了会污染集合）。"""
    ai = [_ai_point("coat", 0)]
    merge_review_points(ai, {"content_hash": "h1", "removed_texts": ["coat"], "added": []}, "h1")
    assert "origin" not in ai[0]
    assert len(ai) == 1


# ---------------------------------------------------------------------------
# merge_review_points — 剔除 / 追加 / 顺序
# ---------------------------------------------------------------------------


def test_merge_removes_by_normalized_text():
    ai = [_ai_point("Coat", 0), _ai_point("shirt", 1)]
    review = {"content_hash": "h1", "removed_texts": ["coat"], "added": [], "updated_at": 100}
    points, summary = merge_review_points(ai, review, "h1")

    # 大小写差异仍命中（归一化比对，非原始字符串）
    assert [p["text"] for p in points] == ["shirt"]
    assert summary["removed_count"] == 1
    assert summary["has_review"] is True
    assert summary["stale"] is False


def test_merge_removes_by_punctuation_insensitive_match():
    ai = [_ai_point("coat.", 0)]
    review = {"content_hash": "h1", "removed_texts": ["coat"], "added": []}
    points, _ = merge_review_points(ai, review, "h1")
    assert points == []


def test_merge_appends_mine_points_after_ai():
    ai = [_ai_point("coat", 0)]
    review = {
        "content_hash": "h1",
        "removed_texts": [],
        "added": [{"id": "mine_0", "type": "word", "text": "wardrobe", "meaning_zh": "衣柜"}],
        "updated_at": 200,
    }
    points, summary = merge_review_points(ai, review, "h1")

    assert [p["origin"] for p in points] == ["ai", "mine"]
    mine = points[1]
    assert mine["id"] == "mine_0"
    assert mine["text"] == "wardrobe"
    # D2：自建点恒无 L1 题面（无后端答案键，不进判分）
    assert mine["l1"] is None
    assert summary["added_count"] == 1
    assert summary["updated_at"] == 200


def test_merge_mine_point_defaults_id_and_type():
    review = {
        "content_hash": "h1",
        "removed_texts": [],
        "added": [{"text": "wardrobe"}],
    }
    points, _ = merge_review_points([], review, "h1")
    assert points[0]["id"] == "mine_0"
    assert points[0]["type"] == ""
    assert points[0]["meaning_zh"] == ""
    assert points[0]["note"] == ""
    assert points[0]["origin"] == "mine"


# ---------------------------------------------------------------------------
# merge_review_points — stale（原句已更新）
# ---------------------------------------------------------------------------


def test_merge_stale_does_not_apply_patch():
    ai = [_ai_point("coat", 0), _ai_point("shirt", 1)]
    review = {
        "content_hash": "old_hash",
        "removed_texts": ["coat"],
        "added": [{"id": "mine_0", "text": "wardrobe"}],
    }
    points, summary = merge_review_points(ai, review, "new_hash")

    # 补丁不应用：AI 点一条不少、自建点也不返回
    assert [p["text"] for p in points] == ["coat", "shirt"]
    assert all(p["origin"] == "ai" for p in points)
    assert summary["stale"] is True
    # 但补丁本身仍然可读（E6 提示「原句已更新，可恢复或重新标注」）
    assert summary["removed_count"] == 1
    assert summary["added_count"] == 1


# ===========================================================================
# B31 / B33：E4' 提交我的补丁 + E6 我的校对与学习历史
# ===========================================================================

_SID = "s_001"
_SCHOLAR = "sch_001"
_HASH = "h1"


def _db_with_ai(ai_points=None, content_hash=_HASH) -> FakeDB:
    """造一个 FakeDB：可选地预置一条 AI 抽取记录（`english_extension_point`）。"""
    db = FakeDB()
    if ai_points is not None:
        db.add(
            config.EXTENSION_POINT_COLLECTION,
            {
                "sentence_id": _SID,
                "content_hash": content_hash,
                "points": ai_points,
                "source": "llm",
                "updated_at": 1,
            },
        )
    return db


def _save(db, **kw):
    payload = {
        "scholar_id": _SCHOLAR,
        "sentence_id": _SID,
        "content_hash": _HASH,
        "removed_texts": [],
        "added": [],
    }
    payload.update(kw)
    return _run(save_review(db, **payload))


def _log_event(db) -> list[tuple[str, str]]:
    return [
        (r["action"], r["target"]) for r in db.all(config.EXTENSION_REVIEW_LOG_COLLECTION)
    ]


# ---------------------------------------------------------------------------
# diff_review — 三类事件（remove / restore / add）
# ---------------------------------------------------------------------------


def _events(old, new):
    return [(e["action"], e["target"]) for e in diff_review(old, new)]


def test_diff_first_submit_emits_remove_and_add():
    new = {"removed_texts": ["coat"], "added": [{"type": "word", "text": "wardrobe"}]}
    assert _events(None, new) == [("remove", "coat"), ("add", "wardrobe")]


def test_diff_restore_event():
    old = {"removed_texts": ["coat", "shirt"], "added": []}
    new = {"removed_texts": ["coat"], "added": []}
    assert _events(old, new) == [("restore", "shirt")]


def test_diff_is_normalization_insensitive():
    """删除主键走归一化：大小写 / 首尾标点差异不算一次变更。"""
    old = {"removed_texts": ["Coat."], "added": []}
    new = {"removed_texts": ["coat"], "added": []}
    assert diff_review(old, new) == []


def test_diff_removing_my_own_point_emits_remove():
    old = {"removed_texts": [], "added": [{"type": "word", "text": "Wardrobe"}]}
    new = {"removed_texts": [], "added": []}
    assert _events(old, new) == [("remove", "wardrobe")]


def test_diff_add_payload_carries_full_point():
    point = {"type": "phrase", "text": "take part in", "meaning_zh": "参加"}
    events = diff_review(None, {"removed_texts": [], "added": [point]})
    assert events[0]["payload"] == point


def test_diff_no_change_returns_empty():
    body = {"removed_texts": ["coat"], "added": [{"type": "word", "text": "wardrobe"}]}
    assert diff_review(body, dict(body)) == []


# ---------------------------------------------------------------------------
# append_review_log — 只 append
# ---------------------------------------------------------------------------


def test_append_review_log_id_format():
    db = FakeDB()
    log_id = _run(
        append_review_log(
            db,
            scholar_id=_SCHOLAR,
            sentence_id=_SID,
            content_hash=_HASH,
            action="remove",
            target="coat",
        )
    )

    assert log_id.startswith("erv_")
    assert len(log_id) == len("erv_") + 32
    rows = db.all(config.EXTENSION_REVIEW_LOG_COLLECTION)
    assert len(rows) == 1
    assert rows[0]["log_id"] == log_id
    assert rows[0]["action"] == "remove"
    assert rows[0]["target"] == "coat"


# ---------------------------------------------------------------------------
# save_review — E4'
# ---------------------------------------------------------------------------


def test_save_review_first_time_inserts_overlay_and_logs():
    db = _db_with_ai([_ai_point("coat", 0), _ai_point("shirt", 1)])
    out = _save(db, removed_texts=["coat"], added=[{"type": "word", "text": "wardrobe"}])

    assert out["status"] == "success"
    assert out["review"]["removed_texts"] == ["coat"]
    assert out["review"]["stale"] is False
    assert out["changes"] == {"added": 1, "removed": 1, "restored": 0}
    # 读侧合并结果随出参返回（AI 剔除项已去、自建点追加在后）
    assert [p["text"] for p in out["points"]] == ["shirt", "wardrobe"]
    assert [p["origin"] for p in out["points"]] == ["ai", "mine"]

    assert len(db.all(config.EXTENSION_REVIEW_COLLECTION)) == 1
    assert set(_log_event(db)) == {("remove", "coat"), ("add", "wardrobe")}


def test_save_review_second_time_updates_in_place():
    db = _db_with_ai([_ai_point("coat", 0)])
    first = _save(db, removed_texts=["coat"])
    second = _save(db, removed_texts=[])

    rows = db.all(config.EXTENSION_REVIEW_COLLECTION)
    assert len(rows) == 1  # 原地替换，不新增行（唯一索引 (scholar_id, sentence_id)）
    assert rows[0]["removed_texts"] == []
    assert rows[0]["created_at"] == first["review"]["created_at"]
    assert second["changes"] == {"added": 0, "removed": 0, "restored": 1}
    assert ("restore", "coat") in _log_event(db)


def test_save_review_never_writes_ai_collection():
    """§4.26 红线②：校对只写 extension_review / extension_review_log。"""
    db = _db_with_ai([_ai_point("coat", 0)])
    before = db.snapshot(config.EXTENSION_POINT_COLLECTION)
    _save(db, removed_texts=["coat"], added=[{"type": "word", "text": "wardrobe"}])

    assert db.snapshot(config.EXTENSION_POINT_COLLECTION) == before
    assert all(coll != config.EXTENSION_POINT_COLLECTION for _, coll in db.write_log)


def test_save_review_normalizes_removed_texts():
    db = _db_with_ai([_ai_point("coat", 0)])
    out = _save(db, removed_texts=["  Coat.  "])
    assert out["review"]["removed_texts"] == ["coat"]


def test_save_review_rejects_empty_scholar_id():
    with pytest.raises(ExtensionError) as ei:
        _save(_db_with_ai([]), scholar_id=" ")
    assert ei.value.error_code == "INVALID_INPUT"


def test_save_review_rejects_bad_type():
    with pytest.raises(ExtensionError) as ei:
        _save(_db_with_ai([]), added=[{"type": "sentence", "text": "whole sentence"}])
    assert ei.value.error_code == "INVALID_INPUT"


def test_save_review_rejects_too_many_added():
    added = [{"type": "word", "text": f"w{i}"} for i in range(config.EXTENSION_REVIEW_MAX_ADDED + 1)]
    with pytest.raises(ExtensionError) as ei:
        _save(_db_with_ai([]), added=added)
    assert ei.value.error_code == "INVALID_INPUT"


def test_save_review_rejects_too_long_text():
    with pytest.raises(ExtensionError) as ei:
        _save(_db_with_ai([]), added=[{"type": "word", "text": "x" * (_MINE_TEXT_MAX_LEN + 1)}])
    assert ei.value.error_code == "INVALID_INPUT"


def test_save_review_disabled(monkeypatch):
    monkeypatch.setattr(config, "EXTENSION_REVIEW_ENABLED", 0)
    with pytest.raises(ExtensionError) as ei:
        _save(_db_with_ai([]))
    assert ei.value.error_code == "EXTENSION_DISABLED"


# ---------------------------------------------------------------------------
# get_review — E6（只读）
# ---------------------------------------------------------------------------


def test_get_review_empty():
    out = _run(get_review(FakeDB(), scholar_id=_SCHOLAR, sentence_id=_SID))
    assert out == {"review": None, "history": []}


def test_get_review_history_is_desc_and_scoped_to_scholar():
    db = FakeDB()
    for at in (1, 3, 2):
        db.add(
            config.EXTENSION_REVIEW_LOG_COLLECTION,
            {
                "log_id": f"erv_{at}",
                "scholar_id": _SCHOLAR,
                "sentence_id": _SID,
                "action": "remove",
                "target": f"t{at}",
                "payload": {},
                "at": at,
            },
        )
    # 另一个学习者的历史不得串台
    db.add(
        config.EXTENSION_REVIEW_LOG_COLLECTION,
        {
            "log_id": "erv_other",
            "scholar_id": "sch_other",
            "sentence_id": _SID,
            "action": "add",
            "target": "x",
            "payload": {},
            "at": 9,
        },
    )

    out = _run(get_review(db, scholar_id=_SCHOLAR, sentence_id=_SID))
    assert [h["at"] for h in out["history"]] == [3, 2, 1]
    assert all(h["log_id"] != "erv_other" for h in out["history"])


def test_get_review_marks_stale_but_keeps_patch():
    """红线⑦：原句已更新只标 stale，**不自动删除**学习者的 removed_texts / added。"""
    db = _db_with_ai([_ai_point("coat", 0)], content_hash="new_hash")
    _save(db, content_hash="old_hash", removed_texts=["coat"], added=[{"type": "word", "text": "wardrobe"}])

    out = _run(get_review(db, scholar_id=_SCHOLAR, sentence_id=_SID))
    assert out["review"]["stale"] is True
    assert out["review"]["removed_texts"] == ["coat"]
    assert [a["text"] for a in out["review"]["added"]] == ["wardrobe"]


def test_get_review_is_read_only():
    db = _db_with_ai([_ai_point("coat", 0)])
    _save(db, removed_texts=["coat"])
    db.reset_write_log()

    _run(get_review(db, scholar_id=_SCHOLAR, sentence_id=_SID))
    assert db.write_calls == {"insert": 0, "update": 0, "delete": 0}


def test_get_review_rejects_empty_scholar_id():
    with pytest.raises(ExtensionError) as ei:
        _run(get_review(FakeDB(), scholar_id="", sentence_id=_SID))
    assert ei.value.error_code == "INVALID_INPUT"


def test_get_review_disabled(monkeypatch):
    monkeypatch.setattr(config, "EXTENSION_REVIEW_ENABLED", 0)
    with pytest.raises(ExtensionError) as ei:
        _run(get_review(FakeDB(), scholar_id=_SCHOLAR, sentence_id=_SID))
    assert ei.value.error_code == "EXTENSION_DISABLED"
