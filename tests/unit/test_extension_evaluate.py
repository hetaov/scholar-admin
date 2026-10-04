"""单元测试：英文语句扩展评测 services/english/extension.py

覆盖：
- grade_l1_fill：大小写 / 标点归一化后相等即命中（零 LLM、零抖动）
- grade_l2_rubric：must_use_hit 后端独立判定（不依赖 LLM 自述）
- run_evaluate_pipeline：l1_fill 确定性判分、l2_sentence 调用 LLM
- **红线 R1 / R9**：零 mastery 状态写入（契约 §3.18 红线 ① / ⑥）

> R9 的断言**必须与 R1 同文件放置**（契约 §3.18 红线⑥ 明文要求，防止 H1 类拦截器
> 绕过），故 E4'/E6 的零状态写入断言也落在本文件，而非 `test_extension_review.py`。
"""
from __future__ import annotations

import asyncio

import config
from tests.fakes.fake_db import FakeDB

from services.english.extension import (
    _persist_point,
    grade_l1_fill,
    grade_l2_rubric,
    run_evaluate_pipeline,
)
from services.english.extension_review import get_review, save_review


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# L1 确定性判分
# ---------------------------------------------------------------------------


def test_grade_l1_fill_exact_match():
    pt = {"text": "take part in", "id": "s1#0"}
    r = grade_l1_fill(pt, "take part in")
    assert r["passed"] is True
    assert r["score"] == 1


def test_grade_l1_fill_case_insensitive():
    pt = {"text": "take part in", "id": "s1#0"}
    assert grade_l1_fill(pt, "TAKE PART IN")["passed"] is True


def test_grade_l1_fill_punctuation_normalized():
    pt = {"text": "take part in", "id": "s1#0"}
    assert grade_l1_fill(pt, "take part in.")["passed"] is True
    assert grade_l1_fill(pt, "take, part, in")["passed"] is True


def test_grade_l1_fill_wrong_answer():
    pt = {"text": "take part in", "id": "s1#0"}
    r = grade_l1_fill(pt, "join")
    assert r["passed"] is False
    assert r["score"] == 0
    assert r["wrong_ids"] == ["s1#0"]


def test_grade_l1_fill_empty_answer():
    pt = {"text": "take part in", "id": "s1#0"}
    assert grade_l1_fill(pt, "")["passed"] is False


# ---------------------------------------------------------------------------
# L2 must_use_hit 独立判定
# ---------------------------------------------------------------------------


def test_must_use_hit_all_points_used():
    pts = [{"text": "take part in"}, {"text": "discussion"}]
    r = grade_l2_rubric(pts, "I take part in the discussion.")
    assert r["must_use_hit"] is True


def test_must_use_hit_partial_used():
    pts = [{"text": "take part in"}, {"text": "discussion"}]
    r = grade_l2_rubric(pts, "I take part in it.")
    assert r["must_use_hit"] is False


def test_must_use_hit_none_used():
    pts = [{"text": "take part in"}]
    r = grade_l2_rubric(pts, "I joined the meeting.")
    assert r["must_use_hit"] is False


def test_must_use_hit_empty_points():
    r = grade_l2_rubric([], "anything")
    assert r["must_use_hit"] is False


def test_must_use_hit_ignores_mine_points():
    """B34：学习者自建点（`origin="mine"`）不进判分 —— 既不计入命中，也不拖累命中。"""
    pts = [
        {"text": "take part in", "origin": "ai"},
        {"text": "scarf", "origin": "mine"},
    ]
    r = grade_l2_rubric(pts, "I take part in the meeting.")
    assert r["must_use_hit"] is True
    assert r["used_points"] == ["take part in"]


def test_must_use_hit_false_when_selected_is_all_mine():
    """B34：只选了自建点 → `ai` 点集合为空 → must_use_hit=False（不是 True）。"""
    pts = [{"text": "scarf", "origin": "mine"}]
    r = grade_l2_rubric(pts, "I wear a scarf.")
    assert r["must_use_hit"] is False
    assert r["used_points"] == []


def test_must_use_hit_treats_missing_origin_as_ai():
    """向后兼容：未带 `origin` 的点（既有抽取产出）按 `ai` 处理，行为不变。"""
    pts = [{"text": "take part in"}]
    assert grade_l2_rubric(pts, "I take part in it.")["must_use_hit"] is True


# ---------------------------------------------------------------------------
# 红线 R1 / R9：零 mastery 状态写入（api-contract §3.18 红线 ① / ⑥）
#
# 判定方式：`FakeDB.write_log` 记录每次 insert / update / delete 的集合名。
#   - R1：E2 评测链路**零写入**；E1 落点只写 `english_extension_point`；
#   - R9：E4' / E6 只写 `extension_review` / `extension_review_log`，AI 集合零写入。
# 白名单之外的集合（含一切既有学习状态集合）出现写入即红线被触碰 —— 因此无需穷举
# 状态集合名，也不会因既有集合改名而失效。
# ---------------------------------------------------------------------------

# extension 链路允许写入的集合（data-model §4.24 ~ §4.27）
_EXTENSION_WRITE_WHITELIST = frozenset(
    {
        config.EXTENSION_POINT_COLLECTION,
        config.EXTENSION_TASK_COLLECTION,
        config.EXTENSION_REVIEW_COLLECTION,
        config.EXTENSION_REVIEW_LOG_COLLECTION,
    }
)

# mastery 语义键：出现在 extension 写入的文档里即触碰红线。
# 注意 `status` 是**任务 / 抽取状态**（pending/success/failed），不是 mastery status，
# 故不在其列；`extension_task` 同理豁免。
_MASTERY_KEYS = ("mastery", "mastery_score", "learned", "mastered")


def _assert_no_state_write(db: FakeDB) -> None:
    """红线护栏：写集合 ⊆ 白名单；非任务集合中无 mastery 语义键。"""
    written = {coll for _, coll in db.write_log}
    outside = written - _EXTENSION_WRITE_WHITELIST
    assert not outside, f"写入了非 extension 集合（触碰 R1/R9）：{outside}"
    for coll in written - {config.EXTENSION_TASK_COLLECTION}:
        for doc in db.all(coll):
            hit = [k for k in _MASTERY_KEYS if k in doc]
            assert not hit, f"{coll} 文档出现 mastery 语义字段：{hit}"


def test_r1_evaluate_pipeline_writes_nothing():
    """R1：E2 评测链路（L1 确定性判分）**零写入** —— 不写任何集合，遑论 mastery。"""
    db = FakeDB()
    pts = [{"id": "s1#0", "type": "word", "text": "take part in", "origin": "ai"}]
    out = _run(
        run_evaluate_pipeline(
            db,
            task_type="l1_fill",
            selected_ids=["s1#0"],
            points_snapshot=pts,
            input_mode="text",
            user_input="take part in",
        )
    )
    assert out["passed"] is True
    assert db.write_log == []
    _assert_no_state_write(db)


def test_r1_l2_evaluate_pipeline_writes_nothing(monkeypatch):
    """R1：E2 评测链路（L2 rubric，LLM 打桩）同样零写入。"""
    import services.english.extension as ext

    async def _fake_llm(points, user_answer):  # noqa: ANN001 - 测试桩
        return {
            "scores": {
                "target_usage": 2,
                "grammar": 2,
                "semantics": 2,
                "register_collocation": 2,
            },
            "total": 8,
            "passed": True,
            "errors": [],
            "suggestion": "",
            "model_sentence": "I take part in the discussion.",
        }

    monkeypatch.setattr(ext, "_call_l2_rubric_llm", _fake_llm)
    db = FakeDB()
    pts = [{"id": "s1#0", "type": "phrase", "text": "take part in", "origin": "ai"}]
    out = _run(
        run_evaluate_pipeline(
            db,
            task_type="l2_sentence",
            selected_ids=["s1#0"],
            points_snapshot=pts,
            input_mode="text",
            user_input="I take part in the discussion.",
        )
    )
    assert out["passed"] is True
    assert out["must_use_hit"] is True
    assert db.write_log == []
    _assert_no_state_write(db)


def test_r1_persist_point_writes_only_ai_collection():
    """R1：E1 唯一的写侧出口只写 `english_extension_point`（只存 AI 产出）。"""
    db = FakeDB()
    _run(
        _persist_point(
            db,
            point_key="pk_1",
            sentence_id="sent_1",
            textbook_id=None,
            lesson_id=None,
            content_hash="h1",
            status="success",
            source="llm",
            points=[{"id": "sent_1#0", "type": "word", "text": "coat"}],
            attempts=1,
            fallback_reason="",
            scholar_id=None,
        )
    )
    assert {coll for _, coll in db.write_log} == {config.EXTENSION_POINT_COLLECTION}
    _assert_no_state_write(db)


def test_r9_review_writes_only_user_collections():
    """R9：E4' 提交只写 `extension_review` / `extension_review_log`；E6 只读。"""
    db = FakeDB()
    db.add(
        config.EXTENSION_POINT_COLLECTION,
        {
            "point_key": "pk_1",
            "sentence_id": "sent_1",
            "content_hash": "h1",
            "status": "success",
            "source": "llm",
            "points": [{"id": "sent_1#0", "type": "word", "text": "coat"}],
            "updated_at": 1,
        },
    )
    ai_before = db.snapshot(config.EXTENSION_POINT_COLLECTION)

    _run(
        save_review(
            db,
            scholar_id="sch_1",
            sentence_id="sent_1",
            content_hash="h1",
            removed_texts=["coat"],
            added=[{"type": "word", "text": "scarf", "meaning_zh": "围巾"}],
            note="",
        )
    )

    written = {coll for _, coll in db.write_log}
    assert written == {
        config.EXTENSION_REVIEW_COLLECTION,
        config.EXTENSION_REVIEW_LOG_COLLECTION,
    }
    _assert_no_state_write(db)
    # AI 集合零写入、零回写（§4.26 红线②）
    assert db.snapshot(config.EXTENSION_POINT_COLLECTION) == ai_before

    # E6：只读
    db.reset_write_log()
    got = _run(get_review(db, scholar_id="sch_1", sentence_id="sent_1"))
    assert db.write_log == []
    assert got["review"]["removed_texts"] == ["coat"]
    # 历史两条：remove(coat) + add(scarf)
    assert [(h["action"], h["target"]) for h in got["history"]] == [
        ("remove", "coat"),
        ("add", "scarf"),
    ]
