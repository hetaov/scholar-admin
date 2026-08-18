"""单元测试: S4.3 AI Planner — build_plan 决策 + get_due_review_items 到期口径

覆盖(对应 会话训练评估重构执行计划.md §3 S4.3):
- build_plan 四策略: cold_start / review / weakness / practice
- review_items / activities 截断(top_review / top_activities)
- rationale 含对应策略理由
- get_due_review_items 到期过滤: next_review_at ≤ 当日 23:59:59 且 status ≠ mastered
- 乐观 pick_state: 同句多能力取 progress 最高者
- 排序: next_review_at 升序 + 同到期日 mastery_score 升序
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from services.cold_start import COLD_START_SEQUENCE
from services.learning_planner import build_plan
from services.learning_scheduler import get_due_review_items
from tests.fakes.fake_db import FakeDB
from tests.fakes.seed_factory import seed_content


def _ts(day: int, hour: int) -> int:
    """2026-08-day hour:00 UTC → 秒级时间戳(测试固定日期,避开"今天"漂移)。"""
    return int(datetime(2026, 8, day, hour, 0, 0, tzinfo=timezone.utc).timestamp())


def _context(**overrides) -> dict:
    ctx = {
        "learner": {
            "scholar_id": "scholar_1",
            "has_history": True,
            "evidence_sparse": False,
            "cold_start": False,
        },
        "currentSkills": {},
        "weakSkills": [],
        "recentAttempts": 10,
        "reviewItems": [],
        "currentIntent": "practice",
        "difficulty": 2,
        "activityType": ["content", "shadowing"],
    }
    ctx.update(overrides)
    return ctx


class TestBuildPlan:
    def test_cold_start_strategy(self):
        """无历史 → cold_start: 标准引导序列 + 冷启动理由。"""
        plan = build_plan(_context(learner={
            "scholar_id": "scholar_1", "has_history": False,
            "evidence_sparse": True, "cold_start": True,
        }))
        assert plan["strategy"] == "cold_start"
        assert plan["activities"] == list(COLD_START_SEQUENCE)
        assert "冷启动" in plan["rationale"]
        assert plan["review_items"] == []
        assert plan["difficulty"] == 2

    def test_review_strategy(self):
        """有到期复习项 → review: 复习优先 + 数量写入理由。"""
        review_items = [
            {"sentence_id": "s1", "content": "Text s1", "next_review_at": "2026-08-16T10:00:00"},
        ]
        plan = build_plan(_context(reviewItems=review_items))
        assert plan["strategy"] == "review"
        assert plan["review_items"] == review_items
        assert "1 个到期复习项" in plan["rationale"]
        assert plan["activities"] == ["content", "shadowing"]

    def test_weakness_strategy(self):
        """无到期但有弱项(mastery < 0.6) → weakness: 弱项驱动。"""
        plan = build_plan(_context(
            weakSkills=["pronunciation"],
            reviewItems=[],
        ))
        assert plan["strategy"] == "weakness"
        assert "pronunciation" in plan["rationale"]
        assert plan["review_items"] == []

    def test_practice_strategy(self):
        """无到期且无弱项 → practice: 常规节奏。"""
        plan = build_plan(_context())
        assert plan["strategy"] == "practice"
        assert "常规练习" in plan["rationale"]

    def test_top_review_truncation(self):
        """top_review 截断到期复习项。"""
        review_items = [
            {"sentence_id": f"s{i}"} for i in range(1, 8)
        ]
        plan = build_plan(_context(reviewItems=review_items), top_review=5)
        assert plan["strategy"] == "review"
        assert len(plan["review_items"]) == 5

    def test_top_activities_truncation(self):
        """top_activities 截断推荐活动。"""
        plan = build_plan(
            _context(reviewItems=[{"sentence_id": "s1"}]),
            top_activities=1,
        )
        assert plan["strategy"] == "review"
        assert len(plan["activities"]) == 1

    def test_weakness_beats_practice_when_no_review(self):
        """弱项优先于常规练习(复习缺席时)。"""
        plan = build_plan(_context(weakSkills=["grammar"]))
        assert plan["strategy"] == "weakness"


class TestGetDueReviewItems:
    def test_due_queue_filter_and_sort(self):
        """到期过滤 + 排序: mastered 排除、未来到期排除、ts/mastery 升序。"""
        db = FakeDB()
        seed_content(db)
        db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
            "status": "learned", "mastery_score": 80, "attempt_count": 2,
            "next_review_at": _ts(16, 10), "last_outcome": "correct",
        })
        db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s2", "skill_code": "translation",
            "status": "learning", "mastery_score": 40, "attempt_count": 1,
            "next_review_at": _ts(16, 10),
        })
        # mastered: 排除
        db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s3", "skill_code": "translation",
            "status": "mastered", "mastery_score": 95, "attempt_count": 3,
            "next_review_at": _ts(16, 9),
        })
        # 未来到期(18 日 > 16 日 23:59:59): 排除
        db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s4", "skill_code": "translation",
            "status": "learning", "mastery_score": 50, "attempt_count": 1,
            "next_review_at": _ts(18, 10),
        })
        queue = asyncio.run(get_due_review_items(
            db, scholar_id="scholar_1", date="2026-08-16",
        ))
        assert [q["sentence_id"] for q in queue] == ["s2", "s1"]
        # 同到期时刻 mastery 升序: s2(40) 在 s1(80) 前
        assert queue[0]["mastery_score"] == 40
        assert queue[1]["mastery_score"] == 80
        s1 = queue[1]
        assert s1["content"] == "Text s1"
        assert s1["translation"] == "译s1"
        assert s1["lesson_id"] == "l1"
        assert s1["next_review_at"].startswith("2026-08-16T10:00:00")
        assert s1["last_result"] == "correct"
        assert s1["review_count"] == 2

    def test_empty_queue_when_nothing_due(self):
        """无到期 → 空列表不报错。"""
        db = FakeDB()
        seed_content(db)
        db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
            "status": "learning", "mastery_score": 40, "attempt_count": 1,
            "next_review_at": _ts(18, 10),  # 未来
        })
        queue = asyncio.run(get_due_review_items(
            db, scholar_id="scholar_1", date="2026-08-16",
        ))
        assert queue == []

    def test_limit(self):
        """limit 截断条数。"""
        db = FakeDB()
        seed_content(db)
        for i, sid in enumerate(["s1", "s2", "s3"]):
            db.add("skill_state", {
                "scholar_id": "scholar_1", "sentence_id": sid,
                "skill_code": "translation", "status": "learning",
                "mastery_score": 30 + i * 10, "attempt_count": 1,
                "next_review_at": _ts(16, 10 + i),
            })
        queue = asyncio.run(get_due_review_items(
            db, scholar_id="scholar_1", date="2026-08-16", limit=2,
        ))
        assert len(queue) == 2

    def test_pick_state_highest_progress(self):
        """乐观 pick_state: 同句多能力取 progress 最高者(learned 80 > learning 40)。"""
        db = FakeDB()
        seed_content(db)
        db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "listening",
            "status": "learning", "mastery_score": 40, "attempt_count": 1,
            "next_review_at": _ts(16, 10),
        })
        db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
            "status": "learned", "mastery_score": 80, "attempt_count": 2,
            "next_review_at": _ts(16, 10),
        })
        queue = asyncio.run(get_due_review_items(
            db, scholar_id="scholar_1", date="2026-08-16",
        ))
        assert len(queue) == 1
        assert queue[0]["sentence_id"] == "s1"
        assert queue[0]["mastery_score"] == 80  # 取 progress 最高者(learned)
