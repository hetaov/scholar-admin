"""单元测试: P2-2 LearningContextBuilder 向量版 — book/lesson/sentences 填充

覆盖（对应 会话训练评估重构执行计划.md §3 S4.3 后续方向① P2-2 / ADR-0017）：
- resolve_current_position：最近 skill_state 定位当前课（lesson/textbook）；无历史 → None
- build_learning_context：
  - 无历史 → 占位（book None / lesson None / sentences []）+ cold_start，不报错
  - 有历史 + FakeCurriculumRetriever → book/lesson 快照 + sentences =
    required（当前课）+ optional（跨课召回，带 score）
  - RAG 内部异常 → 降级不阻断（其余字段完整，无 optional）
  - 默认 retriever 未配置模型 → no-op 降级（仅 required）
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from services.cold_start import COLD_START_SEQUENCE
from services.learning_context import build_learning_context, resolve_current_position
from services.rag_retriever import FakeCurriculumRetriever
from tests.fakes.fake_db import FakeDB
from tests.fakes.seed_factory import seed_content, seed_skill_states


def _ts(day: int, hour: int) -> int:
    return int(datetime(2026, 8, day, hour, 0, 0, tzinfo=timezone.utc).timestamp())


def _seed_history(db, *, with_textbook: bool = True) -> None:
    """内容层级 3 课 6 句 + scholar_1 两个状态（最近学习位置 l3）。"""
    seed_content(
        db,
        textbook_id="tb_1",
        chapter_id="c1",
        lesson_ids=("l1", "l2", "l3"),
        sentence_ids=("s1", "s2", "s3", "s4", "s5", "s6"),
    )
    if with_textbook:
        db.add("textbook_v2", {
            "_id": "tb_1", "textbook_id": "tb_1", "title": "Demo Book",
            "grade": "G1", "level": "L1",
        })
    # 最近学习位置 l3（updated_at 最大）→ 当前课 l3
    seed_skill_states(db, [
        {"scholar_id": "scholar_1", "sentence_id": "s1", "lesson_id": "l1",
         "skill_code": "translation", "attempt_count": 2, "mastery_score": 80,
         "status": "learning", "next_review_at": _ts(16, 10), "updated_at": 1000},
        {"scholar_id": "scholar_1", "sentence_id": "s6", "lesson_id": "l3",
         "skill_code": "translation", "attempt_count": 1, "mastery_score": 50,
         "status": "learning", "next_review_at": _ts(16, 10), "updated_at": 2000},
    ])


class TestResolveCurrentPosition:
    def test_latest_state_wins(self):
        db = FakeDB()
        _seed_history(db)
        pos = asyncio.run(resolve_current_position(db, scholar_id="scholar_1"))
        assert pos["lesson_id"] == "l3"
        assert pos["lesson"]["title"] == "L3"
        assert pos["textbook"]["title"] == "Demo Book"

    def test_no_history(self):
        db = FakeDB()
        assert asyncio.run(resolve_current_position(db, scholar_id="scholar_1")) is None

    def test_no_textbook_snapshot_none(self):
        db = FakeDB()
        _seed_history(db, with_textbook=False)
        pos = asyncio.run(resolve_current_position(db, scholar_id="scholar_1"))
        assert pos["lesson_id"] == "l3"
        assert pos["textbook"] is None


class TestBuildLearningContext:
    def test_no_history_placeholders(self):
        """无历史 → 占位 + cold_start（不报错）。"""
        db = FakeDB()
        seed_content(db, lesson_ids=("l1", "l2"), sentence_ids=("s1", "s2", "s3", "s4"))
        ctx = asyncio.run(build_learning_context(db, scholar_id="scholar_1", date="2026-08-16"))
        assert ctx["learner"]["cold_start"] is True
        assert ctx["book"] is None
        assert ctx["lesson"] is None
        assert ctx["sentences"] == []
        assert ctx["currentIntent"] == "cold_start"
        assert ctx["activityType"] == list(COLD_START_SEQUENCE)

    def test_with_history_and_rag(self):
        """有历史 + RAG → book/lesson 快照 + required（当前课）+ optional（跨课，带 score）。"""
        db = FakeDB()
        _seed_history(db)
        retriever = FakeCurriculumRetriever(sentences=[
            {"sentence_id": "s2", "text": "Text s2", "translation": "译s2",
             "lesson_id": "l1", "textbook_id": "tb_1"},
            {"sentence_id": "s3", "text": "Text s3", "translation": "译s3",
             "lesson_id": "l2", "textbook_id": "tb_1"},
        ])
        ctx = asyncio.run(build_learning_context(
            db, scholar_id="scholar_1", date="2026-08-16", retriever=retriever,
        ))
        assert ctx["learner"]["cold_start"] is False
        assert ctx["book"]["textbook_id"] == "tb_1"
        assert ctx["book"]["title"] == "Demo Book"
        assert ctx["lesson"]["lesson_id"] == "l3"
        assert ctx["lesson"]["title"] == "L3"
        # required = 当前课 l3 的句子（s5/s6，seed 3 课 6 句每课 2 句）
        required = [s for s in ctx["sentences"] if s["source"] == "required"]
        assert [s["sentence_id"] for s in required] == ["s5", "s6"]
        assert all("score" not in s for s in required)
        # optional = 跨课召回（排除 l3），带 score
        optional = [s for s in ctx["sentences"] if s["source"] == "optional"]
        assert [s["sentence_id"] for s in optional] == ["s2", "s3"]
        assert optional[0]["score"] == 1.0
        assert optional[1]["score"] == 0.9

    def test_rag_failure_degrades_gracefully(self):
        """RAG 内部异常 → 降级不阻断（book/lesson 保留、无 optional、其余字段完整）。"""
        db = FakeDB()
        _seed_history(db)

        class BoomRetriever(FakeCurriculumRetriever):
            async def retrieve(self, db, *, scholar_id, query, top_k=None, exclude_lesson_ids=None):
                raise RuntimeError("boom")

        ctx = asyncio.run(build_learning_context(
            db, scholar_id="scholar_1", date="2026-08-16", retriever=BoomRetriever(),
        ))
        assert ctx["learner"]["cold_start"] is False
        assert ctx["lesson"]["lesson_id"] == "l3"
        assert ctx["book"]["textbook_id"] == "tb_1"
        assert all(s["source"] == "required" for s in ctx["sentences"])
        assert not any(s["source"] == "optional" for s in ctx["sentences"])
        assert ctx["currentIntent"] == "review"
        assert ctx["activityType"]

    def test_default_retriever_noop_when_unconfigured(self, monkeypatch):
        """未配置 embedding 模型 → 默认 no-op 降级（仅 required，不触网）。"""
        from services import rag_retriever

        monkeypatch.setattr(rag_retriever, "RAG_EMBEDDING_MODEL", "")
        db = FakeDB()
        _seed_history(db)
        ctx = asyncio.run(build_learning_context(
            db, scholar_id="scholar_1", date="2026-08-16",
        ))
        assert ctx["lesson"]["lesson_id"] == "l3"
        assert all(s["source"] == "required" for s in ctx["sentences"])
        assert not any(s["source"] == "optional" for s in ctx["sentences"])
