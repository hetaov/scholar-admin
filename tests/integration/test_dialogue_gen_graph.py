"""集成测试：T3 生成图 × FakeDB（设计稿 §4.3 / §4.6 / §8.1-3）。

覆盖图与真实数据层协作（不触网）：
- `_load_recall` 召回不足时从「任务组所在课的相邻已学句」补入（走 get_skill_states/get_sentences_by_ids）；
- 全图：召回 id 白名单透传（used/recalled 对照）、覆盖校验通过；
- 全图：覆盖重试用尽 → 非对话降级（`notes` 留痕，覆盖完整）。
"""
from __future__ import annotations

import asyncio

from services.learning.dialogue_gen_graph import node_load_context, run_dialogue_gen_graph
from services.learning.rag_retriever import FakeCurriculumRetriever
from tests.fakes.seed_factory import (
    dialogue_gen_v2_flow_gen,
    seed_content,
    seed_skill_states_with_metrics,
)

SCHOLAR = "scholar_1"


def _run(coro):
    return asyncio.run(coro)


def _context(*, lesson_sentences, recall_enabled=True, top_k=4):
    return {
        "task_group": {
            "lesson_id": "l1",
            "group_id": "g1",
            "group_label": "L1 组",
            "sentences": lesson_sentences,
        },
        "scenario": {"background": "两个同学在讨论上周末的活动"},
        "roles": [
            {"code": "A", "name": "Tom", "identity": "student"},
            {"code": "B", "name": "Lily", "identity": "classmate"},
        ],
        "recall": {"enabled": recall_enabled, "top_k": top_k},
        "metrics": {"enabled": True, "weak_skills": ["past_tense"]},
        "prompt_lang": "zh",
        "preferred_type": "auto",
    }


class TestRecallBackfill:
    def test_backfill_from_same_lesson_when_retriever_empty(self, fake_db):
        # l1: s_1(必用) / s_2 / s_5；l2: s_3 / s_4
        seed_content(
            fake_db,
            lesson_ids=("l1", "l2"),
            sentence_ids=("s_1", "s_2", "s_5", "s_3", "s_4"),
        )
        seed_skill_states_with_metrics(
            fake_db,
            scholar_id=SCHOLAR,
            sentence_ids=("s_2", "s_5"),
            lesson_id="l1",
            mastery=0.5,
        )
        context = _context(
            lesson_sentences=[{"sentence_id": "s_1", "content": "Last week I went to the theatre."}]
        )
        state = {
            "scholar_id": SCHOLAR,
            "context": context,
            "preferred_type": "auto",
            "retry_count": 0,
        }
        out = _run(
            node_load_context(state, db=fake_db, retriever=FakeCurriculumRetriever([]))
        )
        # 跨课召回为空 → 从当前课补入 s_2 / s_5（≥2 条）→ 不记 recall_insufficient
        assert set(out["recall_ids"]) == {"s_2", "s_5"}
        assert out["context_notes"] == []

    def test_recall_disabled_no_hits(self, fake_db):
        seed_content(fake_db, lesson_ids=("l1",), sentence_ids=("s_1", "s_2"))
        context = _context(
            lesson_sentences=[{"sentence_id": "s_1", "content": "Last week I went to the theatre."}],
            recall_enabled=False,
        )
        out = _run(
            node_load_context(
                {"scholar_id": SCHOLAR, "context": context, "retry_count": 0},
                db=fake_db,
                retriever=FakeCurriculumRetriever([]),
            )
        )
        assert out["recalled"] == []


class TestGraphWithFakeDB:
    def test_full_graph_used_and_recalled_comparison(self, fake_db):
        retriever = FakeCurriculumRetriever(
            [
                {"sentence_id": "r1", "text": "It is a cat.", "lesson_id": "l2"},
                {"sentence_id": "r2", "text": "She is my sister.", "lesson_id": "l2"},
            ]
        )

        async def gen(messages):
            import json

            return json.dumps(
                {
                    "content_type": "dialogue",
                    "sub_type": None,
                    "background_intro": "背景",
                    "turns": [
                        {"speaker": "A", "text": "Last week I went to the theatre.", "target_sentence_id": "s_1"},
                        {"speaker": "B", "text": "It is a cat.", "target_sentence_id": None},
                    ],
                    "prompts": [],
                    "used_sentence_ids": ["s_1", "s_fake"],
                    "recalled_sentence_ids": ["r1", "r2", "r_fake"],
                    "difficulty": 2,
                    "notes": None,
                },
                ensure_ascii=False,
            )

        result = _run(
            run_dialogue_gen_graph(
                db=fake_db,
                task_id="dg_full",
                scholar_id=SCHOLAR,
                context=_context(
                    lesson_sentences=[{"sentence_id": "s_1", "content": "Last week I went to the theatre."}]
                ),
                retriever=retriever,
                generator=dialogue_gen_v2_flow_gen(gen),
            )
        )
        # 编造 id 被过滤，召回白名单 = 任务组 ∪ 召回候选
        assert result["used_sentence_ids"] == ["s_1"]
        assert set(result["recalled_sentence_ids"]) == {"r1", "r2"}
        assert result["coverage"] == {"required_total": 1, "required_used": 1, "ratio": 1.0}
        assert result["notes"] is None

    def test_full_graph_retry_exhausted_falls_back(self, fake_db):
        async def gen(messages):
            import json

            return json.dumps(
                {
                    "content_type": "dialogue",
                    "sub_type": None,
                    "turns": [
                        {"speaker": "A", "text": "Last week I went to the theatre.", "target_sentence_id": "s_1"},
                        {"speaker": "B", "text": "ok", "target_sentence_id": None},
                    ],
                    "prompts": [],
                    "used_sentence_ids": ["s_1"],  # 永不覆盖 s_2
                    "recalled_sentence_ids": [],
                    "difficulty": 2,
                    "notes": None,
                },
                ensure_ascii=False,
            )

        result = _run(
            run_dialogue_gen_graph(
                db=fake_db,
                task_id="dg_fb",
                scholar_id=SCHOLAR,
                context=_context(
                    lesson_sentences=[
                        {"sentence_id": "s_1", "content": "Last week I went to the theatre."},
                        {"sentence_id": "s_2", "content": "I had a very good seat."},
                    ],
                    recall_enabled=False,
                ),
                generator=dialogue_gen_v2_flow_gen(gen),
                max_retry=1,
            )
        )
        assert result["content_type"] == "non_dialogue"
        assert result["retry_count"] == 1
        assert "fallback_non_dialogue" in (result["notes"] or "")
        assert result["coverage"]["required_used"] == 2
