"""单元测试：S4.2 L2 ConversationGraph（LangGraph StateGraph + NoSQL checkpointer）

覆盖（执行计划 S4.2 任务⑨）：
- 图状态节点：load_learner_state / load_lesson_knowledge / generate_scenario /
  evaluate_user_response / advance_stage / generate_ai_utterance
- 条件边：router_entry（scenario/evaluate）/ router_after_advance（utterance/end）
- 整图流程：scenario 首轮初始化 → turn 轮次（checkpoint 恢复断点续聊）
- NoSQLCheckpointSaver：put / get_tuple / list / 序列化往返 / 断点恢复

约定：评估与回复均注入 fake（不触网、确定性）。
"""
from __future__ import annotations

import asyncio

import pytest

from services.conversation_graph import (
    GRAPH_STAGE_AWAITING_EVAL,
    GRAPH_STAGE_ENDED,
    GRAPH_STAGE_READY,
    NoSQLCheckpointSaver,
    build_conversation_graph,
    get_compiled_graph,
    node_advance_stage,
    node_evaluate_user_response,
    node_generate_ai_utterance,
    node_generate_scenario,
    node_load_learner_state,
    node_load_lesson_knowledge,
    router_after_advance,
    router_entry,
    run_turn_graph,
    start_scenario_graph,
)
from services.models_conversation import (
    DEFAULT_DIFFICULTY,
    MAX_TURNS,
    TURN_STAGE_ANSWER,
    TURN_STAGE_DOWNGRADE,
    TURN_STAGE_HINT,
)
from tests.fakes.fake_db import FakeDB

SESSION = {
    "_id": "cvs_1",
    "session_id": "cvs_1",
    "scholar_id": "u1",
    "scenario": "free_talk",
    "topic": "daily conversation",
    "difficulty": 1,
    "sentence_ids": ["sent_1"],
    "translation": "这是一块手表。",
    "cold_start": True,
    "consecutive_failures": 0,
    "turn_index": 0,
    "current_target": "",
}


def _fake_reply(**kwargs):
    return {"reply": "Good job! Keep going.", "next_target": ""}


def _fake_eval(original, response):
    return {
        "meaningful": True,
        "confidence": 0.9,
        "faithfulness": True,
        "anomaly": False,
        "score": 90,
    }


def _seed_sentence(fake_db: FakeDB):
    fake_db.add(
        "sentence_v2",
        {
            "_id": "sent_1",
            "sentence_id": "sent_1",
            "text": "It is a watch.",
            "translation": "这是一块手表。",
            "difficulty": 1,
        },
    )


class TestNodeLoadLearnerState:
    def test_aggregates_session_fields(self):
        out = node_load_learner_state(SESSION)
        ls = out["learner_state"]
        assert ls["cold_start"] is True
        assert ls["difficulty"] == 1
        assert ls["consecutive_failures"] == 0
        assert ls["turn_index"] == 0

    def test_defaults_when_absent(self):
        out = node_load_learner_state({})
        ls = out["learner_state"]
        assert ls["cold_start"] is False
        assert ls["difficulty"] == DEFAULT_DIFFICULTY


class TestNodeLoadLessonKnowledge:
    @pytest.mark.asyncio
    async def test_loads_sentences_and_fills_target(self):
        db = FakeDB()
        _seed_sentence(db)
        out = await node_load_lesson_knowledge(SESSION, db=db)
        assert out["lesson_knowledge"][0]["text"] == "It is a watch."
        # 首轮补 current_target / translation
        assert out["current_target"] == "It is a watch."
        assert out["translation"] == "这是一块手表。"

    @pytest.mark.asyncio
    async def test_no_db_returns_empty(self):
        out = await node_load_lesson_knowledge(SESSION, db=None)
        assert out["lesson_knowledge"] == []
        assert out["current_target"] == ""


class TestNodeGenerateScenario:
    def test_generates_opening(self):
        out = node_generate_scenario(SESSION)
        assert out["stage"] == GRAPH_STAGE_READY
        assert "daily conversation" in out["opening_utterance"]
        assert "free_talk" in out["scenario_desc"]

    def test_opening_with_target(self):
        st = {**SESSION, "current_target": "It is a watch."}
        out = node_generate_scenario(st)
        assert "It is a watch." in out["opening_utterance"]


class TestNodeEvaluateUserResponse:
    @pytest.mark.asyncio
    async def test_invokes_evaluator(self):
        st = {**SESSION, "current_target": "It is a watch.", "utterance": "It is a watch."}
        out = await node_evaluate_user_response(st, evaluator=_fake_eval)
        assert out["stage"] == GRAPH_STAGE_AWAITING_EVAL
        assert out["meaningful"] is True
        assert out["low_confidence"] is False
        assert out["verdict"]["score"] == 90

    @pytest.mark.asyncio
    async def test_async_evaluator_supported(self):
        async def async_eval(original, response):
            return {"meaningful": True, "confidence": 0.95}

        st = {**SESSION, "utterance": "hi"}
        out = await node_evaluate_user_response(st, evaluator=async_eval)
        assert out["meaningful"] is True
        assert out["low_confidence"] is False


class TestNodeAdvanceStage:
    def test_success_resets_failures(self):
        st = {**SESSION, "meaningful": True, "low_confidence": False, "consecutive_failures": 2}
        out = node_advance_stage(st)
        assert out["turn_stage"] == TURN_STAGE_ANSWER
        assert out["consecutive_failures"] == 0
        assert out["turn_index"] == 1
        assert out["session_ended"] is False

    def test_failure_goes_hint(self):
        st = {**SESSION, "meaningful": False, "low_confidence": False, "consecutive_failures": 0}
        out = node_advance_stage(st)
        assert out["turn_stage"] == TURN_STAGE_HINT
        assert out["consecutive_failures"] == 1


class TestNodeGenerateAiUtterance:
    @pytest.mark.asyncio
    async def test_uses_injected_generator(self):
        st = {**SESSION, "current_target": "It is a watch.", "turn_stage": TURN_STAGE_ANSWER}
        out = await node_generate_ai_utterance(st, reply_generator=_fake_reply)
        assert out["reply"] == "Good job! Keep going."
        assert out["current_target"] == "It is a watch."  # 未换句 → 保持
        assert out["stage"] == GRAPH_STAGE_READY

    @pytest.mark.asyncio
    async def test_rule_fallback_without_generator(self):
        st = {**SESSION, "current_target": "It is a watch.", "turn_stage": TURN_STAGE_HINT}
        out = await node_generate_ai_utterance(st, reply_generator=None)
        assert "提示" in out["reply"] or "Try" in out["reply"]

    @pytest.mark.asyncio
    async def test_session_ended_stage(self):
        st = {**SESSION, "turn_stage": TURN_STAGE_ANSWER, "session_ended": True}
        out = await node_generate_ai_utterance(st, reply_generator=_fake_reply)
        assert out["stage"] == GRAPH_STAGE_ENDED


class TestRouters:
    def test_entry_scenario_when_no_utterance(self):
        assert router_entry(SESSION) == "scenario"

    def test_entry_evaluate_with_utterance(self):
        assert router_entry({**SESSION, "utterance": "It is a watch."}) == "evaluate"

    def test_after_advance_utterance(self):
        assert router_after_advance({}) == "utterance"

    def test_after_advance_end(self):
        assert router_after_advance({"session_ended": True}) == "end"


class TestGraphFlow:
    @pytest.mark.asyncio
    async def test_scenario_path(self):
        db = FakeDB()
        _seed_sentence(db)
        graph = get_compiled_graph(db=db, evaluator=_fake_eval, reply_generator=_fake_reply)
        out = await graph.ainvoke(
            {
                "session_id": "cvs_1",
                "scholar_id": "u1",
                "scenario": "free_talk",
                "topic": "daily conversation",
                "difficulty": 1,
                "sentence_ids": ["sent_1"],
                "cold_start": True,
                "current_target": "",
                "consecutive_failures": 0,
                "turn_index": 0,
                "stage": GRAPH_STAGE_READY,
            }
        )
        assert out["stage"] == GRAPH_STAGE_READY
        assert out["current_target"] == "It is a watch."
        assert out["opening_utterance"]  # 开场白已生成
        assert out.get("reply") is None  # scenario 路径不产回复

    @pytest.mark.asyncio
    async def test_turn_path(self):
        db = FakeDB()
        _seed_sentence(db)
        graph = get_compiled_graph(db=db, evaluator=_fake_eval, reply_generator=_fake_reply)
        out = await graph.ainvoke(
            {
                "session_id": "cvs_1",
                "scholar_id": "u1",
                "topic": "daily conversation",
                "difficulty": 1,
                "sentence_ids": ["sent_1"],
                "cold_start": True,
                "current_target": "It is a watch.",
                "consecutive_failures": 0,
                "turn_index": 0,
                "stage": GRAPH_STAGE_READY,
                "utterance": "It is a watch.",
            }
        )
        assert out["reply"] == "Good job! Keep going."
        assert out["turn_stage"] == TURN_STAGE_ANSWER
        assert out["turn_index"] == 1
        assert out["stage"] == GRAPH_STAGE_READY  # 一轮结束回到待输入


class TestNoSQLCheckpointSaver:
    def _build_saver(self, db: FakeDB) -> NoSQLCheckpointSaver:
        return NoSQLCheckpointSaver(db)

    def _make_config(self, thread_id: str, checkpoint_id: str | None = None) -> dict:
        cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        if checkpoint_id:
            cfg["configurable"]["checkpoint_id"] = checkpoint_id
        return cfg

    @staticmethod
    def _checkpoint(cp_id: str, turn_index: int = 0) -> dict:
        return {
            "v": 1,
            "id": cp_id,
            "ts": f"2026-08-18T00:00:0{turn_index}Z",  # ts 唯一递增，最新选择排序键
            "channel_values": {"stage": GRAPH_STAGE_READY, "turn_index": turn_index},
            "channel_versions": {},
            "versions_seen": {},
            "pending_writes": [],
        }

    @pytest.mark.asyncio
    async def test_put_and_get_tuple_roundtrip(self):
        db = FakeDB()
        saver = self._build_saver(db)
        cfg = self._make_config("cvs_1")
        metadata = {"source": "loop", "step": 0, "writes": {}}

        returned = await saver.aput(cfg, self._checkpoint("cp_1"), metadata, new_versions={})
        assert returned["configurable"]["checkpoint_id"] == "cp_1"

        # 落库文档：base64 编码存储
        docs = db.all("conversation_graph_checkpoint")
        assert len(docs) == 1
        assert docs[0]["thread_id"] == "cvs_1"
        assert docs[0]["checkpoint_id"] == "cp_1"
        assert docs[0]["checkpoint_type"]  # json / msgpack
        assert "checkpoint" in docs[0] and "metadata" in docs[0]

        # 读取往返（LangGraph 运行时走 async 方法；CheckpointTuple 为 NamedTuple）
        tup = await saver.aget_tuple(self._make_config("cvs_1", "cp_1"))
        assert tup is not None
        assert tup.checkpoint["id"] == "cp_1"
        assert tup.checkpoint["channel_values"]["turn_index"] == 0
        assert tup.metadata["source"] == "loop"
        assert tup.parent_config is None

    @pytest.mark.asyncio
    async def test_get_tuple_latest_when_no_id(self):
        db = FakeDB()
        saver = self._build_saver(db)
        for i in range(3):
            await saver.aput(
                self._make_config("cvs_1"),
                self._checkpoint(f"cp_{i}", i),
                {"source": "loop"},
                new_versions={},
            )
        tup = await saver.aget_tuple(self._make_config("cvs_1"))
        assert tup.checkpoint["channel_values"]["turn_index"] == 2  # 最新

    @pytest.mark.asyncio
    async def test_list_descending(self):
        db = FakeDB()
        saver = self._build_saver(db)
        for i in range(3):
            await saver.aput(
                self._make_config("cvs_1"),
                self._checkpoint(f"cp_{i}", i),
                {"source": "loop"},
                new_versions={},
            )
        listed = [t async for t in saver.alist(self._make_config("cvs_1"), limit=10)]
        ids = [t.config["configurable"]["checkpoint_id"] for t in listed]
        assert ids == ["cp_2", "cp_1", "cp_0"]  # 时间倒序

    @pytest.mark.asyncio
    async def test_put_writes_noop(self):
        db = FakeDB()
        saver = self._build_saver(db)
        assert await saver.aput_writes(self._make_config("cvs_1"), [], "task_1") is None


class TestGraphWithCheckpointer:
    @pytest.mark.asyncio
    async def test_scenario_then_turn_resumes(self):
        """断点续聊：scenario 初始化落 checkpoint → 两轮 turn 状态递增恢复。"""
        db = FakeDB()
        _seed_sentence(db)

        # 1) scenario：图初始化
        init = await start_scenario_graph(db, SESSION, reply_generator=_fake_reply)
        assert init["checkpoint_id"]
        assert init["graph_state"]["current_target"] == "It is a watch."
        assert init["graph_state"]["stage"] == GRAPH_STAGE_READY

        # 模拟路由回写会话
        session = {**SESSION, "graph_state": init["graph_state"], "checkpoint_id": init["checkpoint_id"]}

        # 2) turn1：走 evaluate 路径
        out1 = await run_turn_graph(db, session, "It is a watch.", reply_generator=_fake_reply)
        assert out1["reply"] == "Good job! Keep going."
        assert out1["turn_stage"] == TURN_STAGE_ANSWER
        assert out1["graph_state"]["turn_index"] == 1
        assert out1["graph_state"]["stage"] == GRAPH_STAGE_READY

        # 3) turn2：checkpoint 恢复继续
        session = {**session, "graph_state": out1["graph_state"], "checkpoint_id": out1["checkpoint_id"]}
        out2 = await run_turn_graph(db, session, "It is a watch.", reply_generator=_fake_reply)
        assert out2["graph_state"]["turn_index"] == 2
        assert out2["graph_state"]["difficulty"] == 1

        # checkpoint 落库为 thread_id=session_id；最新一条（按 ts）指向本轮
        docs = db.all("conversation_graph_checkpoint")
        assert all(d["thread_id"] == "cvs_1" for d in docs)
        latest = await db.query(
            "conversation_graph_checkpoint",
            where={"thread_id": "cvs_1"},
            limit=1,
            order=[{"field": "ts", "direction": "desc"}],
        )
        assert latest["records"][0]["checkpoint_id"] == out2["checkpoint_id"]

    @pytest.mark.asyncio
    async def test_conditional_end_branch(self):
        """turn_index 达 MAX_TURNS → advance 置 session_ended → 条件边路由到 END（不生成回复）。"""
        db = FakeDB()
        _seed_sentence(db)
        graph = get_compiled_graph(db=db, evaluator=_fake_eval, reply_generator=_fake_reply)
        out = await graph.ainvoke(
            {
                "session_id": "cvs_1",
                "topic": "daily conversation",
                "difficulty": 1,
                "sentence_ids": ["sent_1"],
                "cold_start": True,
                "current_target": "It is a watch.",
                "consecutive_failures": 0,
                "turn_index": MAX_TURNS - 1,  # 本轮后达上限
                "stage": GRAPH_STAGE_READY,
                "utterance": "It is a watch.",
            }
        )
        assert out["session_ended"] is True
        assert out.get("reply") is None  # END 分支：generate_ai_utterance 未执行
