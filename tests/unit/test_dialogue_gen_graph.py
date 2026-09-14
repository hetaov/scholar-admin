"""单元测试：v2 对话生成图（每句评估扇出 → 选优 → 后置总结 → 覆盖校验闭环）。

不触网不连库：注入 fake generator / FakeCurriculumRetriever（`db=None`），验证
- 节点纯函数：load_context / evaluate_per_sentence / select_best / summarize /
  build_prompt / refine_prompt / validate_coverage / evaluate / persist；
- 条件边路由：有 viable 候选 → summarize → build_prompt；无候选/总结失败 → fallback；
- 全图流程：首轮覆盖 / 重试后覆盖 / 重试用尽降级为非对话 / LLM 失败 / 无可校验句子；
- 召回：clamp 到 [2,6]、召回不足留痕 `recall_insufficient`、`db=None` 跳过补入。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from config import DIALOGUE_CHECKPOINT_COLLECTION
from tests.fakes.seed_factory import dialogue_gen_v2_flow_gen as _v2_flow_gen
from services.learning.dialogue_gen_graph import (
    BRANCH_BUILD_PROMPT,
    BRANCH_EVALUATE,
    BRANCH_FALLBACK,
    BRANCH_REFINE,
    BRANCH_SELECT_FALLBACK,
    BRANCH_SUMMARIZE,
    BRANCH_SUMMARIZE_FALLBACK,
    NOTE_NO_VIABLE_CANDIDATE,
    NOTE_RESUME_WITHOUT_CHECKPOINT,
    NOTE_SUMMARIZE_FAILED,
    STAGE_SELECTED,
    STAGE_SUMMARIZED,
    build_dialogue_gen_graph,
    dialogue_checkpoint_exists,
    dispatch_per_sentence,
    get_checkpointer,
    latest_checkpoint_id,
    list_dialogue_checkpoints,
    node_build_prompt,
    node_evaluate,
    node_evaluate_per_sentence,
    node_load_context,
    node_persist,
    node_refine_prompt,
    node_select_best,
    node_summarize,
    node_validate_coverage,
    route_after_select_best,
    route_after_summarize,
    route_after_validate,
    run_dialogue_gen_graph,
)
from services.learning.rag_retriever import FakeCurriculumRetriever
from services.providers.dialogue_gen import (
    ERR_COVERAGE_FAILED,
    ERR_LLM_PARSE_ERROR,
    ERR_LLM_UNAVAILABLE,
    DialogueGenError,
)

REQUIRED_IDS = ["s_1", "s_2"]


def _context(**overrides) -> dict:
    ctx = {
        "task_group": {
            "lesson_id": "l1",
            "group_id": "g1",
            "group_label": "L1 叙述组",
            "sentences": [
                {"sentence_id": "s_1", "content": "Last week I went to the theatre."},
                {"sentence_id": "s_2", "content": "I had a very good seat."},
            ],
        },
        "scenario": {"background": "两个同学在讨论上周末的活动"},
        "roles": [
            {"code": "A", "name": "Tom", "identity": "student"},
            {"code": "B", "name": "Lily", "identity": "classmate"},
        ],
        "recall": {"enabled": False, "top_k": 4},
        "metrics": {"enabled": True, "weak_skills": ["past_tense"]},
        "prompt_lang": "zh",
        "preferred_type": "auto",
    }
    ctx.update(overrides)
    return ctx


def _llm_json(
    content_type: str = "dialogue",
    *,
    used: tuple[str, ...] = ("s_1", "s_2"),
    recalled: tuple[str, ...] = (),
    turns: list[dict] | None = None,
    **overrides,
) -> str:
    body = {
        "content_type": content_type,
        "sub_type": None if content_type == "dialogue" else "retell",
        "background_intro": "背景介绍",
        "turns": turns
        or [
            {"speaker": "A", "text": "Last week I went to the theatre.", "target_sentence_id": "s_1"},
            {"speaker": "B", "text": "I had a very good seat.", "target_sentence_id": "s_2"},
        ],
        "prompts": [{"after_turn": 1, "lang": "zh", "text": "试着复述一下"}],
        "used_sentence_ids": list(used),
        "recalled_sentence_ids": list(recalled),
        "difficulty": 2,
        "notes": None,
    }
    body.update(overrides)
    return json.dumps(body, ensure_ascii=False)


def _state(**overrides) -> dict:
    state = {
        "task_id": "dg_1",
        "scholar_id": "scholar_1",
        "context": _context(),
        "preferred_type": "auto",
        "retry_count": 0,
    }
    state.update(overrides)
    return state


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 节点：load_context
# ---------------------------------------------------------------------------


class TestLoadContext:
    def test_loads_required_roles_and_forces_non_dialogue_without_roles(self):
        out = _run(node_load_context(_state(context=_context(roles=[]))))
        assert out["required_ids"] == REQUIRED_IDS
        assert out["content_type"] == "non_dialogue"
        assert out["prompt_lang"] == "zh"
        assert out["recalled"] == []
        assert out["recall_ids"] == []

    def test_no_sentences_raises_coverage_failed(self):
        state = _state(context=_context(task_group={"sentences": []}))
        with pytest.raises(DialogueGenError) as exc:
            _run(node_load_context(state))
        assert exc.value.error_code == ERR_COVERAGE_FAILED

    def test_recall_disabled_skips_retriever(self):
        retriever = FakeCurriculumRetriever(
            [{"sentence_id": "r1", "text": "old sentence", "lesson_id": "l2"}]
        )
        out = _run(
            node_load_context(
                _state(context=_context(recall={"enabled": False})), retriever=retriever
            )
        )
        assert out["recalled"] == []

    def test_recall_returns_hits_and_ids(self):
        retriever = FakeCurriculumRetriever(
            [
                {"sentence_id": "r1", "text": "old sentence 1", "lesson_id": "l2"},
                {"sentence_id": "r2", "text": "old sentence 2", "lesson_id": "l2"},
            ]
        )
        out = _run(
            node_load_context(
                _state(context=_context(recall={"enabled": True, "top_k": 4})),
                retriever=retriever,
            )
        )
        assert out["recall_ids"] == ["r1", "r2"]
        assert out["context_notes"] == []

    def test_recall_insufficient_recorded_when_empty(self):
        # db=None 无法补入 → 召回不足留痕（不静默）
        out = _run(
            node_load_context(
                _state(context=_context(recall={"enabled": True, "top_k": 4})),
                retriever=FakeCurriculumRetriever([]),
            )
        )
        assert out["recalled"] == []
        assert out["context_notes"] == ["recall_insufficient"]


# ---------------------------------------------------------------------------
# 节点：build_prompt / refine_prompt
# ---------------------------------------------------------------------------


class TestPromptNodes:
    def test_build_prompt_injects_recall_block(self):
        out = node_build_prompt(
            _state(
                content_type="dialogue",
                roles=_context()["roles"],
                prompt_lang="zh",
                recalled=[{"sentence_id": "r1", "content": "old sentence"}],
            )
        )
        user = out["messages"][1]["content"]
        assert "召回学习语句" in user
        assert "[r1] old sentence" in user
        assert "上一轮遗漏" not in user

    def test_refine_prompt_increments_retry_and_renders_missing(self):
        out = node_refine_prompt(
            _state(
                content_type="dialogue",
                roles=_context()["roles"],
                prompt_lang="zh",
                missing_ids=["s_2"],
                retry_count=0,
            )
        )
        assert out["retry_count"] == 1
        assert out["refine_missing"] == ["s_2"]
        assert "【上一轮遗漏（本必须原样出现，请补齐）】s_2" in out["messages"][1]["content"]


# ---------------------------------------------------------------------------
# 节点：validate_coverage + 条件边
# ---------------------------------------------------------------------------


class TestCoverageRouting:
    def test_validate_coverage_detects_missing(self):
        out = node_validate_coverage(
            _state(parsed={"used_sentence_ids": ["s_1"]}, required_ids=REQUIRED_IDS)
        )
        assert out["missing_ids"] == ["s_2"]
        assert out["coverage_ok"] is False

    def test_validate_coverage_ok(self):
        out = node_validate_coverage(
            _state(parsed={"used_sentence_ids": ["s_1", "s_2"]}, required_ids=REQUIRED_IDS)
        )
        assert out["missing_ids"] == []
        assert out["coverage_ok"] is True

    def test_route_pass_to_evaluate(self):
        assert route_after_validate({"coverage_ok": True}) == BRANCH_EVALUATE

    def test_route_refine_when_budget_left(self):
        assert (
            route_after_validate({"coverage_ok": False, "retry_count": 0}, max_retry=2)
            == BRANCH_REFINE
        )

    def test_route_fallback_when_budget_exhausted(self):
        assert (
            route_after_validate({"coverage_ok": False, "retry_count": 2}, max_retry=2)
            == BRANCH_FALLBACK
        )


# ---------------------------------------------------------------------------
# 节点：evaluate / persist
# ---------------------------------------------------------------------------


class TestEvaluatePersist:
    def test_evaluate_default_is_l1(self):
        out = _run(
            node_evaluate(
                _state(
                    parsed={"used_sentence_ids": ["s_1", "s_2"], "turns": [{"speaker": "A"}, {"speaker": "B"}]},
                    required_ids=REQUIRED_IDS,
                )
            )
        )
        assert out["coverage"]["ratio"] == 1.0
        assert out["metrics"]["level"] == "l1"
        assert out["metrics"]["judge_model"] is None

    def test_evaluate_injected_evaluator_wins(self):
        async def evaluator(parsed, coverage):
            return {"score": 42, "level": "l2", "judge_model": "fake"}

        out = _run(
            node_evaluate(
                _state(
                    parsed={"used_sentence_ids": ["s_1", "s_2"], "turns": [{"speaker": "A"}]},
                    required_ids=REQUIRED_IDS,
                ),
                evaluator=evaluator,
            )
        )
        assert out["metrics"]["level"] == "l2"

    def test_persist_marks_coverage_incomplete(self):
        out = node_persist(
            _state(
                parsed={
                    "content_type": "dialogue",
                    "used_sentence_ids": ["s_1"],
                    "turns": [{"speaker": "A", "text": "hi"}],
                    "notes": None,
                },
                required_ids=REQUIRED_IDS,
                roles=[],
                metrics={"level": "l1"},
                # v2 正常对话路径：select_best 已选中候选
                selected={"learning_sentence_id": "s_1"},
            )
        )
        result = out["result"]
        assert result["coverage"]["required_used"] == 1
        assert result["notes"] == "coverage_incomplete"
        assert result["checkpoint"] is None

    def test_persist_merges_fallback_note_once(self):
        out = node_persist(
            _state(
                parsed={
                    "content_type": "non_dialogue",
                    "used_sentence_ids": ["s_1", "s_2"],
                    "turns": [{"speaker": "T", "text": "retell"}],
                    "notes": "fallback_non_dialogue",
                },
                required_ids=REQUIRED_IDS,
                roles=[],
                fallback=True,
                context_notes=["recall_insufficient"],
            )
        )
        assert out["result"]["notes"] == "recall_insufficient,fallback_non_dialogue"


# ---------------------------------------------------------------------------
# 全图流程
# ---------------------------------------------------------------------------


class TestRunDialogueGenGraph:
    def test_first_try_coverage_complete(self):
        async def gen(messages):
            return _llm_json()

        result = _run(
            run_dialogue_gen_graph(
                db=None, task_id="dg_1", scholar_id="s", context=_context(), generator=_v2_flow_gen(gen)
            )
        )
        assert result["content_type"] == "dialogue"
        assert result["coverage"] == {
            "required_total": 2,
            "required_used": 2,
            "ratio": 1.0,
        }
        assert result["retry_count"] == 0
        assert result["notes"] is None

    def test_refine_then_success(self):
        calls = {"n": 0}

        async def gen(messages):
            calls["n"] += 1
            if calls["n"] == 1:
                return _llm_json(used=("s_1",))
            # 第二次生成的 prompt 应回灌缺失点
            assert "上一轮遗漏" in messages[1]["content"]
            return _llm_json()

        result = _run(
            run_dialogue_gen_graph(
                db=None, task_id="dg_1", scholar_id="s", context=_context(), generator=_v2_flow_gen(gen)
            )
        )
        assert calls["n"] == 2
        assert result["retry_count"] == 1
        assert result["coverage"]["required_used"] == 2
        assert result["notes"] is None

    def test_retry_exhausted_falls_back_to_non_dialogue(self):
        calls = {"n": 0}

        async def gen(messages):
            calls["n"] += 1
            return _llm_json(used=("s_1",))  # 永不覆盖 s_2

        result = _run(
            run_dialogue_gen_graph(
                db=None,
                task_id="dg_1",
                scholar_id="s",
                context=_context(),
                generator=_v2_flow_gen(gen),
                max_retry=2,
            )
        )
        assert calls["n"] == 3  # 1 + max_retry（降级不额外调 LLM）
        assert result["content_type"] == "non_dialogue"
        assert result["sub_type"] == "retell"
        assert result["retry_count"] == 2
        assert "fallback_non_dialogue" in (result["notes"] or "")
        # 降级把全部必用句纳入题面 → 覆盖完整
        assert result["coverage"]["required_used"] == 2

    def test_zero_retry_falls_back_immediately(self):
        calls = {"n": 0}

        async def gen(messages):
            calls["n"] += 1
            return _llm_json(used=("s_1",))

        result = _run(
            run_dialogue_gen_graph(
                db=None,
                task_id="dg_1",
                scholar_id="s",
                context=_context(),
                generator=_v2_flow_gen(gen),
                max_retry=0,
            )
        )
        assert calls["n"] == 1
        assert result["content_type"] == "non_dialogue"

    def test_recall_ids_pass_through_to_result(self):
        retriever = FakeCurriculumRetriever(
            [
                {"sentence_id": "r1", "text": "old one", "lesson_id": "l2"},
                {"sentence_id": "r2", "text": "old two", "lesson_id": "l2"},
            ]
        )

        async def gen(messages):
            assert "召回学习语句" in messages[1]["content"]
            return _llm_json(recalled=("r1", "r_fake"))

        result = _run(
            run_dialogue_gen_graph(
                db=None,
                task_id="dg_1",
                scholar_id="s",
                context=_context(recall={"enabled": True, "top_k": 4}),
                retriever=retriever,
                generator=_v2_flow_gen(gen),
            )
        )
        # 召回白名单过滤掉编造 id
        assert result["recalled_sentence_ids"] == ["r1"]
        assert result["used_sentence_ids"] == ["s_1", "s_2"]

    def test_empty_llm_raises_unavailable(self):
        async def gen(messages):
            return None

        with pytest.raises(DialogueGenError) as exc:
            _run(
                run_dialogue_gen_graph(
                    db=None, task_id="dg_1", scholar_id="s", context=_context(), generator=_v2_flow_gen(gen)
                )
            )
        assert exc.value.error_code == ERR_LLM_UNAVAILABLE

    def test_graph_shape_has_expected_nodes(self):
        graph = build_dialogue_gen_graph()
        nodes = set(graph.nodes)
        assert {
            "load_context",
            "build_prompt",
            "generate_dialogue",
            "validate_coverage",
            "refine_prompt",
            "fallback_non_dialogue",
            "evaluate",
            "persist",
        } <= nodes


# ---------------------------------------------------------------------------
# T4：断点续写（checkpoint 落库 / 轨迹 / 续跑复用）
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    """T4 断点续写：独立集合落 checkpoint + 续跑复用已完成节点（§4.3.1）。"""

    @staticmethod
    async def _gen_ok(messages):
        return _llm_json(used=("s_1", "s_2"))

    def test_checkpoints_written_to_dedicated_collection(self, fake_db):
        saver = get_checkpointer(fake_db)
        result = _run(
            run_dialogue_gen_graph(
                db=fake_db,
                task_id="dg_cp",
                scholar_id="s",
                context=_context(),
                generator=_v2_flow_gen(self._gen_ok),
                checkpointer=saver,
            )
        )
        docs = fake_db.all(DIALOGUE_CHECKPOINT_COLLECTION)
        assert docs  # 每节点落一篇
        # 与会话面集合严格隔离（§4.3.1）
        assert fake_db.all("conversation_graph_checkpoint") == []
        assert result["checkpoint"]["thread_id"] == "dg_cp"
        assert result["checkpoint"]["checkpoint_id"]
        assert result["checkpoint"]["resumable"] is False  # 终态不可续写

    def test_list_checkpoints_old_to_new_with_stages(self, fake_db):
        saver = get_checkpointer(fake_db)
        _run(
            run_dialogue_gen_graph(
                db=fake_db,
                task_id="dg_list",
                scholar_id="s",
                context=_context(),
                generator=_v2_flow_gen(self._gen_ok),
                checkpointer=saver,
            )
        )
        items = _run(list_dialogue_checkpoints(fake_db, "dg_list"))
        assert len(items) >= 5
        stages = [i["stage"] for i in items]
        assert stages[0] == "start"  # 入口 checkpoint 归一
        assert "context_loaded" in stages
        assert stages[-1] == "persisted"
        assert all(isinstance(i["retry_count"], int) for i in items)
        ts = [i["ts"] for i in items]  # 旧→新：ts 单调不减
        assert ts == sorted(ts)

    def test_latest_and_exists_helpers(self, fake_db):
        saver = get_checkpointer(fake_db)
        _run(
            run_dialogue_gen_graph(
                db=fake_db,
                task_id="dg_latest",
                scholar_id="s",
                context=_context(),
                generator=_v2_flow_gen(self._gen_ok),
                checkpointer=saver,
            )
        )
        latest = _run(latest_checkpoint_id(fake_db, "dg_latest"))
        assert latest
        assert _run(dialogue_checkpoint_exists(fake_db, "dg_latest")) is True
        assert _run(dialogue_checkpoint_exists(fake_db, "dg_latest", latest)) is True
        assert _run(dialogue_checkpoint_exists(fake_db, "dg_none")) is False

    def test_resume_reuses_completed_nodes(self, fake_db):
        saver = get_checkpointer(fake_db)
        gen_calls = {"n": 0}

        async def gen(messages):
            gen_calls["n"] += 1
            return _llm_json(used=("s_1", "s_2"))

        eval_calls = {"n": 0}

        async def flaky_eval(parsed, coverage):
            eval_calls["n"] += 1
            if eval_calls["n"] == 1:
                raise RuntimeError("evaluator boom")
            return {"score": 88, "level": "l3", "judge_model": "fake"}

        # 首次执行在 evaluate 节点中断 → 已完成的生成节点落 checkpoint
        with pytest.raises(RuntimeError):
            _run(
                run_dialogue_gen_graph(
                    db=fake_db,
                    task_id="dg_reuse",
                    scholar_id="s",
                    context=_context(),
                    generator=_v2_flow_gen(gen),
                    evaluator=flaky_eval,
                    checkpointer=saver,
                )
            )
        assert gen_calls["n"] == 1
        assert _run(latest_checkpoint_id(fake_db, "dg_reuse"))

        # 续写：从最近 checkpoint 接续，不重跑 load/build/generate/validate
        result = _run(
            run_dialogue_gen_graph(
                db=fake_db,
                task_id="dg_reuse",
                scholar_id="s",
                context=_context(),
                generator=_v2_flow_gen(gen),
                evaluator=flaky_eval,
                checkpointer=saver,
                resume=True,
            )
        )
        assert result["content_type"] == "dialogue"
        assert result["metrics"]["level"] == "l3"
        assert gen_calls["n"] == 1  # 生成节点未二次调用 LLM

    def test_resume_without_checkpoint_reruns_with_note(self, fake_db):
        saver = get_checkpointer(fake_db)
        result = _run(
            run_dialogue_gen_graph(
                db=fake_db,
                task_id="dg_no_cp",
                scholar_id="s",
                context=_context(),
                generator=_v2_flow_gen(self._gen_ok),
                checkpointer=saver,
                resume=True,
            )
        )
        assert result["content_type"] == "dialogue"
        assert NOTE_RESUME_WITHOUT_CHECKPOINT in (result["notes"] or "")


# ---------------------------------------------------------------------------
# v2：每句评估（dispatch / evaluate_per_sentence）
# ---------------------------------------------------------------------------


def _per_sentence_json(viable: bool, *, sid: str = "s_1", naturalness: float = 0.85) -> str:
    """单句评估输出（viable=True 时含 turns/naturalness；False 时仅 reason）。"""
    if not viable:
        return json.dumps({"viable": False, "reason": "定义句无法进入对话"}, ensure_ascii=False)
    return json.dumps(
        {
            "viable": True,
            "recalled_used": [],
            "naturalness": naturalness,
            "turns": [
                {"speaker": "A", "text": "Last week I went to the theatre.", "target_sentence_id": sid},
                {"speaker": "B", "text": "Oh, sounds nice!", "target_sentence_id": None},
            ],
            "reason": "past event",
        },
        ensure_ascii=False,
    )


class TestPerSentenceEvaluate:
    """v2：dispatch_per_sentence 扇出 + node_evaluate_per_sentence 单句评估。"""

    def test_dispatch_returns_send_per_required_id(self):
        state = _state(required_ids=list(REQUIRED_IDS))
        sends = dispatch_per_sentence(state, parallel_enabled=False)
        assert [s.node for s in sends] == ["evaluate_per_sentence"] * len(REQUIRED_IDS)
        assert [s.arg["_learning_sentence_id"] for s in sends] == list(REQUIRED_IDS)
        assert [s.arg["_branch_index"] for s in sends] == [0, 1]

    async def _gen_for(self, payload_by_sid):
        async def fake(messages):
            import re

            m = re.search(r"\[([^\]]+)\]", messages[1]["content"])
            sid = m.group(1) if m else ""
            return payload_by_sid.get(sid, json.dumps({"viable": False, "reason": "unknown"}))

        return fake

    def test_evaluate_viable_returns_candidate_without_stage(self):
        state = _state(
            context=_context(),
            required_ids=["s_1"],
            recall_ids=[],
            recalled=[],
            _learning_sentence_id="s_1",
        )
        fake = asyncio.run(self._gen_for({"s_1": _per_sentence_json(True, sid="s_1")}))
        out = _run(node_evaluate_per_sentence(state, generator=fake))
        assert list(out.keys()) == ["candidate_dialogues"]  # 不写 stage（扇出约束）
        cand = out["candidate_dialogues"][0]
        assert cand["viable"] is True
        assert cand["learning_sentence_id"] == "s_1"
        assert cand["naturalness"] == 0.85

    def test_evaluate_unviable_returns_candidate_with_viable_false(self):
        state = _state(
            context=_context(),
            required_ids=["s_2"],
            recall_ids=[],
            recalled=[],
            _learning_sentence_id="s_2",
        )
        fake = asyncio.run(self._gen_for({"s_2": _per_sentence_json(False)}))
        out = _run(node_evaluate_per_sentence(state, generator=fake))
        assert out["candidate_dialogues"][0]["viable"] is False

    def test_evaluate_parse_failure_raises(self):
        state = _state(
            context=_context(),
            required_ids=["s_1"],
            recall_ids=[],
            recalled=[],
            _learning_sentence_id="s_1",
        )
        async def fake(messages):
            return "not a json"

        with pytest.raises(DialogueGenError) as exc:
            _run(node_evaluate_per_sentence(state, generator=fake))
        assert exc.value.error_code == ERR_LLM_PARSE_ERROR

    def test_evaluate_unknown_sentence_returns_empty(self):
        state = _state(
            context=_context(),
            required_ids=[],
            recall_ids=[],
            recalled=[],
            _learning_sentence_id="s_missing",
        )
        async def fake(messages):  # pragma: no cover
            raise AssertionError("未知句子不应触发 LLM")

        out = _run(node_evaluate_per_sentence(state, generator=fake))
        assert out["candidate_dialogues"] == []


# ---------------------------------------------------------------------------
# v2：select_best / summarize 节点与路由
# ---------------------------------------------------------------------------


def _viable_candidate(sid: str, *, naturalness: float, turn_count: int = 2) -> dict:
    return {
        "viable": True,
        "learning_sentence_id": sid,
        "recalled_used": [],
        "naturalness": naturalness,
        "turns": [
            {"speaker": "A", "text": f"line {i}", "target_sentence_id": sid if i == 0 else None}
            for i in range(turn_count)
        ],
        "reason": "ok",
    }


class TestSelectBestNode:
    """v2：node_select_best / route_after_select_best。"""

    def test_select_best_picks_highest_naturalness(self):
        state = _state(
            candidate_dialogues=[
                _viable_candidate("s_1", naturalness=0.5),
                _viable_candidate("s_2", naturalness=0.9),
            ]
        )
        out = node_select_best(state)
        assert out["stage"] == STAGE_SELECTED
        assert out["selected"]["learning_sentence_id"] == "s_2"
        assert out["selection"]["selected_index"] == 1

    def test_select_best_none_viable_returns_empty_selection(self):
        state = _state(
            candidate_dialogues=[
                {"viable": False, "learning_sentence_id": "s_1", "reason": "x"}
            ]
        )
        out = node_select_best(state)
        assert out["stage"] == STAGE_SELECTED
        assert out["selected"] == {}
        assert out["selected_index"] is None
        assert out["selection"]["selected_index"] is None

    def test_route_with_selected_goes_summarize(self):
        assert route_after_select_best({"selected": {"learning_sentence_id": "s_1"}}) == (
            BRANCH_SUMMARIZE
        )

    def test_route_without_selected_goes_fallback(self):
        assert route_after_select_best({"selected": {}}) == BRANCH_SELECT_FALLBACK


def _summary_json() -> str:
    return json.dumps(
        {
            "background": "周六下午，Tom 和 Lily 在咖啡店讨论周末计划。",
            "roles": [
                {"code": "A", "name": "Tom", "identity": "student"},
                {"code": "B", "name": "Lily", "identity": "classmate"},
            ],
        },
        ensure_ascii=False,
    )


class TestSummarizeNode:
    """v2：node_summarize / route_after_summarize。"""

    def test_summarize_writes_background_and_roles(self):
        state = _state(selected=_viable_candidate("s_1", naturalness=0.9))
        async def fake(messages):
            assert "反推" in messages[0]["content"]
            return _summary_json()

        out = _run(node_summarize(state, generator=fake))
        assert out["stage"] == STAGE_SUMMARIZED
        assert out["summary"]["background"].startswith("周六下午")
        assert [r["code"] for r in out["summary"]["roles"]] == ["A", "B"]

    def test_summarize_failure_returns_none_summary(self):
        state = _state(selected=_viable_candidate("s_1", naturalness=0.9))
        async def fake(messages):
            return "not a json"

        out = _run(node_summarize(state, generator=fake))
        assert out["stage"] == STAGE_SUMMARIZED
        assert out["summary"] is None

    def test_route_with_summary_goes_build_prompt(self):
        assert route_after_summarize({"summary": {"background": "x", "roles": []}}) == (
            BRANCH_BUILD_PROMPT
        )

    def test_route_without_summary_goes_fallback(self):
        assert route_after_summarize({"summary": None}) == BRANCH_SUMMARIZE_FALLBACK


# ---------------------------------------------------------------------------
# v2：整图流程（每句评估 → 选优 → 总结 → 生成 → 校验 → 持久化）
# ---------------------------------------------------------------------------


class TestV2FlowGraph:
    """v2：整图接线与端到端 flow（result 落 summary，不含 v1 scene 字段）。"""

    def test_graph_nodes_contain_v2_pipeline(self):
        nodes = set(build_dialogue_gen_graph().nodes)
        for node in ("load_context", "evaluate_per_sentence", "select_best", "summarize",
                     "build_prompt", "generate_dialogue", "validate_coverage",
                     "refine_prompt", "fallback_non_dialogue", "evaluate", "persist"):
            assert node in nodes, node
        assert "classify_scenes" not in nodes

    def test_v2_flow_writes_summary_and_excludes_v1_keys(self):
        async def fake(messages):
            sys_msg = messages[0]["content"]
            if "能否在真实人际对话里自然地用出" in sys_msg or "单句评估" in sys_msg:
                import re

                m = re.search(r"\[([^\]]+)\]", messages[1]["content"])
                sid = m.group(1) if m else "s_1"
                return _per_sentence_json(True, sid=sid, naturalness=0.8)
            if "反推" in sys_msg:
                return _summary_json()
            return _llm_json()

        result = _run(
            run_dialogue_gen_graph(
                db=None,
                task_id="dg_v2",
                scholar_id="s",
                context=_context(),
                generator=fake,
            )
        )
        assert result["content_type"] == "dialogue"
        assert result["summary"]["background"].startswith("周六下午")
        assert [r["code"] for r in result["summary"]["roles"]] == ["A", "B"]
        # 角色使用 summary 产出（不再依赖请求预填）
        assert [r["code"] for r in result["roles"]] == ["A", "B"]
        for removed in ("scene_candidates", "selection", "generation"):
            assert removed not in result, removed

    def test_v2_flow_no_viable_falls_back_with_note(self):
        async def fake(messages):
            if "单句评估" in messages[0]["content"] or "能否在真实人际对话里" in messages[0]["content"]:
                return _per_sentence_json(False)
            raise AssertionError("无 viable 候选应直接 fallback，不再调用生成 LLM")

        result = _run(
            run_dialogue_gen_graph(
                db=None,
                task_id="dg_v2_none",
                scholar_id="s",
                context=_context(),
                generator=fake,
            )
        )
        assert result["content_type"] == "non_dialogue"
        assert NOTE_NO_VIABLE_CANDIDATE in (result["notes"] or "")

    def test_v2_flow_summarize_failed_falls_back_with_note(self):
        async def fake(messages):
            sys_msg = messages[0]["content"]
            if "单句评估" in sys_msg or "能否在真实人际对话里" in sys_msg:
                import re as _re

                m = _re.search(r"\[([^\]]+)\]", messages[1]["content"])
                return _per_sentence_json(True, sid=m.group(1) if m else "s_1", naturalness=0.8)
            if "反推" in sys_msg:
                return "not a json"
            raise AssertionError("summary 失败应走 fallback，不再调用生成 LLM")

        result = _run(
            run_dialogue_gen_graph(
                db=None,
                task_id="dg_v2_sumfail",
                scholar_id="s",
                context=_context(),
                generator=fake,
            )
        )
        assert result["content_type"] == "non_dialogue"
        assert NOTE_SUMMARIZE_FAILED in (result["notes"] or "")
