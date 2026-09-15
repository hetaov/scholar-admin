"""单元测试：对话生成域纯函数与直连生成（设计稿 §4.4 / §4.5 / §4.6）

纯函数/注入 fake generator，不触网不连库：
- clamp_top_k / normalize_roles / task_group_sentence_ids / choose_content_type
- parse_dialogue_output（合法/代码块包裹/非法形态/空 turns/子形态缺省/id 过滤）
- compute_coverage / basic_metrics
- generate_dialogue（成功 / 空输出 / 解析失败 / 无句子 / 超时）
"""
from __future__ import annotations

import asyncio
import json

import pytest

from services.providers.dialogue_gen import (
    DEFAULT_SELECT_STRATEGY,
    ERR_COVERAGE_FAILED,
    ERR_LLM_PARSE_ERROR,
    ERR_LLM_TIMEOUT,
    ERR_LLM_UNAVAILABLE,
    DialogueGenError,
    basic_metrics,
    build_dialogue_messages,
    build_per_sentence_evaluate_messages,
    build_result,
    build_summarize_messages,
    choose_content_type,
    clamp_top_k,
    compute_coverage,
    generate_dialogue,
    normalize_roles,
    parse_dialogue_output,
    parse_per_sentence_evaluation,
    parse_summary,
    resolve_target_sentence,
    select_best,
    task_group_sentence_ids,
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
        "recall": {"enabled": True, "top_k": 4},
        "metrics": {"enabled": True, "weak_skills": ["past_tense"]},
        "prompt_lang": "zh",
        "preferred_type": "auto",
    }
    ctx.update(overrides)
    return ctx


def _llm_json(content_type: str = "dialogue", **overrides) -> str:
    body = {
        "content_type": content_type,
        "sub_type": None,
        "background_intro": "背景介绍",
        "turns": [
            {"speaker": "A", "text": "Last week I went to the theatre.", "target_sentence_id": "s_1"},
            {"speaker": "B", "text": "I had a very good seat.", "target_sentence_id": "s_2"},
        ],
        "prompts": [{"after_turn": 1, "lang": "zh", "text": "试着复述一下"}],
        "used_sentence_ids": ["s_1", "s_2"],
        "recalled_sentence_ids": [],
        "difficulty": 2,
        "notes": None,
    }
    body.update(overrides)
    return json.dumps(body, ensure_ascii=False)


class TestHelpers:
    def test_clamp_top_k(self):
        assert clamp_top_k(1) == 2
        assert clamp_top_k(4) == 4
        assert clamp_top_k(99) == 6
        assert clamp_top_k(None) == 4
        assert clamp_top_k("abc") == 4

    def test_normalize_roles_drops_invalid(self):
        roles = normalize_roles(
            [
                {"code": "A", "name": "Tom", "identity": "student"},
                {"name": "no code"},
                "junk",
                {"code": "B"},
            ]
        )
        assert roles == [
            {"code": "A", "name": "Tom", "identity": "student"},
            {"code": "B", "name": "B", "identity": ""},
        ]

    def test_task_group_sentence_ids_dedup(self):
        tg = {
            "sentences": [
                {"sentence_id": "s1"},
                {"sentence_id": "s1"},
                {"sentence_id": "s2"},
                {"sentence_id": ""},
            ]
        }
        assert task_group_sentence_ids(tg) == ["s1", "s2"]

    def test_choose_content_type_forces_non_dialogue_without_roles(self):
        assert choose_content_type("auto", [{"code": "A"}]) == "non_dialogue"
        assert choose_content_type("auto", [{"code": "A"}, {"code": "B"}]) == "dialogue"
        assert choose_content_type("non_dialogue", [{"code": "A"}, {"code": "B"}]) == "non_dialogue"

    def test_compute_coverage(self):
        cov = compute_coverage(["s1", "s2"], ["s2", "s3"])
        assert cov == {"required_total": 2, "required_used": 1, "ratio": 0.5}
        assert compute_coverage([], []) == {
            "required_total": 0,
            "required_used": 0,
            "ratio": 0.0,
        }

    def test_basic_metrics_level_is_l1(self):
        metrics = basic_metrics({"ratio": 1.0}, turn_count=4)
        assert metrics["level"] == "l1"
        assert metrics["judge_model"] is None
        assert metrics["score"] == 100
        assert metrics["meaningful"] is True


class TestResolveTargetSentence:
    """T5：用户作答参考句解析（显式 → 结果最后一个目标句 → 任务组首句）。"""

    def test_explicit_hit(self):
        target = resolve_target_sentence(_context(), "s_2")
        assert target == {"sentence_id": "s_2", "content": "I had a very good seat."}

    def test_explicit_miss_returns_none(self):
        assert resolve_target_sentence(_context(), "s_fake") is None

    def test_prefers_last_result_turn(self):
        result = {
            "turns": [
                {"speaker": "A", "text": "x", "target_sentence_id": "s_1"},
                {"speaker": "B", "text": "y", "target_sentence_id": "s_2"},
            ]
        }
        target = resolve_target_sentence(_context(), result=result)
        assert target["sentence_id"] == "s_2"

    def test_ignores_target_not_in_task_group(self):
        result = {"turns": [{"speaker": "A", "text": "x", "target_sentence_id": "s_fake"}]}
        target = resolve_target_sentence(_context(), result=result)
        assert target["sentence_id"] == "s_1"  # 回落任务组首句

    def test_falls_back_to_first_sentence(self):
        target = resolve_target_sentence(_context())
        assert target["sentence_id"] == "s_1"

    def test_empty_task_group_returns_none(self):
        assert resolve_target_sentence({"task_group": {"sentences": []}}) is None


class TestParseDialogueOutput:
    def test_valid_dialogue(self):
        parsed = parse_dialogue_output(
            _llm_json(),
            required_ids=REQUIRED_IDS,
            role_codes=["A", "B"],
            prompt_lang="zh",
        )
        assert parsed["content_type"] == "dialogue"
        assert parsed["sub_type"] is None
        assert len(parsed["turns"]) == 2
        assert parsed["prompts"][0]["lang"] == "zh"
        assert parsed["used_sentence_ids"] == ["s_1", "s_2"]

    def test_code_block_wrapped(self):
        content = "```json\n" + _llm_json() + "\n```"
        parsed = parse_dialogue_output(
            content, required_ids=REQUIRED_IDS, role_codes=["A", "B"], prompt_lang="zh"
        )
        assert parsed is not None
        assert parsed["content_type"] == "dialogue"

    def test_invalid_content_type(self):
        parsed = parse_dialogue_output(
            _llm_json(content_type="poem"),
            required_ids=REQUIRED_IDS,
            role_codes=["A", "B"],
            prompt_lang="zh",
        )
        assert parsed is None

    def test_empty_turns(self):
        parsed = parse_dialogue_output(
            _llm_json(turns=[]),
            required_ids=REQUIRED_IDS,
            role_codes=["A", "B"],
            prompt_lang="zh",
        )
        assert parsed is None

    def test_dialogue_requires_two_turns(self):
        parsed = parse_dialogue_output(
            _llm_json(turns=[{"speaker": "A", "text": "hi"}]),
            required_ids=REQUIRED_IDS,
            role_codes=["A", "B"],
            prompt_lang="zh",
        )
        assert parsed is None

    def test_illegal_speaker_dropped(self):
        content = _llm_json(
            turns=[
                {"speaker": "A", "text": "ok"},
                {"speaker": "Z", "text": "unknown speaker"},
            ]
        )
        parsed = parse_dialogue_output(
            content, required_ids=REQUIRED_IDS, role_codes=["A", "B"], prompt_lang="zh"
        )
        # Z 被剔除后只剩 1 轮 → dialogue 不足 2 轮 → 解析失败
        assert parsed is None

    def test_non_dialogue_sub_type_defaults_to_retell(self):
        content = _llm_json(content_type="non_dialogue", sub_type=None)
        parsed = parse_dialogue_output(
            content, required_ids=REQUIRED_IDS, role_codes=[], prompt_lang="zh"
        )
        assert parsed["content_type"] == "non_dialogue"
        assert parsed["sub_type"] == "retell"

    def test_non_dialogue_sub_type_kept(self):
        content = _llm_json(content_type="non_dialogue", sub_type="fill")
        parsed = parse_dialogue_output(
            content, required_ids=REQUIRED_IDS, role_codes=[], prompt_lang="zh"
        )
        assert parsed["sub_type"] == "fill"

    def test_used_ids_filtered_to_required(self):
        content = _llm_json(used_sentence_ids=["s_1", "s_fake"])
        parsed = parse_dialogue_output(
            content, required_ids=REQUIRED_IDS, role_codes=["A", "B"], prompt_lang="zh"
        )
        assert parsed["used_sentence_ids"] == ["s_1"]

    def test_prompt_lang_fallback(self):
        content = _llm_json(prompts=[{"after_turn": 1, "lang": "fr", "text": "提示"}])
        parsed = parse_dialogue_output(
            content, required_ids=REQUIRED_IDS, role_codes=["A", "B"], prompt_lang="en"
        )
        assert parsed["prompts"][0]["lang"] == "en"


class TestGenerateDialogue:
    def test_success(self):
        async def gen(messages):
            return _llm_json()

        result = asyncio.run(generate_dialogue(context=_context(), generator=gen))
        assert result["content_type"] == "dialogue"
        assert result["session_id"].startswith("s_")
        assert result["coverage"] == {
            "required_total": 2,
            "required_used": 2,
            "ratio": 1.0,
        }
        assert result["retry_count"] == 0
        assert result["checkpoint"] is None
        assert result["metrics"]["level"] == "l1"
        assert result["notes"] is None

    def test_coverage_incomplete_notes(self):
        async def gen(messages):
            return _llm_json(used_sentence_ids=["s_1"])

        result = asyncio.run(generate_dialogue(context=_context(), generator=gen))
        assert result["coverage"]["required_used"] == 1
        assert result["notes"] == "coverage_incomplete"

    def test_empty_output_raises_unavailable(self):
        async def gen(messages):
            return None

        with pytest.raises(DialogueGenError) as exc:
            asyncio.run(generate_dialogue(context=_context(), generator=gen))
        assert exc.value.error_code == ERR_LLM_UNAVAILABLE

    def test_bad_json_raises_parse_error(self):
        async def gen(messages):
            return "not a json"

        with pytest.raises(DialogueGenError) as exc:
            asyncio.run(generate_dialogue(context=_context(), generator=gen))
        assert exc.value.error_code == ERR_LLM_PARSE_ERROR

    def test_no_sentences_raises_coverage_failed(self):
        async def gen(messages):  # pragma: no cover
            return _llm_json()

        ctx = _context(task_group={"sentences": []})
        with pytest.raises(DialogueGenError) as exc:
            asyncio.run(generate_dialogue(context=ctx, generator=gen))
        assert exc.value.error_code == ERR_COVERAGE_FAILED

    def test_timeout_raises_llm_timeout(self):
        async def slow_gen(messages):
            await asyncio.sleep(0.2)
            return _llm_json()

        with pytest.raises(DialogueGenError) as exc:
            asyncio.run(
                generate_dialogue(context=_context(), generator=slow_gen, timeout_seconds=0.01)
            )
        assert exc.value.error_code == ERR_LLM_TIMEOUT

    def test_roles_missing_forces_non_dialogue(self):
        async def gen(messages):
            # 无角色 → 形态强制 non_dialogue；提示里 system 已按非对话形态约束
            return _llm_json(
                content_type="non_dialogue",
                sub_type="retell",
                turns=[{"speaker": "N", "text": "情景复述：..."}],
            )

        result = asyncio.run(
            generate_dialogue(context=_context(roles=[]), generator=gen)
        )
        assert result["content_type"] == "non_dialogue"
        assert result["sub_type"] == "retell"


class TestPerSentenceEvaluateContract:
    """v2：单句评估 Prompt 与解析（调整稿 v2 §3.1）。"""

    @staticmethod
    def _viable_json(**overrides) -> str:
        body = {
            "viable": True,
            "recalled_used": ["r_1"],
            "naturalness": 0.85,
            "turns": [
                {"speaker": "A", "text": "I went to the park.", "target_sentence_id": "s_1"},
                {"speaker": "B", "text": "Oh, sounds nice!", "target_sentence_id": None},
            ],
            "reason": "past event",
        }
        body.update(overrides)
        return json.dumps(body, ensure_ascii=False)

    def test_build_per_sentence_messages_renders_learning_and_recall(self):
        ctx = _context()
        messages = build_per_sentence_evaluate_messages(
            context=ctx,
            learning_sentence={"sentence_id": "s_1", "content": "Last week I went to the theatre."},
            recalled=[{"sentence_id": "r_1", "content": "old sentence"}],
        )
        assert messages[0]["role"] == "system"
        user = messages[1]["content"]
        assert "[s_1] Last week I went to the theatre." in user
        assert "[r_1] old sentence" in user
        # 任务组渲染应排除当前学习句本身（v2 exclude_sid），其余句子保留
        assert "s_2" in user

    def test_parse_viable_evaluation(self):
        parsed = parse_per_sentence_evaluation(
            self._viable_json(),
            learning_sentence_id="s_1",
            required_set={"s_1", "s_2"},
            recall_set={"r_1"},
        )
        assert parsed["viable"] is True
        assert parsed["learning_sentence_id"] == "s_1"
        assert parsed["naturalness"] == 0.85
        assert parsed["recalled_used"] == ["r_1"]
        assert len(parsed["turns"]) == 2

    def test_parse_evaluation_rejects_fabricated_ids(self):
        content = self._viable_json(
            turns=[
                {"speaker": "A", "text": "x", "target_sentence_id": "s_fake"},
                {"speaker": "A", "text": "y", "target_sentence_id": "s_1"},
            ],
            recalled_used=["r_fake"],
        )
        # 编造 id → 直接判 None（杜绝编造，不静默过滤）
        assert parse_per_sentence_evaluation(
            content, learning_sentence_id="s_1", required_set={"s_1"}, recall_set={"r_1"}
        ) is None

    def test_parse_unviable_evaluation(self):
        content = json.dumps({"viable": False, "reason": "定义句"}, ensure_ascii=False)
        parsed = parse_per_sentence_evaluation(
            content, learning_sentence_id="s_2", required_set={"s_1", "s_2"}, recall_set=set()
        )
        assert parsed["viable"] is False
        assert parsed["turns"] == []
        assert parsed["naturalness"] == 0.0

    def test_parse_invalid_returns_none(self):
        for bad in (None, "not json", json.dumps({"viable": "yes"}), json.dumps({"viable": True})):
            assert parse_per_sentence_evaluation(
                bad, learning_sentence_id="s_1", required_set={"s_1"}, recall_set=set()
            ) is None, bad


class TestSummarizeContract:
    """v2：背景 + 角色后置总结（build_summarize_messages / parse_summary）。"""

    def test_build_summarize_messages_renders_selected_turns(self):
        selected = {
            "learning_sentence_id": "s_1",
            "turns": [{"speaker": "A", "text": "I went to the park.", "target_sentence_id": "s_1"}],
        }
        messages = build_summarize_messages(
            selected=selected,
            learning_sentence={"sentence_id": "s_1", "content": "I went to the park."},
            recalled=[],
            task_group=_context()["task_group"],
        )
        assert "反推" in messages[0]["content"]
        assert "I went to the park." in messages[1]["content"]

    def test_parse_summary_ok(self):
        content = json.dumps(
            {
                "background": "周六下午，Tom 和 Lily 在咖啡店讨论周末计划。",
                "roles": [
                    {"code": "A", "name": "Tom", "identity": "student"},
                    {"code": "B", "name": "Lily", "identity": "classmate"},
                ],
            },
            ensure_ascii=False,
        )
        summary = parse_summary(content)
        assert summary["background"].startswith("周六下午")
        assert [r["code"] for r in summary["roles"]] == ["A", "B"]

    def test_parse_summary_rejects_bad_roles(self):
        content = json.dumps(
            {
                "background": "背景",
                "roles": [{"code": "D", "name": "X", "identity": "unknown"}],
            },
            ensure_ascii=False,
        )
        assert parse_summary(content) is None

    def test_parse_summary_rejects_empty_background(self):
        content = json.dumps({"background": "", "roles": []}, ensure_ascii=False)
        assert parse_summary(content) is None

    def test_parse_summary_rejects_role_count_out_of_range(self):
        content = json.dumps(
            {
                "background": "背景",
                "roles": [{"code": "A", "name": "X", "identity": ""}],
            },
            ensure_ascii=False,
        )
        assert parse_summary(content) is None


class TestSelectBest:
    """v2：select_best（natural_first / score_first）。"""

    def test_natural_first_prefers_higher_naturalness(self):
        candidates = [
            {"viable": True, "learning_sentence_id": "s_1", "naturalness": 0.6, "turns": []},
            {"viable": True, "learning_sentence_id": "s_2", "naturalness": 0.9, "turns": []},
        ]
        selected, selection = select_best(candidates)
        assert selected["learning_sentence_id"] == "s_2"
        assert selection["selected_index"] == 1
        assert selection["strategy"] == DEFAULT_SELECT_STRATEGY

    def test_score_first_equivalent_to_natural_first(self):
        # v2 当前 score_first 等价 natural_first（预留多维扩展）
        candidates = [
            {"viable": True, "learning_sentence_id": "s_1", "naturalness": 0.9,
             "turns": [{"speaker": "A", "text": "x"}]},
            {"viable": True, "learning_sentence_id": "s_2", "naturalness": 0.5,
             "turns": [{"speaker": "A", "text": "x"}, {"speaker": "B", "text": "y"}]},
        ]
        selected, selection = select_best(candidates, strategy="score_first")
        assert selected["learning_sentence_id"] == "s_1"
        assert selection["strategy"] == "score_first"

    def test_unviable_candidates_excluded(self):
        candidates = [{"viable": False, "learning_sentence_id": "s_1", "reason": "x"}]
        selected, selection = select_best(candidates)
        assert selected is None
        assert selection["selected_index"] is None

    def test_empty_candidates(self):
        selected, selection = select_best([])
        assert selected is None
        assert selection["selected_index"] is None

    def test_invalid_strategy_falls_back_to_default(self):
        candidates = [{"viable": True, "learning_sentence_id": "s_1", "naturalness": 0.5, "turns": []}]
        _, selection = select_best(candidates, strategy="weird")
        assert selection["strategy"] == DEFAULT_SELECT_STRATEGY


class TestV2ResultContract:
    """v2：result 契约（summary 可选落 key；v1 scene 字段已移除）。"""

    def test_build_result_omits_summary_by_default(self):
        parsed = {
            "content_type": "dialogue",
            "sub_type": None,
            "turns": [{"speaker": "A", "text": "x"}],
            "used_sentence_ids": ["s_1"],
            "notes": None,
        }
        result = build_result(parsed, required_ids=["s_1"], roles=[])
        assert "summary" not in result
        for removed in ("scene_candidates", "selection", "generation"):
            assert removed not in result

    def test_build_result_adds_summary_when_provided(self):
        parsed = {
            "content_type": "dialogue",
            "sub_type": None,
            "turns": [{"speaker": "A", "text": "x"}],
            "used_sentence_ids": ["s_1"],
            "notes": None,
        }
        summary = {
            "background": "周六下午在咖啡店讨论周末计划。",
            "roles": [
                {"code": "A", "name": "Tom", "identity": "student"},
                {"code": "B", "name": "Lily", "identity": "classmate"},
            ],
        }
        result = build_result(parsed, required_ids=["s_1"], roles=[], summary=summary)
        assert result["summary"] == summary

    def test_dialogue_messages_render_summary_roles_and_background(self):
        # 背景合并（summary.background → scenario）发生在 graph 层
        # （_scenario_with_summary）；Provider 层只按 context 渲染。
        ctx = _context(scenario={"background": "周六下午在咖啡店。"})
        roles = [
            {"code": "A", "name": "Tom", "identity": "student"},
            {"code": "B", "name": "Lily", "identity": "classmate"},
        ]
        messages = build_dialogue_messages(
            ctx, content_type="dialogue", roles=roles, prompt_lang="zh"
        )
        assert "周六下午在咖啡店。" in messages[1]["content"]
        assert "A = Tom（student）" in messages[1]["content"]

    def test_dialogue_messages_forbid_verbatim_repeats(self):
        """终稿硬约束：禁止逐字重复的追问（防整段套用同一句）。"""
        ctx = _context()
        roles = [
            {"code": "A", "name": "Tom", "identity": "student"},
            {"code": "B", "name": "Lily", "identity": "classmate"},
        ]
        system = build_dialogue_messages(
            ctx, content_type="dialogue", roles=roles, prompt_lang="zh"
        )[0]["content"]
        assert "文本不得逐字重复" in system
        assert "同一句不得在 turns 中出现两次" in system
