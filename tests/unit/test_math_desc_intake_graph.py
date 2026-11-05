"""单元测试：数学语音出题语义整编图 services/math/desc_intake_graph.py + providers/desc_intake_llm.py

设计：docs_v1/数学/出题-语音描述出题-LangGraph编排方案.md §4；任务卡 T09~T12。
分段：extract（T09）/ graph（T10）/ assemble（T11）/ flag+fallback（T12）。
红线：R16 永不抛异常、R18 开关关→出参不加字段。
"""
from __future__ import annotations

import asyncio
import json

import pytest

import config
from services.providers import desc_intake_llm as llm


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# T09 build_slot_extract_messages / parse_slot_extract_response
# ===========================================================================


class TestBuildMessages:
    def test_basic_shape(self):
        msgs = llm.build_slot_extract_messages("三年级出五道分数题")
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert "三年级出五道分数题" in msgs[1]["content"]
        # 未知填 null / 同音按语义理解两条硬规则必须出现在 prompt
        assert "禁止编造默认值" in msgs[1]["content"]
        assert "evidence" in msgs[1]["content"]

    def test_candidates_capped_and_rendered(self):
        cands = [{"name": f"考点{i}"} for i in range(80)]
        content = llm.build_slot_extract_messages("出题", kp_candidates=cands)[1]["content"]
        assert "考点0" in content
        assert "考点49" in content
        assert "考点50" not in content  # 截断在 50 个

    def test_context_block(self):
        content = llm.build_slot_extract_messages("出题", context={"grade": 3, "semester": "上"})[1]["content"]
        assert "grade: 3" in content
        assert "semester: 上" in content


class TestParseResponse:
    def test_clean_json(self):
        out = llm.parse_slot_extract_response('{"grade": {"value": 3}}')
        assert out["grade"]["value"] == 3

    def test_markdown_fence(self):
        text = "```json\n{\"grade\": {\"value\": 3}}\n```"
        assert llm.parse_slot_extract_response(text)["grade"]["value"] == 3

    def test_surrounding_noise(self):
        text = '好的，结果如下：{"grade": {"value": 3}} 请查收'
        assert llm.parse_slot_extract_response(text)["grade"]["value"] == 3

    def test_invalid_raises(self):
        with pytest.raises(Exception):
            llm.parse_slot_extract_response("完全不是JSON")
        with pytest.raises(Exception):
            llm.parse_slot_extract_response("[1,2,3]")  # 非 JSON 对象


class TestCallSlotExtract:
    """call_slot_extract 永不抛出（红线 R16）：stub httpx 覆盖成功/失败两路。"""

    def _install(self, monkeypatch, *, content=None, raise_exc=None):
        class _FakeResp:
            def raise_for_status(self):
                if raise_exc:
                    raise raise_exc

            def json(self):
                return {"choices": [{"message": {"content": content}}]}

        class _FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **kw):
                if raise_exc:
                    raise raise_exc
                return _FakeResp()

        monkeypatch.setattr(llm.httpx, "AsyncClient", _FakeClient)

    def test_success(self, monkeypatch):
        slots = {"grade": {"value": 3, "confidence": 0.95, "evidence": "三年级"}}
        self._install(monkeypatch, content=json.dumps(slots, ensure_ascii=False))
        out = _run(llm.call_slot_extract("三年级出五道题"))
        assert out["error"] is None
        assert out["slots"]["grade"]["value"] == 3

    def test_network_error_degrades(self, monkeypatch):
        self._install(monkeypatch, raise_exc=ConnectionError("boom"))
        out = _run(llm.call_slot_extract("三年级出五道题"))
        assert out["slots"] is None
        assert out["error"] and "ConnectionError" in out["error"]

    def test_bad_json_degrades(self, monkeypatch):
        self._install(monkeypatch, content="完全不是JSON")
        out = _run(llm.call_slot_extract("出题"))
        assert out["slots"] is None
        assert out["error"]


# ===========================================================================
# T10/T11/T12 整编图组装 / intent_assemble / run_intake
# ===========================================================================

import config  # noqa: E402
from services.math import desc_intake_graph as dg  # noqa: E402


def _slots_payload(grade=3, semester="上", count=5, kps=None, qtype="应用题", scenario=None, conf=0.9):
    """stub LLM 返回的槽位 JSON（slot_extract 直出口径）。"""
    def _s(value):
        return {"value": value, "confidence": conf, "evidence": "e"}
    return {
        "grade": _s(grade),
        "semester": _s(semester),
        "knowledge_points": _s(kps or ["分数的简单加减"]),
        "question_count": _s(count),
        "question_type": _s(qtype),
        "difficulty_hint": _s(None),
        "scenario": _s(scenario),
        "extended_request": _s(None),
    }


_KP_CANDS = [
    {"name": "分数的简单加减", "grade": 3, "node_code": "3-1-2-kp1"},
    {"name": "小数加减法", "grade": 4, "node_code": "4-1-1-kp1"},
]


def _stub_extract(monkeypatch, payload=None, error=None):
    async def _fake(raw_text, **kw):
        if error is not None:
            return {"slots": None, "error": error}
        return {"slots": payload if payload is not None else _slots_payload(), "error": None}

    monkeypatch.setattr(dg, "call_slot_extract", _fake)


class TestIntakeGraphAssembly:
    def test_graph_compiled(self):
        g = dg._get_compiled_graph()
        assert g is dg._get_compiled_graph()  # 缓存复用

    def test_happy_path_full_pipeline(self, monkeypatch):
        _stub_extract(monkeypatch, _slots_payload(scenario="分蛋糕"))
        out = _run(dg.run_intake("三年级上册分数加减出五道应用题分蛋糕", kp_candidates=_KP_CANDS))
        assert out["fallback"] is False
        assert out["raw_text_sanitized"] == "三年级上册分数加减出五道应用题分蛋糕"
        assert out["slots"]["grade"]["value"] == 3
        assert out["slots"]["knowledge_points"]["value"] == ["分数的简单加减"]  # grounding 命中
        assert out["intake_meta"]["completeness"] == "complete"
        assert out["canonical_text"] == "3年级上 · 分数的简单加减 · 5道应用题（分蛋糕情境）"

    def test_scene_repeated_speech_absorbed(self, monkeypatch):
        # 场景①：整句重复 → sanitize 折叠；槽位只取一次值
        _stub_extract(monkeypatch, _slots_payload())
        out = _run(dg.run_intake("我想要分数加减法我想要分数加减法，出五道"))
        assert out["fallback"] is False
        assert out["raw_text_sanitized"] == "我想要分数加减法，出五道"
        assert out["intake_meta"]["completeness"] == "complete"

    def test_scene_homophone_noise(self, monkeypatch):
        # 场景②：同音错词在语义层吸收（evidence 保留原话），grounding 归一 kp 名
        _stub_extract(monkeypatch, _slots_payload(kps=["分数的简单家减"]))
        out = _run(dg.run_intake("三年级家减法出五道", kp_candidates=_KP_CANDS))
        assert out["fallback"] is False
        # 模糊匹配命中 → kp 归一为候选名
        assert out["slots"]["knowledge_points"]["value"] == ["分数的简单加减"]
        # 范式断言：出参不含 edits 类字段（不修文本，语义整编）
        assert not any("edit" in k for k in out)

    def test_scene_truncated_missing_required(self, monkeypatch):
        # 场景③：截断 → 必填槽缺失 → missing_required（追问交上层），不猜
        _stub_extract(monkeypatch, _slots_payload(kps=["分数的简单加减"], count=None))
        out = _run(dg.run_intake("三年级分数加", kp_candidates=_KP_CANDS))
        assert out["fallback"] is False
        assert out["intake_meta"]["completeness"] == "missing_required"
        assert "question_count" in out["intake_meta"]["missing_required"]
        assert "S3_REQUIRED_MISSING" in out["intake_meta"]["issues"]


class TestIntakeFallback:
    def test_llm_error_fallback(self, monkeypatch):
        _stub_extract(monkeypatch, error="ConnectionError: boom")
        out = _run(dg.run_intake("三年级出五道题"))
        assert out["fallback"] is True
        assert out["fallback_reason"].startswith("ConnectionError")
        assert out["intent"] is None
        assert out["intent_source"] == "fallback"
        assert out["raw_text_sanitized"]  # 原文（sanitize 后）保留供 fallback 路径

    def test_never_raises_on_internal_error(self, monkeypatch):
        async def _boom(raw_text, **kw):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(dg, "call_slot_extract", _boom)
        out = _run(dg.run_intake("出题"))
        assert out["fallback"] is True

    def test_flag_off_returns_none(self, monkeypatch):
        monkeypatch.setattr(config, "DESC_INTAKE_ENABLED", False)
        assert _run(dg.run_intake("出题")) is None

    def test_flag_off_no_fields(self, monkeypatch):
        # R18：开关关 → 出参为 None（调用方不加字段），非空壳 dict
        monkeypatch.setattr(config, "DESC_INTAKE_ENABLED", False)
        out = _run(dg.run_intake("出题"))
        assert out is None and not isinstance(out, dict)


class TestIntentAssemble:
    def _validation_meta(self, completeness="complete", low=None, issues=None):
        return {
            "slot_confidence": 0.9,
            "completeness": completeness,
            "missing_required": [],
            "issues": issues or [],
            "not_grounded_kps": [],
            "low_confidence_slots": low or [],
            "dropped_slots": [],
            "context_filled": [],
        }

    def test_fast_path(self):
        slots = _slots_payload()
        intent, source = dg.intent_assemble(slots, self._validation_meta(), "3年级上 · 分数 · 5道")
        assert source == "slots"
        assert intent["knowledge_points"] == ["分数的简单加减"]
        assert intent["question_count"] == 5
        assert intent["grade"] == 3
        assert intent["semester"] == "上"
        assert intent["question_type"] == "应用题"

    def test_missing_required_slow_path(self):
        slots = _slots_payload(count=None)
        intent, source = dg.intent_assemble(slots, self._validation_meta(completeness="missing_required"), "")
        assert intent is None and source == "llm"

    def test_low_confidence_required_slow_path(self):
        slots = _slots_payload(count=5, conf=0.3)
        meta = self._validation_meta(low=["question_count"])
        intent, source = dg.intent_assemble(slots, meta, "")
        assert intent is None and source == "llm"

    def test_issues_slow_path(self):
        slots = _slots_payload()
        intent, source = dg.intent_assemble(slots, self._validation_meta(issues=["S2_KP_NOT_GROUNDED"]), "")
        assert intent is None and source == "llm"

    def test_nulls_not_fabricated(self):
        slots = _slots_payload(qtype=None, scenario=None)
        intent, _ = dg.intent_assemble(slots, self._validation_meta(), "")
        assert "question_type" not in intent  # 未提及的槽不进 intent
        assert "scenario" not in intent
