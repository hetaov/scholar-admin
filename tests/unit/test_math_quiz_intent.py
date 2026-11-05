"""单元测试：语音描述出题契约接线（方案 §7.1/§7.2；任务卡 T19/T20/T21）

- T19 quiz_intent.parseQuizIntent：快/慢双路径 + grounding + 审计
- T20 quiz_desc.generateQuizFromDescription：图路径/单步直出 + 落库 + 出参投影（R3）
- T21 A/B 对照：同 fixture 开/关 `USE_LANGGRAPH_DESC_QUIZ` 出参一致（R18）

LLM 一律 stub（monkeypatch `_call_chat`）。
"""
from __future__ import annotations

import asyncio
import json

import pytest

import config
from services.math import quiz_desc as qd
from services.math import quiz_intent as qi
from services.math.quiz_intent import parseQuizIntent
from services.database import CURRICULUM_NODE_COLLECTION
from tests.fakes.fake_db import FakeDB


def _run(coro):
    return asyncio.run(coro)


def _stub_chat(monkeypatch, responses, module=None):
    """monkeypatch `module._call_chat`：按序弹响应；Exception 项则抛出。默认意图模块。"""
    module = module or qi
    queue = list(responses)
    calls: list[dict] = []

    async def _fake(*, system, user):
        calls.append({"system": system, "user": user})
        item = queue.pop(0) if queue else None
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(module, "_call_chat", _fake)
    return calls


def _db() -> FakeDB:
    return FakeDB(
        seed={
            CURRICULUM_NODE_COLLECTION: [
                {
                    "code": "3-1-2-kp1",
                    "title": "分数的简单加减",
                    "grade": 3,
                    "ai_summary": {"summary": "同分母分数加减的算理"},
                },
                {
                    "code": "3-2-1-kp2",
                    "title": "单位换算",
                    "grade": 3,
                    "ai_summary": {"summary": "元角分与米厘米"},
                },
            ]
        }
    )


def _intent_json(kps="分数的简单加减", count=5):
    return json.dumps(
        {
            "knowledge_points": [kps] if kps else [],
            "question_count": count,
            "question_type_hint": "应用题",
            "difficulty_hint": None,
            "scenario": None,
            "include_extended": False,
        },
        ensure_ascii=False,
    )


def _items_text(count=5, prefix="分数应用题："):
    items = [
        {
            "question": f"{prefix}第{i}题 3/7 + {i}/7 = ?",
            "answer": f"{3 + i}/7",
            "answer_type": "fraction",
            "difficulty": 2,
            "knowledge_point_name": "分数的简单加减",
            "rationale": "考察同分母分数加减。",
            "target_error_step": 2 if i < 3 else 0,
        }
        for i in range(1, count + 1)
    ]
    return json.dumps({"items": items}, ensure_ascii=False)


def _good_card():
    return json.dumps(
        {
            "core_rationale": "同分母分数加减",
            "step_sequence": ["提取分数", "判断分母", "分子相加", "约分"],
            "givent_unknown": {"givens": ["x"], "unknown": "y"},
            "error_prone_steps": [{"step": 3, "trap": "分母相加", "weight": 0.6}],
            "difficulty_band": 2,
            "variant_axes": ["数值"],
        },
        ensure_ascii=False,
    )


# ===========================================================================
# T19 parseQuizIntent
# ===========================================================================


class TestParseIntentResponse:
    def test_basic(self):
        out = qi.parse_intent_response(_intent_json())
        assert out["knowledge_points"] == ["分数的简单加减"]
        assert out["question_count"] == 5

    def test_kp_string_coerced(self):
        out = qi.parse_intent_response('{"knowledge_points": "分数"}')
        assert out["knowledge_points"] == ["分数"]

    def test_invalid_raises(self):
        with pytest.raises(Exception):
            qi.parse_intent_response("垃圾")


class TestParseQuizIntent:
    def test_fast_path_no_llm(self, monkeypatch):
        calls = _stub_chat(monkeypatch, [RuntimeError("不应调用 LLM")])
        db = _db()
        out = _run(
            parseQuizIntent(
                db,
                scholar_id="sch_1",
                intake_meta={
                    "fallback": False,
                    "slot_confidence": 0.93,
                    "canonical_text": "3年级上 · 分数 · 5道",
                    "missing_required": [],
                    "intent": {
                        "knowledge_points": ["分数的简单加减"],
                        "question_count": 5,
                        "question_type_hint": "应用题",
                    },
                },
            )
        )
        assert calls == []  # 快路径零 LLM
        assert out["fallback"] is False
        assert out["intent"]["question_count"] == 5
        assert out["grounded_kps"][0]["node_code"] == "3-1-2-kp1"
        assert out["intent_confidence"] == 0.93

    def test_slow_path_llm(self, monkeypatch):
        calls = _stub_chat(monkeypatch, [_intent_json()])
        out = _run(parseQuizIntent(_db(), scholar_id="sch_1", asr_raw_text="三年级出五道分数题"))
        assert len(calls) == 1
        assert out["fallback"] is False
        assert out["intent"]["knowledge_points"] == ["分数的简单加减"]
        assert out["grounded_kps"]  # grounding 命中

    def test_slow_path_prefers_canonical_text(self, monkeypatch):
        calls = _stub_chat(monkeypatch, [_intent_json()])
        _run(
            parseQuizIntent(
                _db(),
                scholar_id="sch_1",
                asr_raw_text="原文",
                intake_meta={"fallback": True, "canonical_text": "规范描述"},
            )
        )
        assert "规范描述" in calls[0]["user"]  # 慢路径输入用 canonical_text

    def test_llm_failure_fallback(self, monkeypatch):
        _stub_chat(monkeypatch, [ConnectionError("boom")])
        out = _run(parseQuizIntent(_db(), scholar_id="sch_1", asr_raw_text="出题"))
        assert out["fallback"] is True
        assert out["intent"]["knowledge_points"] == []

    def test_grounding_miss_keeps_intent(self, monkeypatch):
        _stub_chat(monkeypatch, [_intent_json(kps="不存在的考点")])
        out = _run(parseQuizIntent(_db(), scholar_id="sch_1", asr_raw_text="出题"))
        assert out["grounded_kps"] == []  # 未定位
        assert out["intent"]["knowledge_points"] == ["不存在的考点"]  # 意图保留

    def test_fast_path_partial_fills_via_llm(self, monkeypatch):
        # 快路径缺 question_count → 慢路径补齐
        _stub_chat(monkeypatch, [_intent_json(count=7)])
        out = _run(
            parseQuizIntent(
                _db(),
                scholar_id="sch_1",
                asr_raw_text="出题",
                intake_meta={
                    "fallback": False,
                    "intent": {"knowledge_points": ["分数的简单加减"]},
                },
            )
        )
        assert out["intent"]["question_count"] == 7
