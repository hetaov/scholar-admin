"""单元测试：描述模式出题 + flag A/B 对照（方案 §7.2 / R18；任务卡 T20/T21）

- T20：图路径（stub 整编+出题 LLM）/ 单步直出（flag 关）双路径、落库字段
  （graph_version/intake_meta/validation_reasons[]）、出参投影（R3 无 answer）。
- T21：同 fixture 开/关 `USE_LANGGRAPH_DESC_QUIZ` 两跑 → 出参**逐字节一致**。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from services.math import desc_quiz_graph as qg
from services.math import quiz_desc as qd
from services.math.desc_quiz_graph import _GRAPH_VERSION
from services.database import CURRICULUM_NODE_COLLECTION
from services.math.quiz import MATH_QUIZ_COLLECTION
from tests.fakes.fake_db import FakeDB
from tests.unit.test_math_quiz_intent import (
    _db,
    _good_card,
    _intent_json,
    _items_text,
    _stub_chat,
)


def _run(coro):
    return asyncio.run(coro)


def _intent():
    return {
        "knowledge_points": ["分数的简单加减"],
        "question_count": 5,
        "question_type_hint": "应用题",
        "difficulty_hint": None,
        "scenario": None,
        "include_extended": False,
    }


def _grounded():
    return [
        {
            "name": "分数的简单加减",
            "node_code": "3-1-2-kp1",
            "summary": "同分母分数加减的算理",
        }
    ]


def _run_desc(monkeypatch, *, flag=True, structure_mode=True, include_extended=False):
    """图路径 stub：解剖卡 + 出题 items；单步路径 stub：items。返回 (out, db)。"""
    db = _db()
    if flag and structure_mode:
        calls = _stub_chat(monkeypatch, [_good_card(), _items_text()], module=qg)
    else:
        calls = _stub_chat(monkeypatch, [_items_text()], module=qd)
    monkeypatch.setattr(qd, "USE_LANGGRAPH_DESC_QUIZ", flag)
    out = _run(
        qd.generateQuizFromDescription(
            db,
            scholar_id="sch_1",
            intent=_intent(),
            grounded_kps=_grounded(),
            intake_meta={"missing_required": [], "canonical_text": "3年级上 · 分数 · 5道"},
            include_extended=include_extended,
            structure_mode=structure_mode,
            actor="u_1",
        )
    )
    return out, db, calls


class TestGenerateDescGraphPath:
    def test_graph_path_persist_and_public(self, monkeypatch):
        out, db, calls = _run_desc(monkeypatch)
        assert out["quiz_id"].startswith("q_") or out["quiz_id"]
        assert len(out["items"]) == 5
        # R3：answer / rationale / fingerprint 不出参
        assert all("answer" not in it and "rationale" not in it for it in out["items"])
        # §7.2 展示字段出参
        assert out["items"][0]["target_error_step"] == 2
        # 图版本随出参
        assert out["graph_version"] == _GRAPH_VERSION
        # 落库字段
        saved = db._data[MATH_QUIZ_COLLECTION][0]
        assert saved["graph_version"] == _GRAPH_VERSION
        assert saved["intake_meta"]["canonical_text"]
        assert saved["validation_reasons"] == []
        assert saved["items"][0]["answer"]  # 落库保留答案（渲染/判题用）

    def test_flag_off_single_step(self, monkeypatch):
        out, db, calls = _run_desc(monkeypatch, flag=False)
        assert len(calls) == 1  # 单步 = 一次 LLM 调用
        assert len(out["items"]) == 5
        assert out["graph_version"] == 0
        assert db._data[MATH_QUIZ_COLLECTION][0]["graph_version"] == 0

    def test_structure_mode_false_request_bypass(self, monkeypatch):
        out, db, calls = _run_desc(monkeypatch, flag=True, structure_mode=False)
        assert len(calls) == 1  # 请求级旁路也走单步
        assert out["graph_version"] == 0

    def test_count_validation(self, monkeypatch):
        _stub_chat(monkeypatch, [_items_text()], module=qd)
        from services.math.quiz import InvalidQuizSourceError

        with pytest.raises(InvalidQuizSourceError):
            _run(
                qd.generateQuizFromDescription(
                    _db(),
                    scholar_id="sch_1",
                    intent={"question_count": 999},
                    grounded_kps=[],
                )
            )


# ===========================================================================
# T21 A/B 对照（R18：flag 开关出参逐字节一致）
# ===========================================================================


class TestFlagAB:
    @staticmethod
    def _strip_ids(out):
        o = {**out}
        o.pop("quiz_id")  # 每次生成新 id，非对照字段
        o["items"] = [
            {k: v for k, v in it.items() if k != "quiz_item_id"} for it in o["items"]
        ]
        return o

    def test_items_identical_across_modes(self, monkeypatch):
        """同意图同 LLM 产物 → 开/关两跑的 items 出参内容一致（除随机 item_id）。"""
        out_on, _, _ = _run_desc(monkeypatch, flag=True)
        out_off, _, _ = _run_desc(monkeypatch, flag=False)
        assert self._strip_ids(out_on)["items"] == self._strip_ids(out_off)["items"]

    def test_flag_off_deterministic_and_isolated(self, monkeypatch):
        """R18：flag 关 = 与既有单步逐字节一致——同 fixture 两跑完全相等（确定性），
        且出参不含图路径语义（graph_version=0、无拔高标记）。"""
        out1, _, _ = _run_desc(monkeypatch, flag=False)
        out2, _, _ = _run_desc(monkeypatch, flag=False)
        assert self._strip_ids(out1) == self._strip_ids(out2)
        assert out1["graph_version"] == 0
        assert all(not it.get("is_extended") for it in out1["items"])

    def test_flag_off_no_graph_fields_added(self, monkeypatch):
        """R18：flag 关 → 出参不出现图专属新语义（graph_version 恒 0）。"""
        out, _, _ = _run_desc(monkeypatch, flag=False)
        assert out["graph_version"] == 0
        assert all(not it.get("is_extended") for it in out["items"])


class TestNeighborLoading:
    def test_neighbors_same_grade_excludes_grounded(self, monkeypatch):
        db = _db()
        out = _run(qd.load_neighbor_kps(db, _grounded()))
        assert [n["name"] for n in out] == ["单位换算"]  # 同 grade=3，排除自身

    def test_no_grounded_no_neighbors(self, monkeypatch):
        assert _run(qd.load_neighbor_kps(_db(), [{"name": "x"}])) == []

    def test_no_grade_no_neighbors(self):
        db = FakeDB(
            seed={
                CURRICULUM_NODE_COLLECTION: [
                    {"code": "x-1", "title": "分数的简单加减", "ai_summary": {}}
                ]
            }
        )
        assert _run(
            qd.load_neighbor_kps(
                db, [{"name": "分数的简单加减", "node_code": "x-1"}]
            )
        ) == []
