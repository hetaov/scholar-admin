"""单元测试：数学语音出题主图（基础题段）services/math/desc_quiz_graph.py

设计：docs_v1/数学/出题-语音描述出题-LangGraph编排方案.md §5；任务卡 T13~T15。
分段：card（T13 解剖/重解剖/降级）/ basic（T14 生成解析）/ route（T15 校验路由）。
LLM 一律 stub（monkeypatch _call_chat 响应队列）。
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from services.math import desc_quiz_graph as qg
from services.math.desc_validators import (
    check_answer_leakage,
    check_kp_in_scope,
)
from services.math.desc_quiz_graph import validate_basic_item


def _run(coro):
    return asyncio.run(coro)


_GOOD_CARD = {
    "core_rationale": "同分母分数加减：分母不变，分子相加减",
    "step_sequence": ["提取两个分数", "判断分母相同", "分子相加减", "结果约分"],
    "givent_unknown": {"givens": ["两个同分母分数"], "unknown": "和/差"},
    "error_prone_steps": [
        {"step": 3, "trap": "分母相加", "weight": 0.6},
        {"step": 4, "trap": "忘记约分", "weight": 0.4},
    ],
    "difficulty_band": 2,
    "variant_axes": ["数值", "情境"],
}


def _card_text(**overrides) -> str:
    card = {**_GOOD_CARD, **overrides}
    return json.dumps(card, ensure_ascii=False)


def _items_text(items: list[dict]) -> str:
    return json.dumps({"items": items}, ensure_ascii=False)


def _good_items(count=4, start=1):
    """通过全部校验的 items（同分母分数加法、答案正确、不重复、无泄露）。

    pairs: (a, b, c, d, 分子之和)——四组全部同分母，answer = 和分子 / 原分母。
    """
    out = []
    pairs = [("3", "7", "2", "7", "5"), ("5", "8", "1", "8", "6"), ("2", "9", "4", "9", "6"), ("1", "5", "2", "5", "3")]
    for i in range(count):
        a, b, c, d, n = pairs[i % len(pairs)]
        out.append({
            "question": f"{a}/{b} + {c}/{d} = ?",
            "answer": f"{n}/{b}",
            "answer_type": "fraction",
            "difficulty": 2,
            "variant_level": "L1" if i < 2 else "L2",
            "knowledge_point_name": "分数的简单加减",
            "rationale": "考察同分母分数加减的算理掌握，分母保持不变。",
            "target_error_step": 3 if i < 2 else 0,
        })
    return out


_KP_GROUNDED = [{"name": "分数的简单加减", "summary": "同分母分数加减", "node_code": "3-1-2-kp1"}]


class _ChatStub:
    """_call_chat 替身：按 (system 关键词, 调用序号) 弹出响应队列。"""

    def __init__(self, monkeypatch, responses):
        self.queue = list(responses)
        self.calls: list[dict] = []
        self.module = monkeypatch

        async def _fake(*, system, user):
            self.calls.append({"system": system, "user": user})
            item = self.queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(qg, "_call_chat", _fake)

    @property
    def n_calls(self):
        return len(self.calls)

    @property
    def last_user(self):
        return self.calls[-1]["user"]


# ===========================================================================
# T13 解剖：解析 / 重解剖 / 降级
# ===========================================================================


class TestParseCard:
    def test_good(self):
        assert qg.parse_card_response(_card_text())["difficulty_band"] == 2

    def test_missing_steps_raises(self):
        with pytest.raises(Exception):
            qg.parse_card_response(_card_text(step_sequence=None))


class TestDeconstructGraph:
    def test_card_ok_single_call(self, monkeypatch):
        stub = _ChatStub(monkeypatch, [_card_text(), _items_text(_good_items())])
        out = _run(qg.run_desc_quiz_basic(grounded_kps=_KP_GROUNDED, question_count=4))
        assert stub.n_calls == 2  # 解剖 1 次 + 生成 1 次
        assert out["card"]["core_rationale"]
        assert out["card_degraded"] is False
        assert out["warnings"] == []
        assert len(out["items"]) == 4

    def test_re_deconstruct_once_then_ok(self, monkeypatch):
        stub = _ChatStub(
            monkeypatch,
            ["不是JSON", _card_text(), _items_text(_good_items())],
        )
        out = _run(qg.run_desc_quiz_basic(grounded_kps=_KP_GROUNDED, question_count=4))
        assert stub.n_calls == 3  # 失败 1 + 重解剖 1 + 生成 1
        assert out["card"] is not None
        assert out["card_degraded"] is False

    def test_degrade_after_re_deconstruct_fails(self, monkeypatch):
        stub = _ChatStub(
            monkeypatch,
            ["不是JSON", _card_text(step_sequence=[]), _items_text(_good_items())],
        )
        out = _run(qg.run_desc_quiz_basic(grounded_kps=_KP_GROUNDED, question_count=4))
        assert out["card"] is None
        assert out["card_degraded"] is True
        assert qg.W_CARD_DEGRADED in out["warnings"]
        # 降级后仍直出（题目照常返回，不整体失败）
        assert len(out["items"]) == 4

    def test_degraded_generate_prompt_has_no_card(self, monkeypatch):
        stub = _ChatStub(
            monkeypatch,
            ["不是JSON", "还不是JSON", _items_text(_good_items())],
        )
        _run(qg.run_desc_quiz_basic(grounded_kps=_KP_GROUNDED, question_count=4))
        assert "无结构卡" in stub.last_user

    def test_retry_prompt_contains_feedback(self, monkeypatch):
        # 第一轮 items 含错误答案（V2），第二轮修正
        bad = _good_items(2)
        bad[0] = {**bad[0], "answer": "5/14"}  # 经典 misconception 答案
        stub = _ChatStub(
            monkeypatch,
            [_card_text(), _items_text(bad), _items_text(_good_items(2))],
        )
        out = _run(qg.run_desc_quiz_basic(grounded_kps=_KP_GROUNDED, question_count=2))
        assert stub.n_calls == 3
        assert "V2_ANSWER_WRONG" in stub.last_user  # 定向修正 prompt 附失败原因
        assert out["retries_used"] == 1
        assert out["items"]

    def test_empty_generation_retried_once_then_ok(self, monkeypatch):
        """空产出（超时/不合法）→ 重出一次；第二次正常 → 照常返回（真机 ReadTimeout 回归）"""
        stub = _ChatStub(
            monkeypatch,
            [_card_text(), '{"items": []}', _items_text(_good_items())],
        )
        out = _run(qg.run_desc_quiz_basic(grounded_kps=_KP_GROUNDED, question_count=4))
        assert stub.n_calls == 3  # 解剖 1 + 空产出 1 + 重出 1
        assert len(out["items"]) == 4

    def test_empty_generation_retry_prompt_mentions_empty(self, monkeypatch):
        stub = _ChatStub(
            monkeypatch,
            [_card_text(), '{"items": []}', _items_text(_good_items())],
        )
        _run(qg.run_desc_quiz_basic(grounded_kps=_KP_GROUNDED, question_count=2))
        assert "未返回任何题目" in stub.last_user

    def test_timeout_generation_leaves_gen_failed_warning(self, monkeypatch):
        """LLM 抛超时异常 → 生成节点留 GEN_FAILED 告警（前端失败文案可区分）"""
        stub = _ChatStub(
            monkeypatch,
            [_card_text(), httpx.ReadTimeout("timed out"), _items_text(_good_items())],
        )
        out = _run(qg.run_desc_quiz_basic(grounded_kps=_KP_GROUNDED, question_count=2))
        assert any("GEN_FAILED" in w for w in out["warnings"])
        assert out["items"]

    def test_empty_generation_twice_fails_with_gen_empty(self, monkeypatch):
        """连续两次空产出 → 停止重试，items=[] + GEN_EMPTY 告警（failed 文案带原因码）"""
        stub = _ChatStub(
            monkeypatch,
            [_card_text(), '{"items": []}', '{"items": []}'],
        )
        out = _run(qg.run_desc_quiz_basic(grounded_kps=_KP_GROUNDED, question_count=2))
        assert stub.n_calls == 3  # 解剖 1 + 两次生成（只重出一次）
        assert out["items"] == []
        assert any("GEN_EMPTY" in w for w in out["warnings"])


# ===========================================================================
# T14 生成解析
# ===========================================================================


class TestParseItems:
    def test_wrapped(self):
        items = qg.parse_basic_items_response(_items_text([{"question": "q", "answer": "1"}]))
        assert len(items) == 1

    def test_bare_list(self):
        items = qg.parse_basic_items_response('[{"question": "q", "answer": "1"}]')
        assert len(items) == 1

    def test_non_dict_dropped(self):
        items = qg.parse_basic_items_response('{"items": [{"question": "q"}, "junk", 42]}')
        assert len(items) == 1

    def test_no_question_dropped(self):
        items = qg.parse_basic_items_response('{"items": [{"answer": "1"}, {"question": "q2", "answer": "2"}]}')
        assert len(items) == 1 and items[0]["question"] == "q2"


# ===========================================================================
# T15 单题校验 V1/V2/V4/V5/V6 + 路由（重试/剔除/不整体失败）
# ===========================================================================


class TestValidateBasicItem:
    _REF = "3/7 + 2/7 = ?"

    def _item(self, **kw):
        item = {
            "question": "5/8 + 1/8 = ?",
            "answer": "6/8",
            "knowledge_point_name": "分数的简单加减",
            "rationale": "考察同分母分数加减。",
        }
        item.update(kw)
        return item

    def test_pass(self):
        assert validate_basic_item(self._item(), ref_question=self._REF, allowed_kps=["分数的简单加减"]) == []

    def test_v1_structure_mismatch(self):
        codes = validate_basic_item(
            self._item(question="1/2 + 1/3 - 1/6 = ?", answer="4/6"),
            ref_question=self._REF,
        )
        assert "V1_STRUCTURE_MISMATCH" in codes

    def test_v2_answer_wrong(self):
        codes = validate_basic_item(self._item(answer="6/16"), ref_question=self._REF)
        assert "V2_ANSWER_WRONG" in codes

    def test_v4_duplicate(self):
        codes = validate_basic_item(
            self._item(question="3/7+2/7=?"), ref_question=self._REF
        )
        assert "V4_DUPLICATE" in codes

    def test_v5_leak_in_rationale(self):
        codes = validate_basic_item(
            self._item(rationale="答案就是 6/8，直接约分即可。"),
            ref_question=self._REF,
        )
        assert "V5_ANSWER_LEAK" in codes

    def test_v5_no_leak_decimal(self):
        # 数值不在题干/rationale 中 → 不判泄露
        item = self._item(question="2.7 + 0.36 = ?", answer="3.06", rationale="考察小数加法对位。")
        assert check_answer_leakage(item) is False

    def test_v6_kp_out_of_scope(self):
        codes = validate_basic_item(
            self._item(knowledge_point_name="小数乘法"),
            ref_question=self._REF,
            allowed_kps=["分数的简单加减"],
        )
        assert "V6_KP_OUT_OF_SCOPE" in codes

    def test_v6_empty_allowed_not_block(self):
        assert validate_basic_item(self._item(knowledge_point_name="量子力学")) == []


# ===========================================================================
# T16 build_extend_plan（零 token 计划器）
# ===========================================================================

from services.math.desc_quiz_graph import build_extend_plan  # noqa: E402

_NEIGHBORS = [
    {"name": "单位换算", "node_code": "3-2-1-kp2", "summary": "元角分与米厘米"},
    {"name": "图形面积", "node_code": "3-5-1-kp1", "summary": "长方形面积"},
]


class TestBuildExtendPlan:
    def test_two_step_ladder(self):
        plan = build_extend_plan(_NEIGHBORS, base_kp_names=["分数的简单加减"], base_difficulty=2, question_count=5)
        assert plan["degraded"] is False
        assert [s["new_kp"]["name"] for s in plan["ladder"]] == ["单位换算", "图形面积"]
        assert [s["difficulty_delta"] for s in plan["ladder"]] == [1, 2]
        assert plan["ladder"][0]["combine_with"] == "分数的简单加减"
        assert plan["ladder"][1]["combine_with"] == "单位换算"  # 第二档与第一档组合

    def test_small_count_one_step(self):
        # count ≤2 → 设计上只出 1 档（非降级）；邻居充足
        plan = build_extend_plan(_NEIGHBORS, base_kp_names=["分数的简单加减"], base_difficulty=2, question_count=2)
        assert len(plan["ladder"]) == 1
        assert plan["degraded"] is False

    def test_insufficient_neighbors_degraded(self):
        plan = build_extend_plan(
            _NEIGHBORS[:1], base_kp_names=["x"], base_difficulty=2, question_count=5
        )
        assert len(plan["ladder"]) == 1
        assert plan["degraded"] is True  # 邻居不足 2 档 → degraded

    def test_empty_pool_degraded(self):
        plan = build_extend_plan([], base_kp_names=["x"], base_difficulty=2, question_count=5)
        assert plan["ladder"] == [] and plan["degraded"] is True

    def test_used_kps_excluded(self):
        plan = build_extend_plan(
            _NEIGHBORS, base_kp_names=["x"], base_difficulty=2, question_count=5,
            used_kps=["单位换算"],
        )
        assert [s["new_kp"]["name"] for s in plan["ladder"]] == ["图形面积"]


# ===========================================================================
# T17 拔高段图流：通过 / 降档 / 剔除
# ===========================================================================


def _ext_text(question, answer, difficulty, name="单位换算"):
    return json.dumps({"items": [{
        "question": question, "answer": answer, "answer_type": "fraction",
        "difficulty": difficulty, "knowledge_point_name": name,
        "rationale": "在基础考点上引入新知识点，考察综合运用。",
    }]}, ensure_ascii=False)


class TestExtendedSegment:
    def _base_state_items(self):
        return _good_items(4)

    def test_both_steps_pass(self, monkeypatch):
        stub = _ChatStub(monkeypatch, [
            _card_text(),
            _items_text(self._base_state_items()),
            _ext_text("单位换算：一条彩带 3/7 米，先用去 2/7 米，剩下多少米？合多少厘米？", "100/7", 3),
            _ext_text("图形面积：一张长方形纸长 3/5 米、宽 1/2 米，面积是多少平方米？", "3/10", 4, name="图形面积"),
        ])
        out = _run(qg.run_desc_quiz_basic(
            grounded_kps=_KP_GROUNDED, question_count=4,
            include_extended=True, neighbors=_NEIGHBORS,
        ))
        assert stub.n_calls == 4  # 解剖 + 基础 + 拔高×2
        assert len(out["items"]) == 6
        assert len(out["extended_items"]) == 2
        assert out["extended_items"][0]["is_extended"] is True
        assert out["extended_items"][0]["extended_meta"]["step"] == 1
        assert out["extend_degrades"] == 0

    def test_fail_then_downgrade_water_level(self, monkeypatch):
        # 第一档生成失败两次（E3 难度跳档）→ 降档重建（第二档 kp 顶上，水位收缩）
        stub = _ChatStub(monkeypatch, [
            _card_text(),
            _items_text(self._base_state_items()),
            _ext_text("3/7 + 2/7 = ?", "5/7", 9),  # E3：难度 9 ≠ 2+1
            _ext_text("图形面积：边长 1/2 米的正方形面积是多少平方米？", "1/4", 3, name="图形面积"),  # 降档后 kp 顶上
        ])
        out = _run(qg.run_desc_quiz_basic(
            grounded_kps=_KP_GROUNDED, question_count=4,
            include_extended=True, neighbors=_NEIGHBORS,
        ))
        assert out["extend_degrades"] == 1
        assert len(out["extended_items"]) == 1
        assert out["extended_items"][0]["extended_meta"]["new_kp"] == "图形面积"

    def test_fail_pool_exhausted_drops_with_warning(self, monkeypatch):
        stub = _ChatStub(monkeypatch, [
            _card_text(),
            _items_text(self._base_state_items()),
            _ext_text("3/7 + 2/7 = ?", "5/7", 9),  # 第一档失败，池只剩一个
            _ext_text("图形面积题超纲", "1/4", 9, name="图形面积"),  # 降档后再失败 → 池空
        ])
        out = _run(qg.run_desc_quiz_basic(
            grounded_kps=_KP_GROUNDED, question_count=4,
            include_extended=True, neighbors=_NEIGHBORS,
        ))
        assert len(out["extended_items"]) == 0
        assert any(qg.W_EXTEND_ITEM_DROPPED in w for w in out["warnings"])
        # 基础题照常返回（不整体失败）
        assert len([i for i in out["items"] if not i.get("is_extended")]) == 4

    def test_no_neighbors_skips_extended(self, monkeypatch):
        stub = _ChatStub(monkeypatch, [_card_text(), _items_text(self._base_state_items())])
        out = _run(qg.run_desc_quiz_basic(
            grounded_kps=_KP_GROUNDED, question_count=4,
            include_extended=True, neighbors=[],
        ))
        assert stub.n_calls == 2
        assert out["extended_items"] == []
        assert any(qg.W_EXTEND_PLAN_DEGRADED in w for w in out["warnings"])

    def test_include_extended_false_no_plan(self, monkeypatch):
        stub = _ChatStub(monkeypatch, [_card_text(), _items_text(self._base_state_items())])
        out = _run(qg.run_desc_quiz_basic(
            grounded_kps=_KP_GROUNDED, question_count=4,
            include_extended=False, neighbors=_NEIGHBORS,
        ))
        assert stub.n_calls == 2
        assert out["extended_items"] == []
        assert not any(qg.W_EXTEND_PLAN_DEGRADED in w for w in out["warnings"])


class TestValidateRoute:
    def test_final_retry_drops_bad_item_not_whole(self, monkeypatch):
        # 三轮生成（初轮 + 2 次重试）均含 1 道坏题 → 重试耗尽后仅剔除坏题，好题保留
        bad1 = [_good_items(2)[0], {**_good_items(2)[1], "answer": "6/16"}]
        bad2 = [_good_items(2)[0], {**_good_items(2)[1], "answer": "6/16"}]
        bad3 = [_good_items(2)[0], {**_good_items(2)[1], "answer": "6/16"}]
        stub = _ChatStub(
            monkeypatch,
            [_card_text(), _items_text(bad1), _items_text(bad2), _items_text(bad3)],
        )
        out = _run(qg.run_desc_quiz_basic(grounded_kps=_KP_GROUNDED, question_count=2))
        assert stub.n_calls == 4  # 解剖 + 生成×3（初轮 + 重试 ≤2）
        assert len(out["items"]) == 1
        assert any(qg.W_ITEM_DROPPED in w and "V2" in w for w in out["warnings"])

    def test_never_raises(self, monkeypatch):
        # 打桩：此用例验证「异常永不向上抛」，不能打真 LLM（混元时代因 400 快速失败
        # 碰巧通过；切火山后真实出题 30s+ 且耗 token）
        _ChatStub(monkeypatch, [httpx.ConnectError("offline")])
        out = _run(
            qg.run_desc_quiz_basic(grounded_kps=_KP_GROUNDED, question_count=4, max_retries=0)
        )
        assert out["status"] in {"basic_ok", "error"}
        assert isinstance(out["items"], list)
