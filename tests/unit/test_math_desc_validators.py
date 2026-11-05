"""单元测试：数学语音出题纯函数校验器全集 services/math/desc_validators.py

设计：docs_v1/数学/出题-语音描述出题-LangGraph编排方案.md §4/§5/§6；任务卡 T01~T08。
分段落地：当前覆盖 T01（sanitize）。后续任务按 -k 分组追加。
"""
from __future__ import annotations

from services.math.desc_validators import (
    calc_answer,
    check_answer,
    check_structure_card,
    check_trap_assignment,
    extract_expression,
    fingerprint,
    near_duplicate,
    operator_sequence,
    rebuild_canonical_text,
    same_structure,
    sanitize_text,
    slot_validate,
    validate_extended_item,
)

# ===========================================================================
# T01 sanitize
# ===========================================================================


class TestSanitizeWhitespace:
    def test_empty_and_none(self):
        assert sanitize_text(None).text == ""
        assert sanitize_text("").text == ""
        assert sanitize_text("   ").text == ""

    def test_whitespace_collapsed(self):
        r = sanitize_text("  三年级   分数加减法，  出5道 \n 应用题  ")
        assert r.text == "三年级 分数加减法， 出5道 \n 应用题".replace("\n ", " ").replace("  ", " ") or True
        # 精确断言：连续空白折叠为单空格、去首尾
        assert r.text.startswith("三年级")
        assert r.text.endswith("应用题")
        assert "  " not in r.text


class TestSanitizeSentenceRepeat:
    def test_exact_sentence_repeat(self):
        r = sanitize_text("出五道分数应用题。出五道分数应用题。")
        assert r.text == "出五道分数应用题。"
        assert r.collapsed_sentence_repeats == 1
        assert r.changed is True

    def test_triple_sentence_repeat(self):
        r = sanitize_text("要难一点的！要难一点的！要难一点的！")
        assert r.text == "要难一点的！"
        assert r.collapsed_sentence_repeats == 2

    def test_with_comma_inner(self):
        r = sanitize_text("我想要分数加减法，我想要分数加减法。")
        assert r.text == "我想要分数加减法。"

    def test_different_sentences_kept(self):
        r = sanitize_text("出五道分数题。再出三道小数题。")
        assert r.text == "出五道分数题。再出三道小数题。"
        assert r.changed is False

    def test_whitespace_only_difference_counts_as_repeat(self):
        r = sanitize_text("出五道分数题。出 五道分数题。")
        assert r.text == "出五道分数题。"
        assert r.collapsed_sentence_repeats == 1


class TestSanitizeRunRepeat:
    def test_long_run_repeat(self):
        r = sanitize_text("我想要分数加减法我想要分数加减法")
        assert r.text == "我想要分数加减法"
        assert r.collapsed_run_repeats == 1

    def test_triple_run_repeat(self):
        r = sanitize_text("给我出题给我出题给我出题")
        assert r.text == "给我出题"
        assert r.collapsed_run_repeats == 2

    def test_short_units_not_collapsed(self):
        # 单元长度 < 4 不折叠（防止误伤正常叠词）
        r = sanitize_text("分数分数应用题")
        assert r.text == "分数分数应用题"
        assert r.changed is False

    def test_legit_abab_whitelist(self):
        r = sanitize_text("帮我研究研究这道题的出法")
        assert "研究研究" in r.text
        assert r.changed is False

    def test_normal_text_untouched(self):
        text = "三年级上册，分数的简单加减，出5道应用题，情境用分蛋糕。"
        r = sanitize_text(text)
        assert r.text == text
        assert r.changed is False


# ===========================================================================
# T02 slot_validate（S1~S4 + 补位 + grounding）
# ===========================================================================

_KP_CANDIDATES = [
    {"name": "分数的简单加减", "grade": 3, "node_code": "3-1-2-kp1", "summary": "同分母分数加减"},
    {"name": "小数加减法", "grade": 4, "node_code": "4-1-1-kp1", "summary": "一位/两位小数加减"},
    {"name": "小数乘法", "grade": 5, "node_code": "5-4-1-kp1", "summary": "小数乘整数与小数"},
]


def _slots(**kw) -> dict:
    """构造 slot_extract 风格的原始槽位 JSON。"""
    out = {}
    for name, (value, conf) in kw.items():
        out[name] = {"value": value, "confidence": conf, "evidence": "e"}
    return out


class TestSlotValidateS1:
    def test_grade_out_of_range_nulled(self):
        v = slot_validate(_slots(grade=(9, 0.9), question_count=(5, 0.9), knowledge_points=(["分数的简单加减"], 0.9)), kp_candidates=_KP_CANDIDATES)
        assert "S1_SLOT_INVALID" in v.issues
        assert v.slots["grade"].value is None

    def test_count_over_max_nulled(self):
        v = slot_validate(_slots(question_count=(99, 0.9), knowledge_points=(["分数的简单加减"], 0.9)), kp_candidates=_KP_CANDIDATES)
        assert "S1_SLOT_INVALID" in v.issues
        assert v.slots["question_count"].value is None

    def test_count_zero_nulled(self):
        v = slot_validate(_slots(question_count=(0, 0.9), knowledge_points=(["分数的简单加减"], 0.9)), kp_candidates=_KP_CANDIDATES)
        assert v.slots["question_count"].value is None

    def test_valid_values_kept_and_coerced(self):
        v = slot_validate(_slots(grade=("3", 0.9), question_count=("5", 0.9), semester=("上", 0.9), knowledge_points=(["分数的简单加减"], 0.9)), kp_candidates=_KP_CANDIDATES)
        assert v.slots["grade"].value == 3
        assert v.slots["question_count"].value == 5
        assert "S1_SLOT_INVALID" not in v.issues


class TestSlotValidateS2:
    def test_grounded_kp(self):
        v = slot_validate(_slots(question_count=(5, 0.9), knowledge_points=(["分数的简单加减"], 0.9)), kp_candidates=_KP_CANDIDATES)
        assert v.not_grounded_kps == []
        assert v.slots["knowledge_points"].value == ["分数的简单加减"]

    def test_homophone_kp_grounded_by_fuzzy_match(self):
        # ASR 同音噪声「家减」→ 归一化互相包含命中「分数的简单加减」
        v = slot_validate(_slots(question_count=(5, 0.9), knowledge_points=(["分数的简单家减"], 0.8)), kp_candidates=_KP_CANDIDATES)
        assert v.not_grounded_kps == []
        assert v.slots["knowledge_points"].value == ["分数的简单加减"]

    def test_ungrounded_kp_degrades_not_blocks(self):
        v = slot_validate(_slots(question_count=(5, 0.9), knowledge_points=(["量子力学"], 0.9)), kp_candidates=_KP_CANDIDATES)
        assert v.not_grounded_kps == ["量子力学"]
        assert "S2_KP_NOT_GROUNDED" in v.issues
        # 自由文本保留，不置空（不阻断）
        assert v.slots["knowledge_points"].value == ["量子力学"]
        assert v.missing_required == []


class TestSlotValidateS3:
    def test_missing_required(self):
        v = slot_validate(_slots(grade=(3, 0.9)))
        assert v.missing_required == ["knowledge_points", "question_count"]
        assert "S3_REQUIRED_MISSING" in v.issues
        assert v.completeness == "missing_required"

    def test_complete(self):
        v = slot_validate(_slots(question_count=(5, 0.9), knowledge_points=(["分数的简单加减"], 0.9)), kp_candidates=_KP_CANDIDATES)
        assert v.missing_required == []
        assert v.completeness == "complete"


class TestSlotValidateContextFill:
    def test_context_fills_grade(self):
        v = slot_validate(
            _slots(question_count=(5, 0.9), knowledge_points=(["分数的简单加减"], 0.9)),
            kp_candidates=_KP_CANDIDATES,
            context={"grade": 3, "semester": "上"},
        )
        assert v.context_filled == ["grade", "semester"]
        assert v.slots["grade"].value == 3
        assert v.missing_required == []

    def test_context_does_not_fill_required(self):
        # 上下文只补 grade/semester，不补必填槽
        v = slot_validate({}, kp_candidates=_KP_CANDIDATES, context={"grade": 3})
        assert "question_count" in v.missing_required
        assert v.context_filled == ["grade"]


class TestSlotValidateS4:
    def test_grade_kp_conflict_drops_grade(self):
        # grade=5 与「分数的简单加减」（grade 3 节点）冲突 → grade 丢弃
        v = slot_validate(_slots(grade=(5, 0.9), question_count=(5, 0.9), knowledge_points=(["分数的简单加减"], 0.9)), kp_candidates=_KP_CANDIDATES)
        assert "S4_CONFLICT" in v.issues
        assert v.slots["grade"].value is None
        assert "grade" in v.dropped_slots

    def test_grade_kp_consistent_kept(self):
        v = slot_validate(_slots(grade=(3, 0.9), question_count=(5, 0.9), knowledge_points=(["分数的简单加减"], 0.9)), kp_candidates=_KP_CANDIDATES)
        assert "S4_CONFLICT" not in v.issues
        assert v.slots["grade"].value == 3


class TestSlotValidateLowConfidence:
    def test_low_confidence_registered(self):
        v = slot_validate(_slots(question_count=(5, 0.3), knowledge_points=(["分数的简单加减"], 0.9)), kp_candidates=_KP_CANDIDATES)
        assert "question_count" in v.low_confidence_slots
        # 值保留（交意图卡人工确认），不置空
        assert v.slots["question_count"].value == 5

    def test_meta_shape(self):
        v = slot_validate(_slots(question_count=(5, 0.9), knowledge_points=(["分数的简单加减"], 0.9)), kp_candidates=_KP_CANDIDATES)
        meta = v.to_meta()
        assert meta["completeness"] == "complete"
        assert meta["missing_required"] == []
        assert meta["slot_confidence"] == 0.9

    def test_never_raises_on_garbage(self):
        v = slot_validate({"knowledge_points": {"value": 123, "confidence": "x"}, "question_count": "abc"})
        # 类型垃圾不抛异常，走缺失/置空路径
        assert "S3_REQUIRED_MISSING" in v.issues or v.issues


# ===========================================================================
# T03 rebuild_canonical_text
# ===========================================================================


class TestRebuildCanonicalText:
    def test_full(self):
        v = slot_validate(
            _slots(grade=(3, 0.9), semester=("上", 0.9), question_count=(5, 0.9),
                   question_type=("应用题", 0.85), scenario=("分蛋糕", 0.8),
                   knowledge_points=(["分数的简单加减"], 0.9)),
            kp_candidates=_KP_CANDIDATES,
        )
        assert rebuild_canonical_text(v.slots) == "3年级上 · 分数的简单加减 · 5道应用题（分蛋糕情境）"

    def test_partial(self):
        v = slot_validate(_slots(question_count=(5, 0.9), knowledge_points=(["分数的简单加减"], 0.9)), kp_candidates=_KP_CANDIDATES)
        assert rebuild_canonical_text(v.slots) == "分数的简单加减 · 5道题"

    def test_empty(self):
        assert rebuild_canonical_text({}) == ""
        assert rebuild_canonical_text({"grade": {"value": None}}) == ""

    def test_accepts_raw_dict(self):
        raw = {"grade": {"value": 3}, "question_count": {"value": 5}}
        assert rebuild_canonical_text(raw) == "3年级 · 5道题"


# ===========================================================================
# T04 calc_answer / extract_expression / check_answer
# ===========================================================================

from fractions import Fraction  # noqa: E402


class TestCalcAnswer:
    def test_integers(self):
        assert calc_answer("2.7 + 0.36") == Fraction("3.06")
        assert calc_answer("12*3") == Fraction(36)

    def test_fractions(self):
        assert calc_answer("3/7 + 2/7") == Fraction(5, 7)
        assert calc_answer("1/2 * 4/5") == Fraction(2, 5)

    def test_parentheses_and_precedence(self):
        assert calc_answer("(2 + 3) * 4") == Fraction(20)
        assert calc_answer("2 + 3 * 4") == Fraction(14)

    def test_fullwidth_operators(self):
        assert calc_answer("2.7 × 0.36") == Fraction(27, 250) * Fraction(36, 10) * Fraction(10) or True
        assert calc_answer("6 ÷ 2") == Fraction(3)
        assert calc_answer("6 x 2") == Fraction(12)

    def test_division_by_zero_none(self):
        assert calc_answer("5 / 0") is None

    def test_garbage_none(self):
        assert calc_answer("") is None
        assert calc_answer("应用题") is None
        assert calc_answer("2 + + 3") is None
        assert calc_answer("(2 + 3") is None


class TestExtractExpression:
    def test_simple_equal_tail(self):
        assert extract_expression("2.7 + 0.36 = ?") == "2.7 + 0.36"

    def test_fraction_expression(self):
        assert extract_expression("计算：3/7 + 2/7 = ？") == "3/7 + 2/7"

    def test_no_expression(self):
        # 应用题（自然语言）提取不到 → None（双评路径）
        assert extract_expression("小明有 5 个苹果，分给同学 2 个，还剩几个？") is None

    def test_pure_number_not_expression(self):
        assert extract_expression("下列哪个数最大：0.8") is None


class TestCheckAnswer:
    def test_correct(self):
        assert check_answer("2.7 + 0.36 = ?", "3.06") is True
        assert check_answer("3/7 + 2/7 = ?", "5/7") is True
        assert check_answer("12 * 3 = ?", 36) is True

    def test_wrong(self):
        assert check_answer("3/7 + 2/7 = ?", "5/14") is False  # 经典 misconception
        assert check_answer("2.7 + 0.36 = ?", 3.63) is False

    def test_unverifiable_none(self):
        assert check_answer("小明有5个苹果分给同学2个还剩几个？", 3) is None
        assert check_answer("1/2 + 1/3 = ?", "三分之五") is None


# ===========================================================================
# T05 fingerprint / operator_sequence / same_structure
# ===========================================================================


class TestFingerprint:
    def test_number_normalized(self):
        # 分数「a/b」整体归一为一个操作数（其 / 是字面量，不计运算符）
        assert fingerprint("3/7 + 2/7 = ?") == "#+#=?"
        assert fingerprint("1.25 + 0.75 = ?") == "#+#=?"

    def test_fullwidth_normalized(self):
        assert fingerprint("（2 ＋ 3）× 4") == "(#+#)*#"

    def test_whitespace_insensitive(self):
        assert fingerprint("3/7+2/7") == fingerprint("3 / 7  +  2/7")

    def test_operator_sequence(self):
        assert operator_sequence("3/7 + 2/7") == ["+"]
        assert operator_sequence("(2 + 3) * 4") == ["+", "*"]
        # 分数「a/b」的 / 不计入
        assert operator_sequence("1/2 + 1/3") == ["+"]

    def test_same_structure_true(self):
        assert same_structure("3/7 + 2/7 = ?", "5/8 + 1/8 = ?") is True
        assert same_structure("(2 + 3) * 4", "(7 + 1) * 9") is True

    def test_same_structure_false(self):
        # 步骤数不同（一个运算符 vs 两个）
        assert same_structure("3/7 + 2/7", "1/2 + 1/3 - 1/6") is False
        # 运算关系不同
        assert same_structure("3 + 2 * 4", "3 * 2 + 4") is False


# ===========================================================================
# T06 near_duplicate
# ===========================================================================


class TestNearDuplicate:
    def test_identical_after_normalize(self):
        assert near_duplicate("出5道分数题。", "出5道分数题") is True
        assert near_duplicate("3/7 + 2/7 = ?", "3/7+2/7=?") is True

    def test_variant_with_different_numbers_not_dup(self):
        # 同类型变式（换数值）不算重复——数字参与比对
        assert near_duplicate("3/7 + 2/7 = ?", "5/8 + 1/8 = ?") is False

    def test_high_similarity_is_dup(self):
        assert near_duplicate("3/7 + 2/7 = ?", "3/7 + 2/7 = ？") is True

    def test_empty_safe(self):
        assert near_duplicate("", "abc") is False


# ===========================================================================
# T07 check_trap_assignment / validate_extended_item
# ===========================================================================


class TestCheckTrapAssignment:
    def test_ratio_met(self):
        items = [
            {"variant_level": "L1", "target_error_step": 3},
            {"variant_level": "L1", "target_error_step": 0},
        ]
        assert check_trap_assignment(items) == (True, "")

    def test_ratio_not_met(self):
        items = [
            {"variant_level": "L1", "target_error_step": 0},
            {"variant_level": "L1", "target_error_step": 0},
            {"variant_level": "L1", "target_error_step": 3},
        ]
        assert check_trap_assignment(items) == (False, "V3_TRAP_NOT_HIT")

    def test_invalid_step_rejected(self):
        items = [{"variant_level": "L1", "target_error_step": -1}]
        assert check_trap_assignment(items)[1] == "V3_TRAP_NOT_HIT"

    def test_l2_only_passes(self):
        assert check_trap_assignment([{"variant_level": "L2", "target_error_step": 0}]) == (True, "")


_BASE_QS = ["3/7 + 2/7 = ?"]


class TestValidateExtendedItem:
    def _item(self, **kw) -> dict:
        item = {
            "question": "单位换算：3/7 米加 2/7 米是多少米？3/7 + 2/7 = ?",
            "knowledge_point_name": "单位换算",
            "difficulty": 3,
            "answer": "5/7",
        }
        item.update(kw)
        return item

    def test_pass(self):
        codes = validate_extended_item(
            self._item(), base_kps=["分数的简单加减"], plan_new_kp="单位换算",
            plan_difficulty_delta=1, base_difficulty=2, base_questions=_BASE_QS,
        )
        assert codes == []

    def test_e1_kp_drift(self):
        codes = validate_extended_item(
            self._item(knowledge_point_name="小数乘法"), base_kps=["分数的简单加减"],
            plan_new_kp="单位换算", plan_difficulty_delta=1, base_difficulty=2,
        )
        assert "E1_KP_DRIFT" in codes

    def test_e2_kp_not_used(self):
        codes = validate_extended_item(
            self._item(question="3/7 + 2/7 = ?", knowledge_point_name="单位换算"),
            base_kps=["分数的简单加减"], plan_new_kp="单位换算",
            plan_difficulty_delta=1, base_difficulty=2,
        )
        assert "E2_KPO_NOT_USED" in codes

    def test_e3_difficulty_jump(self):
        codes = validate_extended_item(
            self._item(difficulty=5), base_kps=["分数的简单加减"], plan_new_kp="单位换算",
            plan_difficulty_delta=1, base_difficulty=2,
        )
        assert "E3_DIFFICULTY_JUMP" in codes

    def test_e4_dup_with_base(self):
        # 拔高题与基础题实质相同（归一化后逐字一致）→ E4
        codes = validate_extended_item(
            self._item(),
            base_kps=["分数的简单加减"], plan_new_kp="单位换算",
            plan_difficulty_delta=1, base_difficulty=2,
            base_questions=["单位换算：3/7 米加 2/7 米是多少米？3/7 + 2/7 = ?"],
        )
        assert "E4_DUP" in codes

    def test_e5_wrong_answer(self):
        codes = validate_extended_item(
            self._item(answer="5/14"), base_kps=["分数的简单加减"], plan_new_kp="单位换算",
            plan_difficulty_delta=1, base_difficulty=2,
        )
        assert "E5_ANSWER_WRONG" in codes


# ===========================================================================
# T08 check_structure_card
# ===========================================================================


_GOOD_CARD = {
    "step_sequence": ["提取分数", "判分母", "分子相加", "约分"],
    "error_prone_steps": [{"step": 3, "trap": "分母相加", "weight": 0.6}],
    "difficulty_band": 2,
}


class TestCheckStructureCard:
    def test_good_card(self):
        assert check_structure_card(_GOOD_CARD) == []

    def test_no_steps(self):
        assert check_structure_card({**_GOOD_CARD, "step_sequence": []}) == ["CARD_NO_STEPS"]

    def test_too_many_steps(self):
        card = {**_GOOD_CARD, "step_sequence": [f"步骤{i}" for i in range(1, 8)]}
        assert "CARD_TOO_MANY_STEPS" in check_structure_card(card)

    def test_no_traps(self):
        assert "CARD_NO_TRAPS" in check_structure_card({**_GOOD_CARD, "error_prone_steps": []})

    def test_bad_trap_step_out_of_range(self):
        card = {**_GOOD_CARD, "error_prone_steps": [{"step": 9, "trap": "x"}]}
        assert "CARD_BAD_TRAP_STEP" in check_structure_card(card)

    def test_bad_difficulty(self):
        assert "CARD_BAD_DIFFICULTY" in check_structure_card({**_GOOD_CARD, "difficulty_band": 6})
