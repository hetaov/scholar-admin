"""单元测试：数学题干符号归一化 services/math/math_text（乘号乱码根修 · 2026-11-04）

背景：出题 / 识别切题返回的题干此前零清洗，LLM 网关 charset 错标导致「×」的
UTF-8 字节被按 Latin-1/CP1252 解读成「Ã—」；prompt 未约束符号写法导致 \\times /
\\frac 等 LaTeX 命令泄漏为纯文本。本模块与小程序端 services/_utils/math-text.js
同口径，测试用例与前端一一对应（含幂等性与零影响锁）。

覆盖：
- mojibake 还原（× ÷ ≤ ≥ − ± ° · π）与合法字符零误伤（° ± – × 原样保留）；
- LaTeX 命令还原（\\times / \\frac / $...$ / \\left \\right / \\text / \\quad）；
  未知命令（\\sqrt）不误删；
- 符号变体统一（✕ ✖ ⨯ → ×，字母 x 不受影响）；
- 替换符 / 私用区 / 控制字符剔除（换行保留）；
- 幂等性（normalize(normalize(x)) == normalize(x)）与空值边界；
- 三个收口点：`_validate_quiz_items`（question/answer/hint）、
  `_to_public_quiz`（存量兜底）、`_validate_quiz_recognize_result`（question_text）。
"""

from __future__ import annotations

import pytest

from services.math.math_text import normalize_math_text
from services.math.quiz import _to_public_quiz, _validate_quiz_items
from services.math.quiz_recognizer import _validate_quiz_recognize_result

# mojibake 形态用码位转义书写，避免源码编码歧义：
# × = C3 97 → Ã(U+00C3) + —(0x97→U+2014)；÷ = C3 B7 → Ã + ·(U+00B7)
MOJIBAKE_TIMES = "\u00c3\u2014"
MOJIBAKE_DIV = "\u00c3\u00b7"
# ≤ = E2 89 A4 → â(â) ‰(0x89→U+2030) ¤(U+00A4)
MOJIBAKE_LE = "\u00e2\u2030\u00a4"
# − = E2 88 92 → â ˆ(0x88→U+02C6) ’(0x92→U+2019)
MOJIBAKE_MINUS = "\u00e2\u02c6\u2019"
# ± = C2 B1 → Â + ±；° = C2 B0 → Â + °；π = CF 80 → Ï + €
MOJIBAKE_PLUSMINUS = "\u00c2\u00b1"
MOJIBAKE_DEGREE = "\u00c2\u00b0"
MOJIBAKE_PI = "\u00cf\u20ac"


# ===========================================================================
# mojibake 还原
# ===========================================================================


class TestMojibake:
    def test_times(self):
        assert normalize_math_text(f"3 {MOJIBAKE_TIMES} 4 = 12") == "3 × 4 = 12"

    def test_times_in_chinese(self):
        assert normalize_math_text(f"求 6 {MOJIBAKE_TIMES} 7 的积") == "求 6 × 7 的积"

    def test_div(self):
        assert normalize_math_text(f"12 {MOJIBAKE_DIV} 4 = 3") == "12 ÷ 4 = 3"

    def test_le_ge(self):
        assert normalize_math_text(f"a {MOJIBAKE_LE} b") == "a ≤ b"

    def test_minus(self):
        assert normalize_math_text(f"5 {MOJIBAKE_MINUS} 3 = 2") == "5 - 3 = 2"

    def test_plusminus_degree_pi(self):
        assert normalize_math_text(MOJIBAKE_PLUSMINUS) == "±"
        assert normalize_math_text(MOJIBAKE_DEGREE) == "°"
        assert normalize_math_text(MOJIBAKE_PI) == "π"

    def test_legit_chars_untouched(self):
        assert normalize_math_text("30°") == "30°"
        assert normalize_math_text("x ± 1") == "x ± 1"
        assert normalize_math_text("3–4") == "3–4"
        assert normalize_math_text("6 × 7") == "6 × 7"

    def test_chinese_untouched(self):
        text = "小明买了 6 × 7 个苹果，共 42 个，正确吗？"
        assert normalize_math_text(text) == text


# ===========================================================================
# LaTeX 泄漏还原
# ===========================================================================


class TestLatex:
    def test_times(self):
        assert normalize_math_text("3 \\times 4 = 12") == "3 × 4 = 12"

    def test_basic_commands(self):
        assert normalize_math_text("8 \\div 2") == "8 ÷ 2"
        assert normalize_math_text("2 \\cdot 3") == "2 · 3"
        assert normalize_math_text("x \\le 5") == "x ≤ 5"
        assert normalize_math_text("x \\geq 5") == "x ≥ 5"
        assert normalize_math_text("x \\neq 5") == "x ≠ 5"
        assert normalize_math_text("x \\pm 1") == "x ± 1"
        assert normalize_math_text("\\pi r") == "π r"

    def test_frac(self):
        assert normalize_math_text("\\frac{3}{4}") == "3/4"
        assert normalize_math_text("\\frac{\\frac{1}{2}}{3}") == "1/2/3"

    def test_delimiters_and_spacing(self):
        assert normalize_math_text("$3 \\times 4$") == "3 × 4"
        assert normalize_math_text("\\(3+4\\)") == "3+4"
        # \left / \right 必须先于 \le 剥离，否则误命中前缀（\left → ≤ft）
        assert normalize_math_text("\\left(3+4\\right)") == "(3+4)"
        assert normalize_math_text("\\text{元}") == "元"
        assert normalize_math_text("3 \\quad 4") == "3   4"

    def test_unknown_command_preserved(self):
        assert normalize_math_text("\\sqrt{2}") == "\\sqrt{2}"


# ===========================================================================
# 符号变体与脏字符
# ===========================================================================


class TestVariantsAndDirty:
    def test_multiply_variants(self):
        assert normalize_math_text("3 \u2715 4") == "3 × 4"
        assert normalize_math_text("3 \u2716 4") == "3 × 4"
        assert normalize_math_text("3 \u2a2f 4") == "3 × 4"
        assert normalize_math_text("3 x 4") == "3 x 4"  # 字母 x 不收编

    def test_dirty_chars_removed_newline_kept(self):
        assert normalize_math_text("3 \ufffd 4") == "3  4"
        assert normalize_math_text("3 \ue000 4") == "3  4"
        assert normalize_math_text("3\x074") == "34"
        assert normalize_math_text("3\n4") == "3\n4"

    def test_strip(self):
        assert normalize_math_text("  3 × 4  ") == "3 × 4"


# ===========================================================================
# 幂等性 + 边界
# ===========================================================================


class TestIdempotencyAndEdges:
    def test_idempotent_on_dirty(self):
        dirty = f"3 {MOJIBAKE_TIMES} 4 = 12，\\frac{{3}}{{4}}，5 {MOJIBAKE_MINUS} 3，30\\circ"
        once = normalize_math_text(dirty)
        assert normalize_math_text(once) == once

    def test_idempotent_on_clean(self):
        clean = "小明买了 6 × 7 个苹果，共 42 个，正确吗？"
        assert normalize_math_text(normalize_math_text(clean)) == clean

    @pytest.mark.parametrize("bad", ["", None, 0, {}, []])
    def test_non_string_returns_empty(self, bad):
        assert normalize_math_text(bad) == ""


# ===========================================================================
# 三个收口点
# ===========================================================================


def _raw_items():
    return [
        {"question": f"12 {MOJIBAKE_TIMES} 3 = ?", "answer": "36", "answer_type": "number"},
        {"question": "x \\le 5 时 x 最大是几？", "answer": "5 \\times 1", "hint": "看 \\frac{1}{2}"},
    ]


class TestFunnels:
    def test_validate_quiz_items_normalizes_question_answer_hint(self):
        cleaned, warnings = _validate_quiz_items(
            _raw_items(), source_question="一个长方形长 6 厘米，宽 4 厘米，面积是多少？"
        )
        assert warnings == []
        assert [it["question"] for it in cleaned] == ["12 × 3 = ?", "x ≤ 5 时 x 最大是几？"]
        assert cleaned[1]["answer"] == "5 × 1"
        assert cleaned[1]["hint"] == "看 1/2"

    def test_to_public_quiz_normalizes_legacy_question(self):
        # 修复前落库的历史脏数据：读路径出参前兜底清洗
        quiz = {
            "quiz_id": "qz_1",
            "items": [
                {"quiz_item_id": "qi1", "question": f"12 {MOJIBAKE_TIMES} 3 = ?", "difficulty": 3}
            ],
            "sources": [{"source_index": 0, "question_text": "3 \\times 4 = 12"}],
        }
        public = _to_public_quiz(quiz)
        assert public["items"][0]["question"] == "12 × 3 = ?"
        assert public["sources"][0]["question_text"] == "3 × 4 = 12"
        # 红线 R3：answer / hint 恒不出参
        assert "answer" not in public["items"][0]
        assert "hint" not in public["items"][0]

    def test_validate_quiz_recognize_result_normalizes_question_text(self):
        result = {
            "items": [
                {
                    "knowledge_point_name": "乘法",
                    "node_code": "n1",
                    "question_text": f"3 {MOJIBAKE_TIMES} 4 = 12",
                    "confidence": 0.9,
                }
            ]
        }
        _validate_quiz_recognize_result(result)
        assert result["items"][0]["question_text"] == "3 × 4 = 12"
        assert result["items"][0]["confidence"] == 0.9
