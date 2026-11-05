"""数学题干文本归一化（2026-11-04 新增：乘号乱码根修）

背景：出题（POST /math/quiz/generate）与识别切题（POST /math/scan/quiz-recognize）
的题干由 LLM / Judge 产出，此前**无任何符号清洗**，三类脏符号直接落库并返回前端：

  1) 双编码乱码（mojibake）：LLM 网关/代理响应 charset 错标时，「×」(UTF-8: C3 97)
     的字节被按 Latin-1/CP1252 解读成「Ã—」；同理 ÷→「Ã·」、≤→「â‰¤」、≥→「â‰¥」、
     −→「âˆ’」。这是用户报障「乘号变乱码」的主形态。
  2) LaTeX 命令泄漏：prompt 此前未约束符号写法，模型偶尔输出 ``\\times`` /
     ``\\frac{3}{4}`` / ``$...$``，前端按纯文本渲染，用户看到反斜杠命令本身。
  3) 替换符 / 私用区 / 控制字符：OCR/模型对不认识的符号输出 U+FFFD 或 PUA 码位。

根修三层：① system prompt 约束符号写法（quiz.py / quiz_recognizer.py）；
② LLM 输出 funnel 统一过本模块归一化（落库前清洗，打印稿 PDF 同步受益）；
③ 读路径 ``_to_public_quiz`` 对存量脏数据兜底。小程序端 `services/_utils/math-text.js`
为同口径前端兜底，**两端口径必须一致且均幂等**（normalize(normalize(x)) == normalize(x)）。

口径：无法识别的形态一律**原样保留**（宁可脏显示，不可误改题意）；
非字符串入参返回空串；空串入 → 空串出。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 双编码乱码（mojibake）还原
# ---------------------------------------------------------------------------

# CP1252 把 0x80~0x9F 区间的字节映射成了一组 Unicode 高位字符（€ ‰ ˆ ' 等），
# mojibake 序列里常混着它们；还原字节时需要反向映射。未收录的码位放弃还原。
CP1252_TO_BYTE: dict[int, int] = {
    0x20AC: 0x80, 0x201A: 0x82, 0x0192: 0x83, 0x201E: 0x84, 0x2026: 0x85,
    0x2020: 0x86, 0x2021: 0x87, 0x02C6: 0x88, 0x2030: 0x89, 0x0160: 0x8A,
    0x2039: 0x8B, 0x0152: 0x8C, 0x017D: 0x8E, 0x2018: 0x91, 0x2019: 0x92,
    0x201C: 0x93, 0x201D: 0x94, 0x2022: 0x95, 0x2013: 0x96, 0x2014: 0x97,
    0x02DC: 0x98, 0x2122: 0x99, 0x0161: 0x9A, 0x203A: 0x9B, 0x0153: 0x9C,
    0x017E: 0x9E, 0x0178: 0x9F,
}


def _is_mojibake_char(ch: str) -> bool:
    c = ord(ch)
    return 0x80 <= c <= 0xFF or c in CP1252_TO_BYTE


def _fix_mojibake(s: str) -> str:
    """扫描「可疑字符」连续段，还原成字节后按 UTF-8 解码；失败原样保留。

    解码失败即放弃整段（如真正的 ° ± – 单独出现时，其字节不是合法 UTF-8 序列），
    中文等高位 BMP 字符不属于可疑字符，天然隔断分段、不受影响。
    """
    out: list[str] = []
    run_chars: list[str] = []
    run_bytes = bytearray()

    def flush() -> None:
        nonlocal run_chars, run_bytes
        if not run_chars:
            return
        try:
            decoded: str | None = bytes(run_bytes).decode("utf-8")
        except UnicodeDecodeError:
            decoded = None
        out.append(decoded if decoded is not None else "".join(run_chars))
        run_chars = []
        run_bytes = bytearray()

    for ch in s:
        if _is_mojibake_char(ch):
            c = ord(ch)
            run_chars.append(ch)
            run_bytes.append(c if c <= 0xFF else CP1252_TO_BYTE[c])
        else:
            flush()
            out.append(ch)
    flush()
    return "".join(out)


# ---------------------------------------------------------------------------
# LaTeX 命令泄漏还原
# ---------------------------------------------------------------------------

# 仅收录中小学数学题干会出现的命令；未知命令不动（交由前端展示与用户编辑）。
_LATEX_SYMBOLS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\^?\{?\\circ\}?"), "°"),
    (re.compile(r"\\degree"), "°"),
    (re.compile(r"\\times"), "×"),
    (re.compile(r"\\div(?:ide)?"), "÷"),
    (re.compile(r"\\cdot"), "·"),
    (re.compile(r"\\pm"), "±"),
    (re.compile(r"\\mp"), "∓"),
    (re.compile(r"\\leq?"), "≤"),
    (re.compile(r"\\geq?"), "≥"),
    (re.compile(r"\\neq?"), "≠"),
    (re.compile(r"\\approx"), "≈"),
    (re.compile(r"\\equiv"), "≡"),
    (re.compile(r"\\angle"), "∠"),
    (re.compile(r"\\parallel"), "∥"),
    (re.compile(r"\\perp"), "⊥"),
    (re.compile(r"\\because"), "∵"),
    (re.compile(r"\\therefore"), "∴"),
    (re.compile(r"\\pi"), "π"),
    (re.compile(r"\\Delta"), "Δ"),
]

_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_TEXT_RE = re.compile(r"\\text\s*\{([^{}]*)\}")
_SPACING_RE = re.compile(r"\\quad|\\qquad|\\,|\\;|\\!")
_DELIMS_RE = re.compile(r"\\[()[\]]|\$")
_LEFT_RIGHT_RE = re.compile(r"\\left|\\right")
_MULTIPLY_VARIANTS_RE = re.compile("[\u2715\u2716\u2a2f]")
_SLASH_DIVIDE_RE = re.compile("\u2215")
_MINUS_VARIANTS_RE = re.compile("[\u2212\u2796]")
_DIRTY_CHARS_RE = re.compile("[\ufffd\ue000-\uf8ff\ufff0-\uffff]")
_CONTROL_CHARS_RE = re.compile("[\x00-\x09\x0b-\x1f\x7f-\x9f]")


def normalize_math_text(text: object) -> str:
    """归一化数学题干文本 → 可直接落库/展示的纯文本（幂等）。

    处理顺序：① mojibake 还原；② LaTeX 命令还原（先剥 ``\\left/\\right``，
    否则 ``\\le`` 会误命中 ``\\left`` 前缀）；③ 符号变体统一；④ 脏字符剔除；⑤ strip。
    """
    if not isinstance(text, str) or not text:
        return ""
    s = _fix_mojibake(text)

    # \frac{a}{b} → a/b（跑两遍覆盖一层嵌套 \frac{\frac{a}{b}}{c} 的简单场景）
    for _ in range(2):
        s = _FRAC_RE.sub(r"\1/\2", s)

    # \left / \right 必须先剥：否则下面的 \le 会误命中 \left 的前缀（\left → ≤ft）
    s = _LEFT_RIGHT_RE.sub("", s)
    for pattern, symbol in _LATEX_SYMBOLS:
        s = pattern.sub(symbol, s)

    # 其余定界符与排版残留：$...$、\( \)、\[ \]、\text{..}、\quad 等
    s = _TEXT_RE.sub(r"\1", s)
    s = _SPACING_RE.sub(" ", s)
    s = _DELIMS_RE.sub("", s)

    # 符号变体统一：乘号家族 → ×（不含字母 x / 全角ｘ）；斜线除号 → ÷；减号变体 → -
    s = _MULTIPLY_VARIANTS_RE.sub("×", s)
    s = _SLASH_DIVIDE_RE.sub("÷", s)
    s = _MINUS_VARIANTS_RE.sub("-", s)

    # 替换符 / 私用区（OCR 豆腐块）/ 控制字符（保留 \n 换行）
    s = _DIRTY_CHARS_RE.sub("", s)
    s = _CONTROL_CHARS_RE.sub("", s)

    return s.strip()
