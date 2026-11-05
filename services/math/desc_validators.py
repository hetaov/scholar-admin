"""数学「语音描述出题」纯函数校验器全集（设计：docs_v1/数学/出题-语音描述出题-LangGraph编排方案.md §4/§5/§6）

全部纯函数（零 IO、零 token、零 import 服务模块），可独立单测；被
`desc_intake_graph`（语义整编）与 `desc_quiz_graph`（出题主图）作为校验层复用。

模块结构（按任务卡 T01~T08 分段落地，未实现段留待后续任务）：
- sanitize_text            T01  ASR 整句重复折叠 + 空白/标点归一（零 token）
- slot_validate            T02  槽位校验 S1~S4
- rebuild_canonical_text   T03  槽位 → 规范描述回显
- calc_answer_*            T04  数值/分数验算（V2 程序部分）
- structure_fingerprint    T05  结构同构指纹（V1）
- is_near_duplicate        T06  查重（V4）
- validate_basic / validate_extended / check_structure_card  T07/T08
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from fractions import Fraction
from typing import Any

# ===========================================================================
# T02 槽位模型 + slot_validate（S1~S4）
# ===========================================================================

# 槽位名（与方案 §4.2 一致）
SLOT_GRADE = "grade"
SLOT_SEMESTER = "semester"
SLOT_KPS = "knowledge_points"
SLOT_COUNT = "question_count"
SLOT_QTYPE = "question_type"
SLOT_DIFFICULTY = "difficulty_hint"
SLOT_SCENARIO = "scenario"
SLOT_EXTENDED = "extended_request"

# 必填槽位：kp 与题量；grade/semester 可由教材上下文补位（方案 §4.2）
REQUIRED_SLOTS = (SLOT_KPS, SLOT_COUNT)

# 上下文可补位槽位 → context 字典键
CONTEXT_FILLABLE = {SLOT_GRADE: "grade", SLOT_SEMESTER: "semester"}

_VALID_GRADES = frozenset(range(1, 7))
_VALID_SEMESTERS = frozenset({"上", "下"})
_COUNT_MAX = 20

# 低置信槽位不直接采纳（方案 §4.2：转为意图卡可编辑项），默认门限对齐 §7.3
DEFAULT_SLOT_CONFIDENCE_MIN = 0.6

# kp grounding 相似度门限（归一化后 SequenceMatcher.ratio()）
_KP_MATCH_RATIO = 0.7

# 学段粗判（S4 槽间冲突：grade 与 kp 学段不符）。kp 学段以候选节点的 grade 为准，
# 此处只做「槽位 grade 与全部命中节点 grade 都不一致」的保守判定。
_KP_GRADE_HINTS: tuple[tuple[tuple[int, ...], frozenset[str]], ...] = (
    ((1, 2), frozenset({"加减", "认识", "比较", "图形", "乘法口诀", "钟表", "元角分"})),
    ((3, 4), frozenset({"分数", "小数", "周长", "面积", "倍", "年月日", "角"})),
    ((5, 6), frozenset({"方程", "百分数", "比例", "圆柱", "圆周率", "负数", "分数乘除"})),
)


@dataclass
class Slot:
    """单槽位：value/confidence/evidence（方案 §4.2）。

    value=None 表示原话未提及（走补位/追问）；不允许编造默认值。
    """

    value: Any = None
    confidence: float = 0.0
    evidence: str = ""
    optional: bool = False


@dataclass
class SlotValidation:
    """slot_validate 结果：规范化槽位表 + 审计信息。"""

    slots: dict[str, Slot] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)          # S1~S4 拒绝码（可能重复）
    missing_required: list[str] = field(default_factory=list)
    not_grounded_kps: list[str] = field(default_factory=list)
    low_confidence_slots: list[str] = field(default_factory=list)
    dropped_slots: list[str] = field(default_factory=list)
    context_filled: list[str] = field(default_factory=list)

    @property
    def completeness(self) -> str:
        return "complete" if not self.missing_required else "missing_required"

    def to_meta(self) -> dict:
        """落库/出参用 intake_meta（不含槽位本体，槽位随请求体传）。"""
        return {
            "slot_confidence": _min_confidence(self.slots),
            "completeness": self.completeness,
            "missing_required": self.missing_required,
            "issues": self.issues,
            "not_grounded_kps": self.not_grounded_kps,
            "low_confidence_slots": self.low_confidence_slots,
            "dropped_slots": self.dropped_slots,
            "context_filled": self.context_filled,
        }


def _min_confidence(slots: dict[str, Slot]) -> float:
    values = [s.confidence for s in slots.values() if s.value is not None]
    return round(min(values), 2) if values else 0.0


def _coerce_slots(raw_slots: dict | None) -> dict[str, Slot]:
    """把 slot_extract 的 JSON 输出收敛为 Slot 表（容错：缺字段/类型错不抛）。"""
    slots: dict[str, Slot] = {}
    for name, item in (raw_slots or {}).items():
        if isinstance(item, Slot):
            slots[name] = item
            continue
        item = item if isinstance(item, dict) else {"value": item}
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        slots[name] = Slot(
            value=item.get("value"),
            confidence=max(0.0, min(1.0, conf)),
            evidence=str(item.get("evidence") or ""),
            optional=bool(item.get("optional", False)),
        )
    return slots


def _normalize_kp_name(name: str) -> str:
    return re.sub(r"[\s，、。的]", "", str(name or "")).lower()


def _match_kp_candidates(
    kp_value: str, candidates: list[dict]
) -> dict | None:
    """kp 与教材候选节点模糊匹配（归一化后互相包含**或**字符相似度 ≥0.7 即命中）。

    相似度用 difflib.SequenceMatcher（stdlib、确定性），容忍 ASR 同音噪声
    （「分数的简单家减」≈「分数的简单加减」）。取第一个命中的候选。
    """
    norm = _normalize_kp_name(kp_value)
    if not norm:
        return None
    for cand in candidates or []:
        cand_name = _normalize_kp_name(cand.get("name") or "")
        if not cand_name:
            continue
        if cand_name in norm or norm in cand_name:
            return cand
        if SequenceMatcher(None, norm, cand_name).ratio() >= _KP_MATCH_RATIO:
            return cand
    return None


def _grade_conflicts_with_kps(
    grade: int, matched: list[dict], not_grounded: list[str]
) -> bool:
    """S4 判定：槽位 grade 与**全部**已 grounding 节点的学段都不符 → 冲突。

    未 grounding 的自由文本 kp 走关键词学段粗判；无任何可判依据时不判冲突。
    """
    node_grades = [
        int(c["grade"]) for c in matched if str(c.get("grade") or "").isdigit()
    ]
    if not node_grades:
        hint_grades: set[int] = set()
        for name in not_grounded:
            for grades, keywords in _KP_GRADE_HINTS:
                if any(kw in name for kw in keywords):
                    hint_grades.update(grades)
        node_grades = sorted(hint_grades)
    if not node_grades:
        return False
    return grade not in node_grades


def slot_validate(
    raw_slots: dict | None,
    *,
    kp_candidates: list[dict] | None = None,
    context: dict | None = None,
    confidence_min: float = DEFAULT_SLOT_CONFIDENCE_MIN,
) -> SlotValidation:
    """槽位校验（纯函数，方案 §4.3 S1~S4 + 上下文补位）。

    Args:
        raw_slots: slot_extract 输出的原始槽位 JSON。
        kp_candidates: 教材候选节点 [{name, grade, node_code, summary}]（grounding 白名单）。
        context: 页面上下文 {grade, semester}（textbook_id 已知时提供，用于补位）。
        confidence_min: 低置信门限，低于它的非 optional 槽位不直接采纳。

    Returns:
        SlotValidation：规范化槽位表 + issues/missing_required 等审计信息。
        永不抛异常（容错输入由 _coerce_slots 兜底）。
    """
    v = SlotValidation()
    slots = _coerce_slots(raw_slots)

    # —— S1 枚举/边界越界 → 置空（按缺失处理，不采纳垃圾值）；合法值强制转换回写 ——
    grade_slot = slots.get(SLOT_GRADE)
    if grade_slot is not None and grade_slot.value is not None:
        try:
            g = int(grade_slot.value)
        except (TypeError, ValueError):
            g = 0
        if g not in _VALID_GRADES:
            grade_slot.value = None
            v.issues.append("S1_SLOT_INVALID")
        else:
            grade_slot.value = g
    sem_slot = slots.get(SLOT_SEMESTER)
    if sem_slot is not None and sem_slot.value not in (None, "") and sem_slot.value not in _VALID_SEMESTERS:
        sem_slot.value = None
        v.issues.append("S1_SLOT_INVALID")
    count_slot = slots.get(SLOT_COUNT)
    if count_slot is not None and count_slot.value is not None:
        try:
            n = int(count_slot.value)
        except (TypeError, ValueError):
            n = 0
        if n <= 0 or n > _COUNT_MAX:
            count_slot.value = None
            v.issues.append("S1_SLOT_INVALID")
        else:
            count_slot.value = n

    # —— 上下文补位（grade/semester 缺失且 context 提供时补位，不算缺失） ——
    ctx = context or {}
    for slot_name, ctx_key in CONTEXT_FILLABLE.items():
        slot = slots.get(slot_name)
        if (slot is None or slot.value in (None, "")) and ctx.get(ctx_key) is not None:
            slots[slot_name] = Slot(
                value=ctx[ctx_key],
                confidence=1.0,
                evidence=f"context:{ctx_key}",
                optional=True,
            )
            v.context_filled.append(slot_name)

    # —— S2 kp grounding（失败降级自由文本，不阻断） ——
    kp_slot = slots.get(SLOT_KPS)
    matched_nodes: list[dict] = []
    if kp_slot is not None:
        kp_values = kp_slot.value if isinstance(kp_slot.value, list) else (
            [kp_slot.value] if kp_slot.value else []
        )
        grounded: list[str] = []
        for kp in kp_values:
            cand = _match_kp_candidates(str(kp), kp_candidates or [])
            if cand is not None:
                grounded.append(str(cand.get("name") or kp))
                matched_nodes.append(cand)
            else:
                grounded.append(str(kp))
                v.not_grounded_kps.append(str(kp))
                if not v.issues or "S2_KP_NOT_GROUNDED" not in v.issues:
                    v.issues.append("S2_KP_NOT_GROUNDED")
        kp_slot.value = grounded

    # —— S4 槽间冲突：grade 与 kp 学段不符 → 以教材节点为准，丢弃 grade 槽 ——
    if (
        grade_slot is not None
        and grade_slot.value is not None
        and (matched_nodes or v.not_grounded_kps)
    ):
        try:
            g = int(grade_slot.value)
        except (TypeError, ValueError):
            g = 0
        if g and _grade_conflicts_with_kps(g, matched_nodes, v.not_grounded_kps):
            grade_slot.value = None
            v.dropped_slots.append(SLOT_GRADE)
            v.issues.append("S4_CONFLICT")

    # —— 低置信槽位（非 optional 且 value 非空）不直接采纳，转意图卡可编辑项 ——
    for name, slot in slots.items():
        if slot.value is not None and not slot.optional and slot.confidence < confidence_min:
            if name not in v.low_confidence_slots:
                v.low_confidence_slots.append(name)

    # —— S3 必填缺失（补位后仍为空） ——
    for name in REQUIRED_SLOTS:
        slot = slots.get(name)
        value = slot.value if slot is not None else None
        empty = value is None or value == "" or value == []
        if empty:
            v.missing_required.append(name)
            if "S3_REQUIRED_MISSING" not in v.issues:
                v.issues.append("S3_REQUIRED_MISSING")

    v.slots = slots
    return v


# ===========================================================================
# T03 rebuild：槽位 → 规范描述回显（零 token 纯模板）
# ===========================================================================


def _slot_value(slots: dict, name: str) -> Any:
    slot = slots.get(name)
    return slot.value if isinstance(slot, Slot) else (slot or {}).get("value")


def rebuild_canonical_text(slots: dict) -> str:
    """槽位表 → 规范描述文本（方案 §4.4）。

    例："3年级上册 · 分数的简单加减 · 5道应用题（分蛋糕情境）"
    规则：只拼接有值的槽；optional 槽有值才拼；无任何有效槽返回空串。
    接受 Slot 表（SlotValidation.slots）或原始 dict（slot_extract 直出）。
    """
    segments: list[str] = []

    grade = _slot_value(slots, SLOT_GRADE)
    semester = _slot_value(slots, SLOT_SEMESTER)
    if grade is not None:
        seg = f"{grade}年级"
        if semester:
            seg += str(semester)
        segments.append(seg)

    kps = _slot_value(slots, SLOT_KPS)
    if kps:
        kp_list = kps if isinstance(kps, list) else [kps]
        kp_str = "、".join(str(k) for k in kp_list if k)
        if kp_str:
            segments.append(kp_str)

    count = _slot_value(slots, SLOT_COUNT)
    if count is not None:
        qtype = _slot_value(slots, SLOT_QTYPE)
        segments.append(f"{count}道{qtype or '题'}")

    if not segments:
        return ""

    text = " · ".join(segments)

    scenario = _slot_value(slots, SLOT_SCENARIO)
    if scenario:
        text += f"（{scenario}情境）"
    return text


# ===========================================================================
# T04 数值/分数验算器（V2 程序部分：纯四则表达式的确定性求解）
# ===========================================================================

# 数字字面量：整数 / 小数 / 分数（a/b）。优先匹配分数（含整数/2 形式）。
_NUM_RE = re.compile(r"\d+(?:\.\d+)?(?:/\d+)?")


def _to_fraction(token: str) -> Fraction:
    """字面量 → Fraction（支持 3 / 2.7 / 3/7）。"""
    if "/" in token:
        a, b = token.split("/", 1)
        return Fraction(int(a), int(b))
    if "." in token:
        # 小数用 Fraction(十进制字符串) 精确表示，避免二进制浮点误差
        return Fraction(token)
    return Fraction(int(token))


def calc_answer(expression: str) -> Fraction | None:
    """四则表达式精确求值（+ - ×÷、括号、整数/小数/分数）。

    自研递归下降解析（**禁用 eval**）；任何解析/除零/未知字符 → None（交双评路径）。
    运算符支持：+ - * / × x X ÷ （全角与字母变体归一）。
    """
    if not expression:
        return None
    expr = (
        str(expression)
        .replace("×", "*")
        .replace("÷", "/")
        .replace("x", "*")
        .replace("X", "*")
        .replace("（", "(")
        .replace("）", ")")
        .replace("＝", "=")
        .replace("－", "-")
    )
    expr = expr.split("=", 1)[0]  # 去掉「= ?」尾巴
    tokens = _NUM_RE.findall(expr)
    if not tokens:
        return None
    # 递归下降：expr := term (('+'|'-') term)* ; term := factor (('*'|'/') factor)*
    pos = 0
    n = len(expr)

    def skip_ws() -> None:
        nonlocal pos
        while pos < n and expr[pos] in " \t":
            pos += 1

    def parse_factor() -> Fraction | None:
        nonlocal pos
        skip_ws()
        if pos < n and expr[pos] == "(":
            pos += 1
            val = parse_expr()
            skip_ws()
            if pos >= n or expr[pos] != ")":
                return None
            pos += 1
            return val
        m = _NUM_RE.match(expr, pos)
        if not m:
            return None
        pos = m.end()
        try:
            return _to_fraction(m.group(0))
        except (ZeroDivisionError, ValueError):
            return None

    def parse_term() -> Fraction | None:
        nonlocal pos
        val = parse_factor()
        if val is None:
            return None
        while True:
            skip_ws()
            if pos < n and expr[pos] in "*/":
                op = expr[pos]
                pos += 1
                rhs = parse_factor()
                if rhs is None:
                    return None
                try:
                    val = val * rhs if op == "*" else val / rhs
                except ZeroDivisionError:
                    return None
            else:
                return val

    def parse_expr() -> Fraction | None:
        nonlocal pos
        val = parse_term()
        if val is None:
            return None
        while True:
            skip_ws()
            if pos < n and expr[pos] in "+-":
                op = expr[pos]
                pos += 1
                rhs = parse_term()
                if rhs is None:
                    return None
                val = val + rhs if op == "+" else val - rhs
            else:
                return val

    result = parse_expr()
    if result is None:
        return None
    skip_ws()
    if pos != n:  # 有未消费字符（非法结构，如连续运算符）→ 不可程序判定
        return None
    return result


def _parse_answer_value(answer: Any) -> Fraction | None:
    """答案字段 → Fraction（接受 int/float/str；str 支持 '3/7' '0.75' '2'）。"""
    if isinstance(answer, bool) or answer is None:
        return None
    if isinstance(answer, int):
        return Fraction(answer)
    if isinstance(answer, float):
        return Fraction(str(answer))
    s = str(answer).strip().replace("×", "*")
    m = _NUM_RE.fullmatch(s)
    if not m:
        return None
    try:
        return _to_fraction(s)
    except (ZeroDivisionError, ValueError):
        return None


def extract_expression(question: str) -> str | None:
    """从题干提取可程序求解的四则子式（V2 第一层；提取不到返回 None → 双评路径）。

    策略：取题干中最长的「数字 运算符 数字」连续片段（允许括号与分数），
    要求存在**分数字面量以外**的运算符（fingerprint 口径：'/' 属于 a/b 字面量）
    且操作数 ≥2——防止叙述中的孤立分数「3/7 米」被误判为除法算式。
    纯数字（如答案选项）不算表达式。
    """
    if not question:
        return None
    text = (
        str(question)
        .replace("×", "*").replace("÷", "/")
        .replace("x", "*").replace("X", "*")
        .replace("（", "(").replace("）", ")")
    )
    # 候选片段：以数字开头/结尾、只含数字/运算符/括号/空白
    cand_re = re.compile(r"[\d.][\d.+\-*/() ]*[\d.)]")
    best: str | None = None
    for m in cand_re.finditer(text):
        seg = m.group(0).strip()
        # fingerprint 口径下存在真实运算符（排除分数字面量内的 /）且 ≥2 个操作数
        if operator_sequence(seg) and fingerprint(seg).count("#") >= 2:
            if best is None or len(seg) > len(best):
                best = seg
    return best


def check_answer(question: str, answer: Any) -> bool | None:
    """程序验算：题干表达式求值与 answer 比对（V2）。

    Returns:
        True/False：可程序判定；None：无法程序求解（应用题/无算式）→ 双评路径。
    """
    expr = extract_expression(question)
    if expr is None:
        return None
    expected = calc_answer(expr)
    if expected is None:
        return None
    actual = _parse_answer_value(answer)
    if actual is None:
        return None
    return expected == actual


# ===========================================================================
# T05 结构同构指纹（V1）
# ===========================================================================

# 题干模板化：数字（含小数/分数）→ #；保留运算符与括号结构。
_TEMPLATE_NUM_RE = re.compile(r"\d+(?:\.\d+)?(?:/\d+)?")


def fingerprint(question: str) -> str:
    """题干 → 结构模板串（操作数归一为 #，空白折叠，运算符/括号保留）。

    「3/7 + 2/7 = ?」→「#+#=?」；「(2 + 3) * 4」→「(#+#)*#」。
    分数「a/b」整体归一为一个操作数（其「/」是字面量成分，不计运算符，
    与 operator_sequence 语义一致）。空白全部去除（空白对结构比对无意义，
    且 ASR 断句导致的空格差异不应产生不同指纹）。
    """
    text = (
        str(question or "")
        .replace("×", "*").replace("÷", "/")
        .replace("x", "*").replace("X", "*")
        .replace("＋", "+").replace("－", "-")
        .replace("（", "(").replace("）", ")")
    )
    # 先去空白再归一数字：分数字面量内的空格（「3 / 7」）不得改变分词结果
    text = re.sub(r"\s+", "", text)
    return _TEMPLATE_NUM_RE.sub("#", text)


def operator_sequence(question: str) -> list[str]:
    """题干中按序出现的运算符序列（V1 的核心比对物）。

    「3/7 + 2/7」→ ['+']；「(2 + 3) * 4」→ ['+', '*']。
    分数「a/b」的 '/' 不计入（属于字面量）——由 _TEMPLATE_NUM_RE 先吃掉。
    """
    fp = fingerprint(question)
    return [ch for ch in fp if ch in "+-*/"]


def same_structure(q1: str, q2: str) -> bool:
    """V1 同构判定：运算符序列一致且运算步数（操作数个数）一致。"""
    seq1, seq2 = operator_sequence(q1), operator_sequence(q2)
    if seq1 != seq2:
        return False
    # 操作数个数 = 模板中 # 的个数
    return fingerprint(q1).count("#") == fingerprint(q2).count("#")


# ===========================================================================
# T06 查重（V4）
# ===========================================================================


def _normalize_for_dup(text: str) -> str:
    """查重归一：去空白与标点，保留数字/字母/汉字（数字参与比对——变式换数后不应判重）。"""
    return re.sub(r"[\s，。、；：？！,.;:?!()（）「」『』\"'=+\-*/×÷]", "", str(text or ""))


def near_duplicate(a: str, b: str, *, threshold: float = 0.98) -> bool:
    """V4 题间查重：归一化后完全相同或相似度 ≥ threshold（默认 0.98）。

    阈值取高：同类型变式题本来就该高度相似（仅换数值），只有**实质相同**
    （换汤不换药到数值都一样）才算重复。threshold 可由调用方调低用于
    「与源意图描述查重」场景。
    """
    na, nb = _normalize_for_dup(a), _normalize_for_dup(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


# ===========================================================================
# T07 易错位命中（V3）+ 拔高校验（E1~E5）
# ===========================================================================


def check_trap_assignment(
    items: list[dict], *, min_ratio: float = 0.5
) -> tuple[bool, str]:
    """V3 易错位设点检查（方案 §5.3/§5.4）。

    可程序判定的不变量：
    1. target_error_step 允许为 0（未设点）；非零值必须是 ≥1 的整数；
    2. L1 题中设点题（target_error_step > 0）占比 ≥ min_ratio（默认 1/2）。

    Returns:
        (ok, reason)：ok=False 时 reason 为拒绝码（V3_TRAP_NOT_HIT）。
    """
    l1_items = [it for it in items or [] if it.get("variant_level") == "L1"]
    if not l1_items:
        return True, ""
    for it in l1_items:
        step = it.get("target_error_step", 0)
        if step == 0:
            continue
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            return False, "V3_TRAP_NOT_HIT"
    hit = sum(1 for it in l1_items if it.get("target_error_step", 0) > 0)
    if hit / len(l1_items) < min_ratio:
        return False, "V3_TRAP_NOT_HIT"
    return True, ""


def validate_extended_item(
    item: dict,
    *,
    base_kps: list[str],
    plan_new_kp: str,
    plan_difficulty_delta: int,
    base_difficulty: int,
    base_questions: list[str] | None = None,
) -> list[str]:
    """拔高题四层约束校验（方案 §6.4，E1~E5；E4/E5 复用 V4/V2）。

    Returns:
        拒绝码列表（空 = 通过）。永不抛异常。
    """
    codes: list[str] = []
    question = str(item.get("question") or "")
    kp_name = str(item.get("knowledge_point_name") or "")

    # E1 kp 集合差 ⊆ 基础 kp ∪ {本档 new_kp}
    allowed = {str(k) for k in base_kps} | {plan_new_kp}
    allowed_norm = {_normalize_kp_name(k) for k in allowed}
    if kp_name and _normalize_kp_name(kp_name) not in allowed_norm:
        codes.append("E1_KP_DRIFT")

    # E2 new_kp 术语必须真实出现于题干（归一化后包含或相似度）
    q_norm = _normalize_for_dup(question)
    kp_norm = _normalize_kp_name(plan_new_kp)
    if kp_norm and kp_norm not in q_norm:
        best = max(
            (SequenceMatcher(None, kp_norm, _normalize_for_dup(seg)).ratio()
             for seg in re.split(r"[，。;；,.\n]", q_norm) if seg),
            default=0.0,
        )
        if best < 0.5:
            codes.append("E2_KPO_NOT_USED")

    # E3 难度 = 基础档 + delta（越界拒，不允许跳档）
    try:
        diff = int(item.get("difficulty"))
    except (TypeError, ValueError):
        diff = 0
    if diff != base_difficulty + plan_difficulty_delta:
        codes.append("E3_DIFFICULTY_JUMP")

    # E4 与基础题及彼此查重（此处仅对基础题查重；题间查重由调用方两两执行）
    for base_q in base_questions or []:
        if near_duplicate(question, str(base_q)):
            codes.append("E4_DUP")
            break

    # E5 答案可计算性
    verdict = check_answer(question, item.get("answer"))
    if verdict is False:
        codes.append("E5_ANSWER_WRONG")
    return codes


# ===========================================================================
# T08 结构卡程序校验
# ===========================================================================


def _answer_leak_variants(answer: Any) -> list[str]:
    """answer 的泄露探测形态：原样串 + 分数的十进制等值（去空白）。"""
    if isinstance(answer, bool) or answer is None:
        return []
    s = re.sub(r"\s+", "", str(answer))
    variants = [s] if s else []
    try:
        frac = _parse_answer_value(answer)
        if frac is not None and frac.denominator != 1:
            # 分数补充小数形态（截 4 位，防 1/3 无限循环串）
            variants.append(str(round(float(frac), 4)))
        elif frac is not None and frac.denominator == 1 and s != str(frac.numerator):
            variants.append(str(frac.numerator))
    except Exception:  # noqa: BLE001 —— 探测失败不影响主流程
        pass
    return [v for v in variants if v]


def check_answer_leakage(item: dict) -> bool:
    """V5 答案泄露：answer 数值不得出现在 question / rationale（去空白双查）。

    Returns:
        True = 泄露（拒绝）；False = 未发现泄露。answer 无法解析时不判泄露
        （该题由 V2 兜底）。
    """
    variants = _answer_leak_variants(item.get("answer"))
    if not variants:
        return False
    haystack = re.sub(
        r"\s+", "",
        f"{item.get('question') or ''}{item.get('rationale') or ''}",
    )
    if not haystack:
        return False
    return any(v in haystack for v in variants if len(v) >= 2 or v.isdigit())


def check_kp_in_scope(kp_name: Any, allowed_kps: list[str] | None) -> bool:
    """V6 考点越界：knowledge_point_name 必须落在 grounded_kps 名集内（归一化包含）。

    allowed_kps 为空 → 不判越界（返回 True，交上层按无约束处理）。
    """
    if not allowed_kps:
        return True
    name = _normalize_kp_name(str(kp_name or ""))
    if not name:
        return True  # 缺失字段交 schema 校验/上层，不在此判越界
    allowed_norm = {_normalize_kp_name(str(k)) for k in allowed_kps if k}
    if name in allowed_norm:
        return True
    return any(a in name or name in a for a in allowed_norm)


def check_structure_card(card: dict) -> list[str]:
    """题目解剖卡校验（方案 §5.2，零 token）。

    Returns:
        问题码列表（空 = 通过）：CARD_NO_STEPS / CARD_TOO_MANY_STEPS /
        CARD_NO_TRAPS / CARD_BAD_TRAP_STEP / CARD_BAD_DIFFICULTY。
    """
    issues: list[str] = []
    steps = card.get("step_sequence") or []
    if not isinstance(steps, list) or not steps:
        return ["CARD_NO_STEPS"]
    if len(steps) > 6:
        issues.append("CARD_TOO_MANY_STEPS")

    traps = card.get("error_prone_steps") or []
    if not isinstance(traps, list) or not traps:
        issues.append("CARD_NO_TRAPS")
    else:
        for t in traps:
            step = t.get("step") if isinstance(t, dict) else None
            if not isinstance(step, int) or isinstance(step, bool) or not (1 <= step <= len(steps)):
                issues.append("CARD_BAD_TRAP_STEP")
                break

    band = card.get("difficulty_band")
    if not isinstance(band, int) or isinstance(band, bool) or not (1 <= band <= 5):
        issues.append("CARD_BAD_DIFFICULTY")
    return issues

# ---------------------------------------------------------------------------
# T01 sanitize：整句重复折叠 + 空白/标点归一
# ---------------------------------------------------------------------------

# 句级分隔符（保留分隔符本身，用于重建文本）。含逗号/顿号/冒号：
# ASR 整句重复常以逗号衔接（「我想要分数加减法，我想要分数加减法。」），
# 不把逗号当句读会漏折叠；正常文本中相邻子句逐字相同的概率可忽略。
_SENT_SPLIT_RE = re.compile(r"([。！？；，、：\n])")

# 长重复串折叠的最小重复单元长度。
# 取 4：口语合法叠词最长约 4 字（AABB 如「清清楚楚」不是 ABAB 结构不会命中；
# ABAB 动词叠用如「研究研究」由 _LEGIT_ABAB 白名单豁免）。
_RUN_MIN_LEN = 4

# 合法 ABAB 动词叠用白名单（命中则不折叠，防止把正常口语当 ASR 重复）
_LEGIT_ABAB = frozenset(
    {
        "研究研究", "商量商量", "讨论讨论", "考虑考虑", "检查检查",
        "尝试尝试", "打扫打扫", "准备准备", "整理整理", "浏览浏览",
    }
)

_RUN_REPEAT_RE = re.compile(r"(.{%d,}?)\1+" % _RUN_MIN_LEN)


@dataclass
class SanitizeResult:
    """sanitize_text 的结果：折叠后文本 + 可审计元信息。"""

    text: str
    collapsed_sentence_repeats: int = 0
    collapsed_run_repeats: int = 0
    changed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.changed = (
            self.collapsed_sentence_repeats > 0 or self.collapsed_run_repeats > 0
        )


def _collapse_adjacent_sentences(text: str) -> tuple[str, int]:
    """相邻完全相同句折叠：按句读切分为（句内容, 句读）对，相邻句内容相同则整对丢弃。

    「出五道分数应用题。出五道分数应用题。」→「出五道分数应用题。」
    「我想要分数加减法，我想要分数加减法。」→「我想要分数加减法。」
    比较前先剥离空白（「出 五道」与「出五道」视为相同句）。
    """
    tokens = _SENT_SPLIT_RE.split(text)
    # tokens 交替为 [句内容, 句读, 句内容, 句读, ..., 句内容?]
    out: list[tuple[str, str]] = []  # (句内容, 句读)
    collapsed = 0
    for i in range(0, len(tokens), 2):
        content = tokens[i]
        delim = tokens[i + 1] if i + 1 < len(tokens) else ""
        norm = re.sub(r"\s+", "", content)
        if norm and out and re.sub(r"\s+", "", out[-1][0]) == norm:
            # 与上一句逐字相同：丢弃本句，并把上一句的句读替换为本次的句读
            # （否则「A，A。」折叠后残留尾逗号）
            out[-1] = (out[-1][0], delim)
            collapsed += 1
            continue
        out.append((content, delim))
    return "".join(c + d for c, d in out), collapsed


def _collapse_run_repeats(text: str) -> tuple[str, int]:
    """相邻长重复串折叠：「我想要分数加减法我想要分数加减法」→「我想要分数加减法」。

    仅折叠长度 ≥ _RUN_MIN_LEN 的相邻重复单元；命中合法 ABAB 白名单的单元跳过。
    循环应用直到不再变化（处理三重及以上重复：A A A → A）。
    """
    collapsed_total = 0
    current = text
    while True:
        units_removed = 0

        def _sub(m: re.Match) -> str:
            nonlocal units_removed
            unit = m.group(1)
            if unit in _LEGIT_ABAB:
                return m.group(0)
            units_removed += len(m.group(0)) // len(unit) - 1
            return unit

        current = _RUN_REPEAT_RE.sub(_sub, current)
        if units_removed == 0:
            break
        collapsed_total += units_removed
    return current, collapsed_total


def sanitize_text(raw: str | None) -> SanitizeResult:
    """ASR 文本机械整编（纯函数，零 token）。

    只做三类确定性操作：
    1. 空白归一：连续空白折叠为单个空格、去首尾；
    2. 相邻完全相同句折叠（句读切分，标点保留）；
    3. 相邻长重复串折叠（单元 ≥4 字，ABAB 白名单豁免，循环应用处理多重重复）。

    **不做**：同音纠错、数字归一、补全、句意改写——这些是 slot_extract（语义层）
    的职责，本函数对文本的任何变更都必须是确定性的、可由 meta 审计的。
    """
    text = (raw or "").strip()
    if not text:
        return SanitizeResult(text="")

    text = re.sub(r"\s+", " ", text).strip()

    text, sent_collapsed = _collapse_adjacent_sentences(text)
    text, run_collapsed = _collapse_run_repeats(text)
    text = re.sub(r"\s+", " ", text).strip()

    return SanitizeResult(
        text=text,
        collapsed_sentence_repeats=sent_collapsed,
        collapsed_run_repeats=run_collapsed,
    )
