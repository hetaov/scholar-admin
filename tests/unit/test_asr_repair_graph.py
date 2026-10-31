"""单元测试：ASR 判前修复子图 + `run_round_turn` 接线（第六期）

设计：docs_v1/扩展/第六期-ASR断句修复与判句回显-修改方案-v1.md §4
触发：docs_v1/扩展/issue.md（语音造句被 ASR 切成碎片 → 判分反馈失真）

分三段：
| 段 | 覆盖 | 对应红线 |
|---|---|---|
| A 纯函数 | `norm_tokens` / `dup_excess` / `precheck` / `guard`（5 判据逐条） | R15 |
| B 子图 | `run_asr_repair` 三条路径 + 失败回落 | R16 / R18 |
| C 接线 | `run_round_turn` 判分口径、落库、返回体 | R17 / R18 |

> 「同文件原则」：一期 R1 / R9、三期 R10~R14 都要求护栏**彼此同文件**放置
> （防拦截器绕过）。本期 R15~R18 沿用同一原则，集中在本文件。
"""
from __future__ import annotations

import asyncio
import json

import pytest

import config
from services.english import asr_repair_graph as ag
from services.english import extension_round as er
from services.english.asr_repair_graph import (
    guard,
    norm_tokens,
    precheck,
    run_asr_repair,
)
from services.providers.extension_llm import ExtensionError

# 造数与桩复用三期（该文件导出即供它文件复用，避免夹具漂移）
from tests.fakes.fake_db import FakeDB
from tests.unit.test_extension_round import (  # noqa: E402
    ORIGINAL,
    POINTS,
    SCHOLAR,
    SENTENCE,
    TRANSLATION,
    _RUBRIC_PASS,
    _install_judge_stub,
    _install_prompt_stub,
    _start,
)


def _run(coro):
    return asyncio.run(coro)


def _enable(monkeypatch) -> None:
    monkeypatch.setattr(config, "EXTENSION_ROUND_ASR_REPAIR_ENABLED", 1)


def _disable(monkeypatch) -> None:
    monkeypatch.setattr(config, "EXTENSION_ROUND_ASR_REPAIR_ENABLED", 0)


def _install_repair_stub(monkeypatch, repaired: str | None = None,
                         error: Exception | None = None):
    """替换修复 LLM：`repaired` 为 None 时回原文；`error` 非空则抛出。"""
    calls: dict = {"count": 0, "messages": []}

    async def _fake(messages, timeout_seconds=None, thinking_disabled=None):  # noqa: ANN001
        calls["count"] += 1
        calls["messages"].append(messages)
        if error is not None:
            raise error
        text = repaired if repaired is not None else "@@RAW@@"
        if text == "@@RAW@@":
            # 从 messages 里取回原文，模拟「无需修复」
            user = next(m["content"] for m in messages if m["role"] == "user")
            text = user.split("\n", 1)[1].strip()
        return json.dumps({"repaired": text, "dropped": [], "notes": "stub"},
                          ensure_ascii=False)

    monkeypatch.setattr(ag, "call_asr_repair_llm", _fake)
    return calls


# ===========================================================================
# A · 纯函数
# ===========================================================================


def test_norm_tokens_lowercases_and_drops_punct():
    assert norm_tokens("He finally, took-up!") == ["he", "finally", "took", "up"]
    assert norm_tokens("Don't go.") == ["don't", "go"]
    assert norm_tokens("") == []


def test_dup_excess_counts_runs():
    assert ag.dup_excess(["he", "he", "finally"]) == {"he": 1}
    assert ag.dup_excess(["had", "had", "had"]) == {"had": 2}
    assert ag.dup_excess(["he", "went", "he"]) == {}  # 非紧邻不算富余
    assert ag.dup_excess([]) == {}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("He finally. Took up the question.", ["fragmented"]),
        ("He he finally took up it.", ["repeated"]),
        ("He finally took up the question", ["unpunctuated"]),
        (" He  finally took up it.", ["whitespace"]),
        ("He finally took up the question.", []),  # 干净 → 无需修复
    ],
)
def test_precheck_reasons(text, expected):
    reasons = precheck(text)
    for r in expected:
        assert r in reasons, f"{text!r} 应命中 {r}，实际 {reasons}"
    if expected == []:
        assert reasons == []


def test_has_fragment_ignores_titles_and_initials():
    """A 轨实测：裸 `[.!?]\\s+\\S` 对真实教材句误报 10%（`Mrs.` 等），必须排除。"""
    assert ag.has_fragment("He finally. Took up the question.") is True
    assert ag.has_fragment("Mrs. Smith's living room is large.") is False
    assert ag.has_fragment("Mr. Smith went home.") is False
    assert ag.has_fragment("J. Smith went home.") is False
    assert ag.has_fragment("He finally took up the question.") is False


def test_has_fragment_ignores_multi_period_abbrev():
    """A 轨 120 条实测：`800 B.C. has just had` 被误判断句。

    根因：正则 `[A-Za-z][A-Za-z.'’]*` **贪婪**，把 `B.C` 整个吃进来变成 `b.c`，
    而白名单里只有 `bc` 之类没有点的形式 → 必须把多点缩写逐个列全。
    """
    assert ag.has_fragment(
        "The mummy of an Egyptian woman who died in 800 B.C. has just had an operation."
    ) is False
    assert ag.has_fragment("He arrived at 8 a.m. and left at noon.") is False
    assert ag.has_fragment("She works in the U.S. and travels often.") is False


def test_precheck_ignores_trailing_quote_before_punctuation():
    """A 轨 120 条实测：`"Macaroni ... years."` 被误判 unpunctuated。

    根因：末字符是 `"` 不是 `.` ⇒ 必须先剥掉收尾引号/括号再判句末标点。
    """
    assert "unpunctuated" not in precheck(
        '"Macaroni has been grown in this area for over six hundred years."'
    )
    assert precheck(
        '"Macaroni has been grown in this area for over six hundred years."'
    ) == []
    # 真的缺标点仍然要判出来
    assert "unpunctuated" in precheck("He went home")


def test_precheck_clean_textbook_sentence_has_no_reasons():
    """回归：`Mrs.` 句不得触发任何修复（否则白白多一次 LLM 调用）。"""
    assert precheck("Mrs. Smith's living room is large.") == []
    assert ag.needs_llm(precheck("Mrs. Smith's living room is large.")) is False


def test_precheck_grammatic_redup_is_not_repeated():
    """G4 前置：`had had` 是正确英语 ⇒ 不得判为「需去重」，否则会白跑一次 LLM。"""
    assert "repeated" not in precheck("She said she had had enough.")
    assert ag.needs_llm(precheck("She said she had had enough.")) is False


def test_needs_llm_only_for_fragmented_or_repeated():
    assert ag.needs_llm(["fragmented"]) is True
    assert ag.needs_llm(["repeated"]) is True
    assert ag.needs_llm(["unpunctuated"]) is False
    assert ag.needs_llm(["whitespace"]) is False


# ---- 护栏 5 判据（R15）------------------------------------------------------


@pytest.mark.parametrize(
    "raw,repaired,ok",
    [
        # 允许：合并断句（词序列不变）
        ("He finally. Took up the question.", "He finally took up the question.", True),
        # 允许：删紧邻重复
        ("he he finally", "He finally.", True),
        ("He he finally took took up it.", "He finally took up it.", True),
        # 允许：删填充词
        ("um he went home", "He went home.", True),
        # 允许：原样
        ("He went home.", "He went home.", True),
        # G2 拒绝：加词 / 换词 / 重排
        ("He went home", "He went to home.", False),
        ("He go to school", "He went to school.", False),
        ("He took up it", "He up took it.", False),
        # G3① 拒绝：删实词（无据）
        ("He took up the question", "He took up question.", False),
        # G4 拒绝：破坏合法语法重复
        ("She said she had had enough.", "She said she had enough.", False),
        ("the book that that boy read", "the book that boy read.", False),
        # G1 拒绝：空
        ("He went home.", "   ", False),
    ],
)
def test_guard_matrix(raw, repaired, ok):
    got, reason = guard(raw, repaired)
    assert got is ok, f"{raw!r} → {repaired!r} 期望 ok={ok}，实际 {got}（{reason}）"


def test_guard_g5_too_long():
    long_text = "he " * 300
    ok, reason = guard(long_text, long_text)
    assert ok is False and reason == "G5_TOO_LONG"


def test_guard_reason_codes_are_stable():
    """拒绝码是落库留痕与实验页观测的口径，不可随意改名。"""
    assert guard("He went home", "He went to home.")[1] == "G2_NOT_SUBSEQUENCE"
    assert guard("He took up the question", "He took up question.")[1] == (
        "G3_UNJUSTIFIED_DELETION"
    )
    assert guard("She said she had had enough.", "She said she had enough.")[1] == (
        "G4_GRAMMATIC_REDUP_BROKEN"
    )


def test_rule_fix_only_touches_punctuation():
    assert ag.rule_fix_text("he went home") == "He went home."
    assert ag.rule_fix_text("He went home.") == "He went home."
    assert ag.rule_fix_text("   ") == "   "


# ===========================================================================
# B · 子图 run_asr_repair
# ===========================================================================


def test_disabled_returns_none_payload(monkeypatch):
    """R18：关开关 → repaired_input=None，调用方据此不加字段 ⇒ 逐字不变。"""
    _disable(monkeypatch)
    calls = _install_repair_stub(monkeypatch)
    out = _run(run_asr_repair("He finally. Took up"))
    assert out == {"repaired_input": None, "asr_repair": None}
    assert calls["count"] == 0


def test_clean_input_does_not_call_llm(monkeypatch):
    """干净输入走 Tier 0 快路径：零 LLM、零改动。"""
    _enable(monkeypatch)
    calls = _install_repair_stub(monkeypatch)
    out = _run(run_asr_repair("He finally took up the question."))
    assert calls["count"] == 0
    assert out["repaired_input"] == "He finally took up the question."
    assert out["asr_repair"]["source"] == ag.SOURCE_NONE
    assert out["asr_repair"]["applied"] is False


def test_unpunctuated_goes_rule_tier_without_llm(monkeypatch):
    """仅缺标点 → Tier 1 规则层，仍然零 LLM。"""
    _enable(monkeypatch)
    calls = _install_repair_stub(monkeypatch)
    out = _run(run_asr_repair("he went home"))
    assert calls["count"] == 0
    assert out["repaired_input"] == "He went home."
    assert out["asr_repair"]["source"] == ag.SOURCE_RULE
    assert out["asr_repair"]["applied"] is True


def test_llm_merges_fragments(monkeypatch):
    _enable(monkeypatch)
    _install_repair_stub(monkeypatch, repaired="He finally took up the question.")
    out = _run(run_asr_repair("He finally. Took up the question"))
    assert out["repaired_input"] == "He finally took up the question."
    meta = out["asr_repair"]
    assert meta["source"] == ag.SOURCE_LLM
    assert meta["applied"] is True
    assert meta["fallback"] is False
    assert "fragmented" in meta["reasons"]


def test_llm_failure_falls_back_and_marks(monkeypatch):
    """R16：失败不阻断 —— 回落仍产出可用文本，且 fallback=True 留痕。"""
    _enable(monkeypatch)
    _install_repair_stub(monkeypatch, error=ExtensionError("LLM_TIMEOUT", "llm", "t"))
    out = _run(run_asr_repair("He finally. Took up the question"))
    assert out["repaired_input"]
    meta = out["asr_repair"]
    assert meta["fallback"] is True
    assert meta["source"] == ag.SOURCE_FALLBACK
    assert meta["error"] == "LLM_TIMEOUT"


def test_guard_rejection_falls_back(monkeypatch):
    """护栏拒绝 → 回落，并把 guard_reason 落进元信息（实验页可见）。"""
    _enable(monkeypatch)
    _install_repair_stub(monkeypatch, repaired="He finally took up the question well.")
    out = _run(run_asr_repair("He finally. Took up the question"))
    meta = out["asr_repair"]
    assert meta["source"] == ag.SOURCE_FALLBACK
    assert meta["fallback"] is True
    assert meta["guard_reason"] == "G2_NOT_SUBSEQUENCE"
    # 回落的文本必须仍是原文词序列的子序列（G2 结构性保证）
    assert guard("He finally. Took up the question", out["repaired_input"])[0] is True


def test_never_raises_on_unexpected_error(monkeypatch):
    _enable(monkeypatch)

    async def _boom(*a, **kw):  # noqa: ANN001
        raise RuntimeError("unexpected")

    monkeypatch.setattr(ag, "call_asr_repair_llm", _boom)
    raw = "He finally. Took up"
    out = _run(run_asr_repair(raw))
    assert out["asr_repair"]["fallback"] is True
    # 真正的不变量：回落文本必须仍是原文词序列的子序列（G2），且非空可用
    assert out["repaired_input"]
    assert guard(raw, out["repaired_input"])[0] is True


def test_empty_input_passthrough(monkeypatch):
    _enable(monkeypatch)
    calls = _install_repair_stub(monkeypatch)
    out = _run(run_asr_repair(""))
    assert out["repaired_input"] == ""
    assert out["asr_repair"]["applied"] is False
    assert calls["count"] == 0


# ===========================================================================
# C · 接线 run_round_turn（R17 / R18）
# ===========================================================================


def _turn(db, monkeypatch, raw: str, *, repaired: str | None = None,
          error: Exception | None = None):
    """建会话 → 提交一轮（判分桩 + 修复桩已装）。"""
    _start(db, monkeypatch, max_turns=3)
    judge_calls = _install_judge_stub(monkeypatch)
    prompt_calls = _install_prompt_stub(monkeypatch)
    _install_repair_stub(monkeypatch, repaired=repaired, error=error)
    result = _run(
        er.run_round_turn(
            db, round_id=_round_id(db), scholar_id=SCHOLAR, user_input=raw
        )
    )
    return result, judge_calls, prompt_calls


def _round_id(db) -> str:
    docs = db.all(config.EXTENSION_ROUND_COLLECTION)
    return docs[-1]["round_id"]


def test_turn_judges_repaired_text(monkeypatch):
    """D5：判分必须用修复后文本（否则碎片直接进 rubric）。"""
    _enable(monkeypatch)
    db = FakeDB()
    _, judge_calls, _ = _turn(
        db, monkeypatch, "He finally. Took part in the discussion",
        repaired="He finally took part in the discussion.",
    )
    assert judge_calls["answers"][0] == "He finally took part in the discussion."


def test_turn_prev_turn_uses_repaired_text(monkeypatch):
    """D5：出题看到的也应是修复后句子，否则 focus_hint 仍盯着碎片。"""
    _enable(monkeypatch)
    db = FakeDB()
    _, _, prompt_calls = _turn(
        db, monkeypatch, "He finally. Took part in the discussion",
        repaired="He finally took part in the discussion.",
    )
    messages = prompt_calls["messages"][-1]
    joined = json.dumps(messages, ensure_ascii=False)
    assert "He finally took part in the discussion." in joined


def test_turn_persists_raw_and_repaired(monkeypatch):
    """R17：user_input 恒为原文、不被覆盖；repaired_input 另存。"""
    _enable(monkeypatch)
    db = FakeDB()
    raw = "He finally. Took part in the discussion"
    _turn(db, monkeypatch, raw, repaired="He finally took part in the discussion.")
    doc = db.all(config.EXTENSION_ROUND_COLLECTION)[-1]
    turn = doc["turns"][0]
    assert turn["user_input"] == raw
    assert turn["repaired_input"] == "He finally took part in the discussion."
    assert turn["asr_repair"]["source"] == ag.SOURCE_LLM


def test_turn_returns_repair_fields_when_enabled(monkeypatch):
    _enable(monkeypatch)
    db = FakeDB()
    result, _, _ = _turn(
        db, monkeypatch, "He finally. Took part in the discussion",
        repaired="He finally took part in the discussion.",
    )
    assert result["repaired_input"] == "He finally took part in the discussion."
    assert result["asr_repair"]["applied"] is True


def test_turn_omits_repair_fields_when_disabled(monkeypatch):
    """R18：关开关 → 返回体**不出现**两个新键（老前端零感知）。"""
    _disable(monkeypatch)
    db = FakeDB()
    result, judge_calls, _ = _turn(db, monkeypatch, "He finally. Took part in the discussion")
    assert "repaired_input" not in result
    assert "asr_repair" not in result
    # 判分用原文
    assert judge_calls["answers"][0] == "He finally. Took part in the discussion"


def test_turn_survives_repair_failure(monkeypatch):
    """R16：修复失败不阻断判分，turn_index 正常。"""
    _enable(monkeypatch)
    db = FakeDB()
    result, judge_calls, _ = _turn(
        db, monkeypatch, "He finally. Took part in the discussion",
        error=ExtensionError("LLM_TIMEOUT", "llm", "t"),
    )
    assert result["result"]["score"] == _RUBRIC_PASS["total"]
    assert judge_calls["count"] == 1
    assert result["asr_repair"]["fallback"] is True


def test_turn_no_mastery_writes(monkeypatch):
    """R10 继承：修复接入后仍只写 extension_round。"""
    _enable(monkeypatch)
    db = FakeDB()
    _turn(db, monkeypatch, "He finally. Took part in the discussion",
          repaired="He finally took part in the discussion.")
    written = {coll for _, coll in db.write_log}
    assert written <= {config.EXTENSION_ROUND_COLLECTION}, written


# 本文件未直接用但保留引用，避免 lint 误报未使用导入
assert ORIGINAL and TRANSLATION and POINTS and SENTENCE
