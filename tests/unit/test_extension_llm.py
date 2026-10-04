"""单元测试：英文语句扩展 LLM 客户端 services/providers/extension_llm.py

覆盖：
- call_extension_llm：fake provider 返回字符串 → 原样返回
- provider 返回 None → ExtensionError(PROVIDER_UNAVAILABLE)
- extract_json：code block 包裹 / 前后空白剥离 / 垃圾返回 None
- ExtensionError：错误码 + failure_stage + detail
"""
from __future__ import annotations

import asyncio

import pytest

from services.providers.extension_llm import (
    EXTENSION_SYSTEM_PROMPT,
    ExtensionError,
    build_extension_messages,
    call_extension_llm,
    extract_json,
)


def _run(coro):
    return asyncio.run(coro)


def test_call_extension_llm_returns_content(monkeypatch):
    monkeypatch.setattr(
        "services.providers.extension_llm._call_extension_llm",
        lambda *a, **k: '{"ok": true}',
    )
    result = _run(call_extension_llm(messages=[]))
    assert result == '{"ok": true}'


def test_call_extension_llm_provider_none_raises(monkeypatch):
    monkeypatch.setattr(
        "services.providers.extension_llm._call_extension_llm",
        lambda *a, **k: None,
    )
    with pytest.raises(ExtensionError) as exc:
        _run(call_extension_llm(messages=[]))
    assert exc.value.error_code == "PROVIDER_UNAVAILABLE"
    assert exc.value.failure_stage == "llm"


def test_extract_json_strips_code_block():
    raw = '```json\n{"a": 1}\n```'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_handles_whitespace():
    raw = '   {"a": 1}   '
    assert extract_json(raw) == {"a": 1}


def test_extract_json_fallback_returns_none_for_garbage():
    assert extract_json("hello world") is None


def test_extension_error_attributes():
    e = ExtensionError("LLM_TIMEOUT", "llm", "超时")
    assert e.error_code == "LLM_TIMEOUT"
    assert e.failure_stage == "llm"
    assert str(e) == "超时"


# ---------------------------------------------------------------------------
# Prompt v2（分层抽取：LLM 只对候选清单判定，只回 candidate_id）
# ---------------------------------------------------------------------------

_CANDS = [
    {
        "candidate_id": "c1",
        "text": "take part in",
        "span": [7, 19],
        "hint_type": "phrase",
        "source": "lexicon",
    }
]


def test_system_prompt_v2_forbids_new_candidates_and_requires_candidate_id():
    assert "candidate_id" in EXTENSION_SYSTEM_PROMPT
    assert "不得新增候选" in EXTENSION_SYSTEM_PROMPT
    # 不再要求模型自造 span
    assert "不要输出 `text` / `span` 字段" in EXTENSION_SYSTEM_PROMPT
    # §4 慎用表达风险标记
    assert "risk" in EXTENSION_SYSTEM_PROMPT


def test_build_extension_messages_includes_numbered_candidate_list():
    msgs = build_extension_messages("I will take part in the discussion.", "我会参加讨论。", _CANDS)
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert "I will take part in the discussion." in user
    assert "我会参加讨论。" in user
    assert "c1" in user and "take part in" in user and "span=7:19" in user


def test_build_extension_messages_includes_abbreviation_hint():
    msgs = build_extension_messages("I'm gonna go.", "", [])
    assert "gonna = going to" in msgs[1]["content"]


def test_build_extension_messages_handles_empty_candidates():
    msgs = build_extension_messages("Hello there friend.", "")
    assert "候选清单为空" in msgs[1]["content"]
