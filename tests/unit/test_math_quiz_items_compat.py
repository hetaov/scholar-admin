"""单元测试：出题 LLM 结果取 `items` 的形态兼容（`_extract_quiz_items`）

对应故障（2026-10-14 真机）：`POST /math/quiz/generate` 返回 HTTP 500
「全部源题出题失败」，后端日志显示 LLM `chat/completions` 全部 **200 OK**，
但每题都走到「出题返回空 items」→ 两次重试均空 → 全部源题失败。

根因：prompt 第 3 条只写「输出 `[{...}]`」（**裸数组**），而解析侧只认
`{"items": [...]}` —— 模型越「听话」越失败，且与题量/并发无关（2 道源题也全挂）。

修法（本文件覆盖）：
- ① prompt 明确要求 `{"items": [...]}`（与 `practice_sheet._QUESTION_USER_TEMPLATE`
     的「输出 JSON 结构」段同口径）→ 见 `test_prompt_requires_items_object`；
- ② 解析侧兜底兼容裸数组 / `{"questions": [...]}` → 见 `_extract_quiz_items` 相关用例。

红线：**R3** 本层不做公开出参校验，只保证「拿得到 items」。
"""
from __future__ import annotations

import asyncio
import json

import pytest

import services.math.quiz as q

_ITEM = {
    "question": "1.25 + 3.4 = ?",
    "answer": "4.65",
    "answer_type": "number",
    "difficulty": 2,
    "hint": "小数点对齐，低位补零",
}


def _run(coro):
    return asyncio.run(coro)


class _FakeClient:
    """占位 client：`_call_quiz_chat_sync` 被打桩后不会被真正调用。"""


def _patch_llm(monkeypatch, raw: str, calls: list | None = None) -> None:
    """把出题 LLM 同步调用打桩为返回 `raw` 文本（可选记录每次调用参数）。"""
    monkeypatch.setattr(q, "LLM_SUMMARY_MODEL", "ep-test")
    monkeypatch.setattr(q, "_get_llm_client", lambda: _FakeClient())

    def _fake_sync(client, model, system, prompt):
        if calls is not None:
            calls.append({"model": model, "system": system, "prompt": prompt})
        return raw

    monkeypatch.setattr(q, "_call_quiz_chat_sync", _fake_sync)


class TestExtractQuizItems:
    """纯函数层：三种形态的取值口径。"""

    def test_items_object_is_contract_shape(self):
        assert q._extract_quiz_items({"items": [_ITEM]}) == [_ITEM]

    def test_top_level_array_is_compatible(self):
        # 旧 prompt / 模型自由发挥：直接输出数组 → 兜底取用，不再判空
        assert q._extract_quiz_items([_ITEM]) == [_ITEM]

    def test_questions_key_is_compatible(self):
        assert q._extract_quiz_items({"questions": [_ITEM]}) == [_ITEM]

    def test_unknown_shape_returns_none(self):
        # 取不到 → None（调用方按「空 items」处理并重试，不让脏数据静默通过）
        assert q._extract_quiz_items({"foo": "bar"}) is None
        assert q._extract_quiz_items({"items": "not-a-list"}) is None
        assert q._extract_quiz_items(None) is None


class TestGenerateQuizQuestionsCompat:
    """打桩 LLM，验证端到端不再「200 但整页失败」。"""

    def test_bare_array_response_succeeds(self, monkeypatch):
        calls: list = []
        _patch_llm(monkeypatch, json.dumps([_ITEM], ensure_ascii=False), calls)

        items = _run(
            q._generate_quiz_questions(source_question="0.36 + 2.7 = ?", count=1)
        )

        assert len(items) == 1
        assert items[0]["question"] == _ITEM["question"]
        assert len(calls) == 1  # 一次命中，未触发重试

    def test_questions_key_response_succeeds(self, monkeypatch):
        _patch_llm(
            monkeypatch,
            json.dumps({"questions": [_ITEM]}, ensure_ascii=False),
        )

        items = _run(q._generate_quiz_questions(source_question="0.36 + 2.7 = ?", count=1))

        assert [it["question"] for it in items] == [_ITEM["question"]]

    def test_empty_items_still_fails_after_retry(self, monkeypatch):
        calls: list = []
        _patch_llm(monkeypatch, json.dumps({"items": []}), calls)

        with pytest.raises(q.LLMResponseError) as exc:
            _run(q._generate_quiz_questions(source_question="0.36 + 2.7 = ?", count=2))

        assert "出题结果 items 为空" in str(exc.value)
        assert len(calls) == 2  # 首次 + 重试 1 次

    def test_truncated_to_count(self, monkeypatch):
        _patch_llm(monkeypatch, json.dumps([_ITEM, dict(_ITEM)], ensure_ascii=False))

        items = _run(q._generate_quiz_questions(source_question="0.36 + 2.7 = ?", count=1))

        assert len(items) == 1  # items[:count]


class TestPromptRequiresItemsObject:
    """① 四套 prompt 模板都必须明确要求 `{"items": [...]}`（杜绝再出裸数组）。"""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},  # 无知识点
            {"knowledge_point_name": "小数加减法"},  # 有知识点
            {"extended": True},  # 拔高·无知识点
            {"extended": True, "knowledge_point_name": "小数加减法"},  # 拔高·有知识点
        ],
    )
    def test_prompt_specifies_items_object(self, kwargs):
        prompt = q._build_quiz_prompt(source_question="0.36 + 2.7 = ?", count=2, **kwargs)

        assert '"items"' in prompt
        assert "不要直接输出顶层数组" in prompt
        # 与 practice_sheet 同口径：给出完整结构示例，而非只描述单元素
        assert '"question"' in prompt and '"answer"' in prompt
