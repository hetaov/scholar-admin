"""单元测试：P1 拔高题 `include_extended`（设计文档 §4.3 入参表 · 契约 §3.10 接口 29）

对应任务：docs_v1/数学/同类型题-原子任务清单.md B09 的 P1 分支（B16/B17 之前补齐实现）。

口径（`include_extended`，默认 false）：
- `true` 且该源题节点存在 `ai_summary.extended_points[]` → **每道源题追加 1 道拔高题**
  （`QUIZ_EXTENDED_COUNT_PER_SOURCE`），难度档位（入门 / 普及 / 竞赛）注入 prompt，
  产物打 `is_extended=True`；
- 节点无扩展点 / 未传 `node_code` → 跳过 + `warnings[]`（**不影响基础题**）；
- 拔高题同样过 D6 防背题校验；拔高出题失败只降级写 warning，不拖累基础题；
- `include_extended` **参与幂等签名**（开关不同不命中既有 quiz）。

红线：**R3** 出参 `items[]` 无 `answer` / `hint`；**D4** prompt 不含 `error_type` / 错因。
"""
from __future__ import annotations

import asyncio

import pytest

import services.math.quiz as q
from services.math.a4_renderer import build_quiz_html
from tests.fakes.fake_db import FakeDB

NODE_CODE = "5-3-1-kp2"

NODE = {
    "code": NODE_CODE,
    "title": "小数加减法",
    "ai_summary": {
        "summary": "小数点对齐，低位补零",
        "knowledge_points": [{"name": "小数加减法", "summary": "小数点对齐，低位补零"}],
        "extended_points": [
            {
                "name": "小数凑整",
                "summary": "凑整简算",
                "difficulty_band": "普及",
                "related_knowledge_name": "小数加减法",
            }
        ],
    },
}
NODE_NO_EP = {"code": "5-3-1-kp9", "title": "无扩展点节点", "ai_summary": {"summary": "s"}}

SOURCE = {
    "source_index": 0,
    "question_text": "0.36 + 2.7 = ?",
    "knowledge_point_name": "小数加减法",
    "node_code": NODE_CODE,
}


def _run(coro):
    return asyncio.run(coro)


class _LLMSpy:
    """替换 `_generate_quiz_questions`：记录调用参数，按需返回基础题 / 拔高题。"""

    def __init__(self, extended_ok: bool = True):
        self.extended_ok = extended_ok
        self.calls: list[dict] = []

    async def __call__(
        self,
        *,
        source_question: str,
        knowledge_point_name: str = "",
        summary: str = "",
        count: int = 2,
        extended: bool = False,
        extended_band: str = "",
    ) -> list[dict]:
        self.calls.append(
            {
                "extended": extended,
                "extended_band": extended_band,
                "count": count,
                "kp": knowledge_point_name,
            }
        )
        if extended:
            if not self.extended_ok:
                raise q.LLMResponseError("拔高题 LLM 失败")
            return [
                {
                    "question": "用凑整法计算 9.98 + 3.47 + 0.02，写出简算过程",
                    "answer": "13.47",
                    "answer_type": "text",
                    "difficulty": 4,
                    "hint": "先凑整",
                }
            ]
        return [
            {
                "question": f"长 {i}{i} 分米、宽 {i} 分米的长方形面积是多少？",
                "answer": str(i * 11),
                "answer_type": "number",
                "difficulty": 2,
                "hint": "",
            }
            for i in range(count)
        ]


@pytest.fixture
def llm(monkeypatch):
    def install(extended_ok: bool = True) -> _LLMSpy:
        spy = _LLMSpy(extended_ok=extended_ok)
        monkeypatch.setattr(q, "_generate_quiz_questions", spy)
        monkeypatch.setattr(q, "LLM_SUMMARY_MODEL", "mock-model")
        return spy

    return install


def _db(node: dict | None = None) -> FakeDB:
    db = FakeDB()
    if node is not None:
        db.add("curriculum_node", node)
    return db


class TestIncludeExtended:
    def test_extended_appended_when_node_has_extended_point(self, llm):
        spy = llm()
        db = _db(NODE)

        data = _run(
            q.generateQuiz(
                db,
                scholar_id="s1",
                source_questions=[dict(SOURCE)],
                count_per_source=2,
                include_extended=True,
                actor="p1",
            )
        )

        # 2 基础 + 1 拔高；拔高题排在对应源题产出之后，且带标记
        assert len(data["items"]) == 3
        assert [it["is_extended"] for it in data["items"]] == [False, False, True]
        assert data["items"][-1]["difficulty"] == 4
        # 拔高调用带上了扩展点难度档位
        assert spy.calls[-1] == {
            "extended": True,
            "extended_band": "普及",
            "count": q.QUIZ_EXTENDED_COUNT_PER_SOURCE,
            "kp": "小数加减法",
        }
        assert data["warnings"] == []
        # R3：出参无 answer / hint
        assert all("answer" not in it and "hint" not in it for it in data["items"])
        # 拔高题同样回指源题下标
        assert {it["source_index"] for it in data["items"]} == {0}

    def test_no_extended_point_skipped_with_warning(self, llm):
        spy = llm()
        db = _db(NODE_NO_EP)

        data = _run(
            q.generateQuiz(
                db,
                scholar_id="s1",
                source_questions=[
                    {**SOURCE, "node_code": NODE_NO_EP["code"], "knowledge_point_name": "加法"}
                ],
                count_per_source=1,
                include_extended=True,
                actor="p1",
            )
        )

        assert len(data["items"]) == 1
        assert all(c["extended"] is False for c in spy.calls)
        assert data["warnings"] == ["第 1 道原题所属节点无奥数扩展点，已跳过拔高题"]

    def test_missing_node_code_skipped_with_warning(self, llm):
        spy = llm()
        db = _db(NODE)

        data = _run(
            q.generateQuiz(
                db,
                scholar_id="s1",
                source_questions=[{**SOURCE, "node_code": ""}],
                count_per_source=1,
                include_extended=True,
                actor="p1",
            )
        )

        assert len(data["items"]) == 1
        assert data["warnings"][-1] == "第 1 道原题未提供 node_code，无法出拔高题，已跳过"

    def test_extended_failure_only_degrades(self, llm):
        spy = llm(extended_ok=False)
        db = _db(NODE)

        data = _run(
            q.generateQuiz(
                db,
                scholar_id="s1",
                source_questions=[dict(SOURCE)],
                count_per_source=2,
                include_extended=True,
                actor="p1",
            )
        )

        # 基础题照常产出，拔高题失败只写 warning（不整体失败）
        assert len(data["items"]) == 2
        assert data["warnings"] == ["第 1 道原题拔高题出题失败，已跳过"]

    def test_flag_off_has_zero_extended_call(self, llm):
        spy = llm()
        db = _db(NODE)

        data = _run(
            q.generateQuiz(
                db,
                scholar_id="s1",
                source_questions=[dict(SOURCE)],
                count_per_source=1,
                include_extended=False,  # 默认口径
                actor="p1",
            )
        )

        assert all(c["extended"] is False for c in spy.calls)
        assert all(it["is_extended"] is False for it in data["items"])

    def test_include_extended_participates_in_idempotent_signature(self, llm):
        llm()
        db = _db(NODE)
        kwargs = dict(
            scholar_id="s1",
            source_questions=[dict(SOURCE)],
            count_per_source=1,
            actor="p1",
        )
        first = _run(q.generateQuiz(db, **kwargs, include_extended=True))
        second = _run(q.generateQuiz(db, **kwargs, include_extended=True))
        third = _run(q.generateQuiz(db, **kwargs, include_extended=False))

        assert second["quiz_id"] == first["quiz_id"]  # 同开关 → 幂等命中
        assert third["quiz_id"] != first["quiz_id"]  # 开关不同 → 不命中


class TestExtendedPrompt:
    def test_prompt_has_band_and_no_error_type(self):
        prompt = q._build_quiz_prompt(
            source_question="0.36 + 2.7 = ?",
            knowledge_point_name="小数加减法",
            summary="小数点对齐",
            count=1,
            extended=True,
            extended_band="普及",
        )
        assert "奥数拔高题" in prompt and "普及" in prompt
        assert "error_type" not in prompt and "错因" not in prompt

    def test_prompt_no_kp_variant(self):
        prompt = q._build_quiz_prompt(
            source_question="0.36 + 2.7 = ?",
            count=1,
            extended=True,
            extended_band="",  # 缺省 → 入门
        )
        assert "所属知识点" not in prompt
        assert "入门" in prompt

    def test_html_marks_extended_item(self):
        html = build_quiz_html(
            {
                "quiz_id": "qz_1",
                "scholar_id": "s1",
                "created_at": 1760000000000,
                "items": [
                    {"source_index": 0, "question": "基础题", "answer": "1"},
                    {
                        "source_index": 0,
                        "question": "拔高题",
                        "answer": "2",
                        "is_extended": True,
                    },
                ],
            }
        )
        assert html.count("来自原题 1") == 2
        assert html.count("奥数拔高") == 1
