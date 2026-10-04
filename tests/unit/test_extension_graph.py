"""单元测试：非连续短语 LangGraph 编排 services/english/extension_graph.py

覆盖：
- discover_structures：LLM 输出结构校验（≥2 区间才视为非连续）
- validate_spans：R7 多区间坐标校验（去空白比对）
- enrich_points：structure_id 映射
- assemble：text/spans 与释义合并，discontinuous 标记
- 端到端：正常产出 / 无非连续结构 / discover 失败降级
"""
from __future__ import annotations

import pytest

from services.english import extension_graph as eg
from services.providers.extension_llm import ExtensionError

ORIGINAL = "Please turn the light off before leaving."
# "turn" = [7,11], "off" = [22,25]


# ---------------------------------------------------------------------------
# discover 节点：结构校验
# ---------------------------------------------------------------------------


def test_validate_raw_structures_accepts_valid_multi_span():
    raw = [
        {"text": "turn off", "spans": [[7, 11], [22, 25]], "type": "phrase"},
    ]
    out = eg._validate_raw_structures(raw, ORIGINAL)
    assert len(out) == 1
    assert out[0]["text"] == "turn off"
    assert out[0]["spans"] == [[7, 11], [22, 25]]


def test_validate_raw_structures_rejects_single_span():
    """非连续必须 ≥2 区间；单区间由候选层处理。"""
    raw = [{"text": "turn", "spans": [[7, 11]], "type": "word"}]
    out = eg._validate_raw_structures(raw, ORIGINAL)
    assert out == []


def test_validate_raw_structures_rejects_bad_spans():
    raw = [
        {"text": "turn off", "spans": "bad"},
        {"text": "x", "spans": [[7, 11]]},  # 单区间
        {"text": "y", "spans": [[7, 11], "bad"]},
    ]
    assert eg._validate_raw_structures(raw, ORIGINAL) == []


def test_validate_raw_structures_normalizes_type():
    raw = [{"text": "turn off", "spans": [[7, 11], [22, 25]], "type": "weird"}]
    out = eg._validate_raw_structures(raw, ORIGINAL)
    assert out[0]["type"] == "phrase"


# ---------------------------------------------------------------------------
# validate 节点：R7 多区间校验
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_spans_accepts_correct_coordinates():
    state = {"original": ORIGINAL, "raw_structures": [
        {"text": "turn off", "spans": [[7, 11], [22, 25]], "type": "phrase"},
    ]}
    out = await eg.validate_spans(state)
    assert len(out["validated_structures"]) == 1
    assert out["discarded_count"] == 0


@pytest.mark.asyncio
async def test_validate_spans_discards_mismatch():
    state = {"original": ORIGINAL, "raw_structures": [
        {"text": "turn off", "spans": [[7, 11], [21, 24]], "type": "phrase"},  # " of" ≠ "off"
    ]}
    out = await eg.validate_spans(state)
    assert out["validated_structures"] == []
    assert out["discarded_count"] == 1


@pytest.mark.asyncio
async def test_validate_spans_discards_overlapping_intervals():
    state = {"original": ORIGINAL, "raw_structures": [
        {"text": "turn off", "spans": [[7, 11], [9, 12]], "type": "phrase"},
    ]}
    out = await eg.validate_spans(state)
    assert out["discarded_count"] == 1


# ---------------------------------------------------------------------------
# enrich 节点：structure_id 映射
# ---------------------------------------------------------------------------


def test_validate_enriched_items_filters_bad_ids():
    raw = [
        {"structure_id": "n1", "meaning_zh": "关掉"},
        {"structure_id": "n9", "meaning_zh": "x"},  # 未知 id
        {"structure_id": "n1", "meaning_zh": "重复"},  # 重复
    ]
    out = eg._validate_enriched_items(raw, {"n1", "n2"})
    assert len(out) == 1
    assert out[0]["meaning_zh"] == "关掉"


# ---------------------------------------------------------------------------
# assemble 节点：合并
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemble_merges_text_spans_with_enrichment():
    state = {
        "original": ORIGINAL,
        "validated_structures": [
            {"text": "turn off", "spans": [[7, 11], [22, 25]], "type": "phrase"},
        ],
        "enriched_points": [
            {
                "structure_id": "n1",
                "meaning_zh": "关掉",
                "register": "neutral",
                "example_en": "Turn off the light.",
                "example_zh": "关灯。",
                "collocations": ["turn off the light"],
                "reason": "常用短语动词",
                "confidence": 0.9,
            }
        ],
    }
    out = await eg.assemble(state)
    assert out["status"] == "success"
    pts = out["final_points"]
    assert len(pts) == 1
    p = pts[0]
    assert p["text"] == "turn off"
    assert p["spans"] == [[7, 11], [22, 25]]
    assert p["discontinuous"] is True
    assert p["meaning_zh"] == "关掉"
    assert p["span"] == [7, 11]


@pytest.mark.asyncio
async def test_assemble_no_structures_returns_no_structures_status():
    state = {"original": ORIGINAL, "validated_structures": [], "enriched_points": []}
    out = await eg.assemble(state)
    assert out["status"] == "no_structures"
    assert out["final_points"] == []


# ---------------------------------------------------------------------------
# 端到端图执行（mock LLM）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_extension_graph_happy_path(monkeypatch):
    """正常流程：discover 出 turn off → validate 通过 → enrich 释义 → assemble。"""
    discover_resp = '{"structures": [{"text": "turn off", "spans": [[7,11],[22,25]], "type": "phrase", "confidence": 0.9}]}'
    enrich_resp = '{"points": [{"structure_id": "n1", "meaning_zh": "关掉", "register": "neutral", "example_en": "Turn off the light.", "example_zh": "关灯。", "collocations": ["turn off the light"], "reason": "常用", "confidence": 0.9}]}'
    call_count = {"n": 0}

    async def fake_call(messages, **kw):
        call_count["n"] += 1
        # 第 1 次 discover，第 2 次 enrich
        return discover_resp if call_count["n"] == 1 else enrich_resp

    monkeypatch.setattr(eg, "call_extension_llm", fake_call)

    result = await eg.run_extension_graph(original=ORIGINAL, translation="请关灯")
    assert result["status"] == "success"
    assert len(result["points"]) == 1
    assert result["points"][0]["text"] == "turn off"
    assert result["points"][0]["discontinuous"] is True


@pytest.mark.asyncio
async def test_run_extension_graph_no_structures(monkeypatch):
    """无非连续结构 → status=no_structures, points=[]。"""
    async def fake_call(messages, **kw):
        return '{"structures": []}'

    monkeypatch.setattr(eg, "call_extension_llm", fake_call)
    result = await eg.run_extension_graph(original=ORIGINAL)
    assert result["status"] == "no_structures"
    assert result["points"] == []


@pytest.mark.asyncio
async def test_run_extension_graph_discover_failure_degrades(monkeypatch):
    """discover LLM 失败 → status=failed, points=[]（不抛异常，主流水线降级）。"""
    async def fake_call(messages, **kw):
        raise ExtensionError("LLM_TIMEOUT", "llm", "timeout")

    monkeypatch.setattr(eg, "call_extension_llm", fake_call)
    result = await eg.run_extension_graph(original=ORIGINAL)
    assert result["status"] == "failed"
    assert result["points"] == []


@pytest.mark.asyncio
async def test_run_extension_graph_enrich_failure_keeps_bare_points(monkeypatch):
    """enrich 失败 → assemble 用 validated 的 text/spans，释义字段为空（降级而非全丢）。"""
    discover_resp = '{"structures": [{"text": "turn off", "spans": [[7,11],[22,25]], "type": "phrase"}]}'
    call_count = {"n": 0}

    async def fake_call(messages, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return discover_resp
        raise ExtensionError("LLM_TIMEOUT", "llm", "timeout")

    monkeypatch.setattr(eg, "call_extension_llm", fake_call)
    result = await eg.run_extension_graph(original=ORIGINAL)
    # enrich 失败，但 validated 仍在 → assemble 产出 bare point（释义为空）
    assert result["status"] == "success"
    assert len(result["points"]) == 1
    assert result["points"][0]["text"] == "turn off"
    assert result["points"][0]["meaning_zh"] == ""
