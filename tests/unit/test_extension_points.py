"""单元测试：英文语句扩展抽取核心 services/english/extension.py

覆盖：
- extension_idempotency_key 稳定性（sha256，64 位 hex）
- validate_and_normalize_points：坏字段丢弃策略、span 一致性（R7）、slang 强制 informal
- apply_quota_and_overlap：配额截断、重叠优先级 idiom>slang>phrase>word
- mark_cross_sentence_excluding：跨句去重标 excluding 不删除
- rule_fallback_points：只产 word、停用词过滤
- build_l1_items：题面 + 4 选项（候选池补干扰项 / 确定性打乱 / 无占位符 / 缺原句告警）、不含答案键
"""
from __future__ import annotations

import asyncio
import logging

import config
from services.english.extension import (
    L1_OPTIONS_MAX,
    POINT_TYPES,
    QUOTA,
    TOTAL_QUOTA,
    apply_quota_and_overlap,
    audit_points,
    build_l1_items,
    extension_idempotency_key,
    grade_l1_fill,
    map_candidate_points,
    mark_cross_sentence_excluding,
    rank_points,
    run_evaluate_pipeline,
    rule_fallback_points,
    validate_and_normalize_points,
)
from services.english.extension_review import get_review, save_review
from services.models.content import compute_text_hash
from tests.fakes.fake_db import FakeDB

ORIGINAL = "I will take part in the discussion."


def _run(coro):
    return asyncio.run(coro)


def _make_point(text, span, ptype="word", **kw):
    base = {
        "type": ptype,
        "text": text,
        "span": span,
        "meaning_zh": "释义",
        "register": "neutral",
        "example_en": f"an example with {text}",
        "example_zh": "",
        "reason": "理由",
        "literal_zh": "",
        "pos": "",
        "collocations": [],
        "synonyms": [],
        "confusable": [],
        "cefr": "",
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# 幂等键
# ---------------------------------------------------------------------------


def test_idempotency_key_is_sha256_64hex_and_stable():
    h = compute_text_hash(ORIGINAL)
    k1 = extension_idempotency_key(
        sentence_id="s1", content_hash=h, prompt_version="v1", model="m"
    )
    k2 = extension_idempotency_key(
        sentence_id="s1", content_hash=h, prompt_version="v1", model="m"
    )
    assert len(k1) == 64
    assert all(c in "0123456789abcdef" for c in k1)
    assert k1 == k2  # 稳定


def test_idempotency_key_changes_with_inputs():
    h = compute_text_hash(ORIGINAL)
    base = dict(sentence_id="s1", content_hash=h, prompt_version="v1", model="m")
    k0 = extension_idempotency_key(**base)
    assert k0 != extension_idempotency_key(**{**base, "sentence_id": "s2"})
    assert k0 != extension_idempotency_key(**{**base, "prompt_version": "v2"})
    assert k0 != extension_idempotency_key(**{**base, "model": "other"})


# ---------------------------------------------------------------------------
# Schema + span 校验（R7）
# ---------------------------------------------------------------------------


def test_validate_span_mismatch_discards_point():
    """R7：span 切出字符 ≠ text → 丢该条（不整句失败）。"""
    raw = [
        _make_point("take part in", [7, 19], "phrase", collocations=["x"]),
        _make_point("WRONG", [7, 19], "word"),  # span 切出的是 "take part" ≠ WRONG
    ]
    valid, discarded = validate_and_normalize_points(raw, ORIGINAL)
    assert discarded == 1
    assert len(valid) == 1
    assert valid[0]["text"] == "take part in"


# ---------------------------------------------------------------------------
# 多 span（非连续短语）
# ---------------------------------------------------------------------------

MULTI_ORIGINAL = "Please turn the light off before leaving."


def test_validate_multi_span_r7_passes():
    """R7 多区间：turn the light off 中 turn...off 拼接 == text。"""
    raw = [
        _make_point(
            "turn off",
            None,
            "phrase",
            spans=[[7, 11], [22, 25]],
            collocations=["turn off the light"],
        )
    ]
    valid, discarded = validate_and_normalize_points(raw, MULTI_ORIGINAL)
    assert discarded == 0
    assert len(valid) == 1
    assert valid[0]["spans"] == [[7, 11], [22, 25]]
    assert valid[0]["discontinuous"] is True
    assert valid[0]["span"] == [7, 11]  # 兼容首区间


def test_validate_multi_span_r7_mismatch_discards():
    """R7 多区间：拼接 ≠ text → 丢弃。"""
    raw = [
        _make_point(
            "turn off",
            None,
            "phrase",
            spans=[[7, 11], [21, 24]],  # [21,24] 切出 " of"，拼接 "turn of" ≠ "turn off"
            collocations=["turn off the light"],
        )
    ]
    valid, discarded = validate_and_normalize_points(raw, MULTI_ORIGINAL)
    assert discarded == 1
    assert valid == []


def test_validate_multi_span_overlapping_intervals_discards():
    """多区间必须升序不重叠。"""
    raw = [
        _make_point(
            "turn off",
            None,
            "phrase",
            spans=[[7, 11], [9, 12]],  # 重叠
            collocations=["turn off the light"],
        )
    ]
    valid, discarded = validate_and_normalize_points(raw, MULTI_ORIGINAL)
    assert discarded == 1


def test_l1_multi_span_blanks_all_intervals():
    """L1 多区间挖空：turn the light off → ____ the light ____。"""
    p = _make_point(
        "turn off", None, "phrase",
        spans=[[7, 11], [22, 25]],
        collocations=["turn off the light"],
    )
    build_l1_items(
        [p],
        original=MULTI_ORIGINAL,
        candidates=[
            {"candidate_id": "c1", "text": "turn", "span": [7, 11], "hint_type": "word"},
            {"candidate_id": "c2", "text": "light", "span": [16, 21], "hint_type": "word"},
            {"candidate_id": "c3", "text": "before", "span": [26, 32], "hint_type": "word"},
        ],
    )
    assert p["l1"] is not None
    assert p["l1"]["stem"] == "Please ____ the light ____ before leaving."
    # 干扰项不得与正解互为子串（turn / off 都包含在 "turn off" 里）
    assert "light" in p["l1"]["options"]
    assert "turn" not in p["l1"]["options"]


def test_multi_span_overlap_with_single_span():
    """多 span 点与单 span 点任一区间相交即重叠。"""
    # phrase "turn off" spans [[7,11],[22,25]]；word "light" span [16,21) 与 [22,25] 不重叠
    multi = _make_point("turn off", None, "phrase", spans=[[7, 11], [22, 25]], collocations=["x"])
    word_light = _make_point("light", [16, 21], "word")
    result = apply_quota_and_overlap([multi, word_light])
    # phrase 优先级高于 word，且不重叠 → 都保留
    texts = {p["text"] for p in result}
    assert "turn off" in texts
    assert "light" in texts


def test_multi_span_overlap_removes_lower_priority():
    """多 span 点区间与低优先级点重叠 → 保留高优先级。"""
    multi = _make_point("turn off", None, "phrase", spans=[[7, 11], [22, 25]], collocations=["x"])
    word_turn = _make_point("turn", [7, 11], "word")  # 与 [7,11] 重叠
    result = apply_quota_and_overlap([multi, word_turn])
    texts = {p["text"] for p in result}
    assert "turn off" in texts
    assert "turn" not in texts  # word 被 phrase 重叠剔除


def test_validate_bad_type_discarded():
    raw = [_make_point("take", [7, 11], "not_a_type")]
    valid, discarded = validate_and_normalize_points(raw, ORIGINAL)
    assert discarded == 1
    assert valid == []


def test_validate_slang_forces_informal_and_requires_literal_zh():
    # slang 无 literal_zh → 丢弃
    raw = [_make_point("cool", [0, 4], "slang")]
    valid, _ = validate_and_normalize_points(raw, ORIGINAL.replace("I will", "cool"))
    assert valid == []
    # slang 有 literal_zh → register 强制 informal
    raw2 = [_make_point("cool", [0, 4], "slang", literal_zh="酷", register="formal")]
    valid2, _ = validate_and_normalize_points(raw2, "cool beans")
    assert len(valid2) == 1
    assert valid2[0]["register"] == "informal"


def test_validate_phrase_requires_collocations():
    raw = [_make_point("take part in", [7, 19], "phrase", collocations=[])]
    valid, discarded = validate_and_normalize_points(raw, ORIGINAL)
    assert discarded == 1


# ---------------------------------------------------------------------------
# 配额 + 重叠
# ---------------------------------------------------------------------------


def test_quota_truncates_by_type():
    points = []
    for i in range(6):
        points.append(_make_point(f"word{i}", [i * 2, i * 2 + 3], "word"))
    result = apply_quota_and_overlap(points)
    assert len(result) <= TOTAL_QUOTA
    types = [p["type"] for p in result]
    assert types.count("word") <= QUOTA["word"]


def test_overlap_priority_idiom_over_word():
    idiom = _make_point("take part in", [7, 19], "idiom")
    word = _make_point("part", [12, 16], "word")
    result = apply_quota_and_overlap([word, idiom])
    assert len(result) == 1
    assert result[0]["type"] == "idiom"


# ---------------------------------------------------------------------------
# 跨句去重
# ---------------------------------------------------------------------------


def test_cross_sentence_excluding_marks_not_deletes():
    p1 = _make_point("take part in", [7, 19], "phrase", collocations=["x"])
    p2 = _make_point("take part in", [0, 12], "phrase", collocations=["x"])
    seen = set()
    mark_cross_sentence_excluding([p1], seen)
    mark_cross_sentence_excluding([p2], seen)
    assert p1["excluding"] is False
    assert p2["excluding"] is True  # 重复标 excluding，不删除


# ---------------------------------------------------------------------------
# 规则兜底
# ---------------------------------------------------------------------------


def test_rule_fallback_only_word_and_filters_stopwords():
    pts = rule_fallback_points("I will take part in the discussion.")
    assert all(p["type"] == "word" for p in pts)
    texts = [p["text"].lower() for p in pts]
    assert "i" not in texts
    assert "the" not in texts
    assert "will" not in texts


# ---------------------------------------------------------------------------
# L1 题面
# ---------------------------------------------------------------------------


def _candidates_for_original():
    """同句候选池（含一条已入选的 discussion，应被排除在干扰项之外）。"""
    return [
        {"candidate_id": "c1", "text": "take part in", "span": [7, 19], "hint_type": "phrase"},
        {"candidate_id": "c2", "text": "will", "span": [2, 6], "hint_type": "word"},
        {"candidate_id": "c3", "text": "discussion", "span": [24, 34], "hint_type": "word"},
        {"candidate_id": "c4", "text": "take", "span": [7, 11], "hint_type": "word"},
        {"candidate_id": "c5", "text": "part", "span": [12, 16], "hint_type": "word"},
    ]


def test_build_l1_items_generates_stem_and_options():
    p = _make_point("discussion", [24, 34], "word")
    build_l1_items([p], original=ORIGINAL, candidates=_candidates_for_original())
    assert p["l1"] is not None
    assert "____" in p["l1"]["stem"]
    assert len(p["l1"]["options"]) == L1_OPTIONS_MAX
    assert "discussion" in p["l1"]["options"]
    # C：不再用（选项N）占位符补齐
    assert "（选项" not in "".join(p["l1"]["options"])
    # 不含答案键（stem 是挖空的，选项里有答案但无单独 answer 字段）
    assert "answer" not in p["l1"]


def test_build_l1_items_original_fallback_still_works():
    """兼容路径：未显式传 original 时仍回退 p["_original"]（已废弃但不可静默断）。"""
    p = _make_point("discussion", [24, 34], "word")
    p["_original"] = ORIGINAL
    build_l1_items([p], candidates=_candidates_for_original())
    assert p["l1"] is not None
    assert "____" in p["l1"]["stem"]


def test_build_l1_items_missing_original_logs_warning(caplog):
    """原句缺失不再静默丢题：必须打 warning 且 l1=None。"""
    p = _make_point("discussion", [24, 34], "word")
    with caplog.at_level(logging.WARNING, logger="scholar-admin.extension"):
        build_l1_items([p], candidates=_candidates_for_original())
    assert p["l1"] is None
    assert "缺少原句" in caplog.text


def test_build_l1_items_no_distractor_returns_none(caplog):
    """零干扰项（无同伴点、无 confusable、无候选池）→ 不出题。"""
    p = _make_point("discussion", [24, 34], "word")
    with caplog.at_level(logging.WARNING, logger="scholar-admin.extension"):
        build_l1_items([p], original=ORIGINAL)
    assert p["l1"] is None
    assert "干扰项不足" in caplog.text


def test_build_l1_items_options_incomplete_flagged():
    """真实干扰项不足 3 个 → 按实际数量返回并置 options_incomplete。"""
    p = _make_point("discussion", [24, 34], "word", confusable=["meeting"])
    build_l1_items([p], original=ORIGINAL)
    assert p["l1"] is not None
    assert len(p["l1"]["options"]) == 2  # 正解 + 1 干扰项
    assert p["l1"]["options_incomplete"] is True
    assert "（选项" not in "".join(p["l1"]["options"])


def test_build_l1_items_shuffle_is_deterministic_and_gradeable():
    """B：选项打乱可复现（同 id 同结果），且不影响判分。"""
    def _build():
        p = _make_point("discussion", [24, 34], "word")
        p["id"] = "s1#0"
        build_l1_items([p], original=ORIGINAL, candidates=_candidates_for_original())
        return p

    for _ in range(10):
        p = _build()
        assert p["l1"]["options"] == _build()["l1"]["options"]
        # 打乱后仍可判分（判分只比 text）
        assert grade_l1_fill(p, "discussion")["passed"] is True
        assert grade_l1_fill(p, "will")["passed"] is False


def test_build_l1_items_excludes_substring_distractors():
    """与正解互为子串的候选（take part ↔ take part in）不得作为干扰项。"""
    p = _make_point("take part in", [7, 19], "phrase", collocations=["take part in the meeting"])
    build_l1_items(
        [p],
        original=ORIGINAL,
        candidates=[
            {"candidate_id": "c1", "text": "take part", "span": [7, 16], "hint_type": "phrase"},
            {"candidate_id": "c2", "text": "part in", "span": [12, 19], "hint_type": "phrase"},
            {"candidate_id": "c3", "text": "discussion", "span": [24, 34], "hint_type": "word"},
        ],
    )
    opts = p["l1"]["options"]
    assert "take part" not in opts
    assert "part in" not in opts
    assert "take part in" in opts


# ---------------------------------------------------------------------------
# 新字段：confidence / risk（分层抽取新增，仅 2 个可选字段）
# ---------------------------------------------------------------------------

PHRASE_SPAN = [7, 19]  # "take part in"


def _make_candidate(cid, text, span):
    return {
        "candidate_id": cid,
        "text": text,
        "span": list(span),
        "hint_type": "phrase",
        "source": "lexicon",
    }


def _llm_item(cid, **kw):
    base = {
        "candidate_id": cid,
        "type": "phrase",
        "meaning_zh": "参加",
        "register": "neutral",
        "example_en": "an example with take part in",
        "reason": "固定搭配",
        "collocations": ["take part in the meeting"],
        "confidence": 0.9,
    }
    base.update(kw)
    return base


def test_confidence_clamped_and_risk_validated():
    hi = _make_point(
        "take part in", PHRASE_SPAN, "phrase", collocations=["x"], confidence=1.8, risk="vulgar"
    )
    valid, _ = validate_and_normalize_points([hi], ORIGINAL)
    assert valid[0]["confidence"] == 1.0
    assert valid[0]["risk"] == "vulgar"

    bad = _make_point(
        "take part in", PHRASE_SPAN, "phrase", collocations=["x"], confidence="oops", risk="bogus"
    )
    v2, _ = validate_and_normalize_points([bad], ORIGINAL)
    assert v2[0]["confidence"] == 0.0
    assert v2[0]["risk"] == ""


def test_confidence_and_risk_default_when_absent():
    p = _make_point("take part in", PHRASE_SPAN, "phrase", collocations=["x"])
    valid, _ = validate_and_normalize_points([p], ORIGINAL)
    assert valid[0]["confidence"] == 0.0
    assert valid[0]["risk"] == ""


# ---------------------------------------------------------------------------
# 候选口径映射（map_candidate_points）
# ---------------------------------------------------------------------------


def test_map_candidate_points_uses_candidate_span_and_drops_bad_ids():
    cands = [_make_candidate("c1", "take part in", PHRASE_SPAN)]
    raw = [
        _llm_item("c1"),  # ok
        _llm_item("c9"),  # 未知 candidate_id
        _llm_item("c1"),  # 重复
        _llm_item("c1", meaning_zh=""),  # 字段非法
    ]
    valid, discarded = map_candidate_points(raw, cands, ORIGINAL)
    assert len(valid) == 1
    assert valid[0]["text"] == "take part in"
    assert valid[0]["span"] == PHRASE_SPAN
    assert discarded == 3


def test_map_candidate_points_ignores_model_supplied_text_and_span():
    cands = [_make_candidate("c1", "take part in", PHRASE_SPAN)]
    raw = [_llm_item("c1", text="WRONG", span=[0, 5])]
    valid, _ = map_candidate_points(raw, cands, ORIGINAL)
    assert valid[0]["text"] == "take part in"
    assert valid[0]["span"] == PHRASE_SPAN


def test_map_candidate_points_all_invalid_returns_empty():
    cands = [_make_candidate("c1", "take part in", PHRASE_SPAN)]
    valid, discarded = map_candidate_points([_llm_item("c9")], cands, ORIGINAL)
    assert valid == []
    assert discarded == 1


def test_map_candidate_points_empty_raw_returns_empty():
    cands = [_make_candidate("c1", "take part in", PHRASE_SPAN)]
    valid, discarded = map_candidate_points([], cands, ORIGINAL)
    assert valid == [] and discarded == 0


# ---------------------------------------------------------------------------
# 敏感词与质量审核（audit_points）：标 risk，不删除
# ---------------------------------------------------------------------------


def test_audit_points_marks_risk_without_dropping():
    pts = audit_points([_make_point("damn", [0, 4], "slang", literal_zh="该死")])
    assert len(pts) == 1
    assert pts[0]["risk"] == "offensive"


def test_audit_points_keeps_existing_risk():
    p = _make_point("damn", [0, 4], "slang", literal_zh="该死", risk="regional")
    audit_points([p])
    assert p["risk"] == "regional"


def test_audit_points_leaves_plain_points_untouched():
    # 流水线中 points 均来自 _validate_point，risk 字段必存在
    pts = audit_points([_make_point("discussion", [24, 34], "word", risk="")])
    assert pts[0]["risk"] == ""


# ---------------------------------------------------------------------------
# 按学习价值排序（rank_points）：仅展示序
# ---------------------------------------------------------------------------


def test_rank_points_orders_by_type_priority_then_confidence():
    pts = [
        _make_point("a", [0, 1], "word", confidence=0.99),
        _make_point("b", [2, 3], "idiom", confidence=0.1),
        _make_point("c", [4, 5], "phrase", collocations=["x"], confidence=0.5),
        _make_point("d", [6, 7], "word", confidence=0.2),
    ]
    ranked = rank_points(pts)
    assert [p["type"] for p in ranked] == ["idiom", "phrase", "word", "word"]
    assert [p["text"] for p in ranked[2:]] == ["a", "d"]  # 同类型按 confidence 降序


def test_rank_points_is_stable_for_equal_keys():
    pts = [
        _make_point(t, [i, i + 1], "word", confidence=0.5)
        for i, t in enumerate(["x", "y", "z"])
    ]
    assert [p["text"] for p in rank_points(pts)] == ["x", "y", "z"]


def test_rank_points_then_id_assignment_matches_order():
    ranked = rank_points(
        [
            _make_point("a", [0, 1], "word", confidence=0.1),
            _make_point("b", [2, 3], "idiom", confidence=0.9),
        ]
    )
    for i, p in enumerate(ranked):
        p["id"] = f"s1#{i}"
    assert ranked[0]["type"] == "idiom"
    assert ranked[0]["id"] == "s1#0"


# ---------------------------------------------------------------------------
# 规则兜底的字段一致性
# ---------------------------------------------------------------------------


def test_rule_fallback_points_emit_confidence_and_risk_defaults():
    pts = rule_fallback_points(ORIGINAL)
    assert pts
    assert all(p["confidence"] == 0.0 and p["risk"] == "" for p in pts)
