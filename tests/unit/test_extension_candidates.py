"""单元测试：候选召回层 services/english/extension_candidates.py

覆盖（分层抽取第一段：规则/词表生成候选）：
- tokenize_with_spans：字符偏移正确、不拆缩写/连字符词、兼容弯撇号
- generate_candidates：`text == original[start:end]`（R7 前提，绝不 re-join token）
- 词表命中（slang/idiom/phrase）定位真实 span；同区间优先级 lexicon > n-gram > word
- 实词候选排除停用词与短词；去重；候选上限与 word 桶保留
- expand_abbreviations：仅作 prompt 提示
"""
from __future__ import annotations

import config
from services.english.extension_candidates import (
    expand_abbreviations,
    generate_candidates,
    tokenize_with_spans,
)

SENTENCES = [
    "I will take part in the discussion.",
    "I'm gonna figure it out later.",
    "It is a well-known fact.",
    "Hello, world! Don't worry.",
    "She said it’s up to you.",
    "Let's hit the books tonight.",
    "No biggie, we can hang out later.",
    "As a matter of fact, I disagree.",
]


def _texts(cands):
    return [c["text"] for c in cands]


def _lower_texts(cands):
    return [c["text"].lower() for c in cands]


# ---------------------------------------------------------------------------
# 切词
# ---------------------------------------------------------------------------


def test_tokenize_keeps_char_offsets():
    toks = tokenize_with_spans("Hello, world!")
    assert toks == [("Hello", 0, 5), ("world", 7, 12)]


def test_tokenize_keeps_contractions_and_hyphenated_words_intact():
    toks = tokenize_with_spans("don't it's well-known")
    assert [t[0] for t in toks] == ["don't", "it's", "well-known"]


def test_tokenize_handles_curly_apostrophe():
    toks = tokenize_with_spans("it’s fine")
    assert toks[0][0] == "it’s"
    assert toks[0][1] == 0 and toks[0][2] == 4


def test_tokenize_empty_input():
    assert tokenize_with_spans("") == []


# ---------------------------------------------------------------------------
# R7 前提：候选 text 必须是原句的精确切片
# ---------------------------------------------------------------------------


def test_every_candidate_text_slices_original_exactly():
    for s in SENTENCES:
        for c in generate_candidates(s):
            start, end = c["span"]
            assert s[start:end] == c["text"], (s, c)


def test_hyphenated_and_punctuated_spans_are_not_rejoined():
    """"well-known" / "Hello, world" 若由 token 拼接会与原文不符 → 必须取切片。"""
    for s in ("It is a well-known fact.", "Hello, world! Don't worry."):
        spans = [tuple(c["span"]) for c in generate_candidates(s)]
        assert spans, s
        for c in generate_candidates(s):
            assert s[c["span"][0] : c["span"][1]] == c["text"]


# ---------------------------------------------------------------------------
# 词表命中
# ---------------------------------------------------------------------------


def test_phrase_seed_hit_has_real_span():
    s = "I will take part in the discussion."
    hits = [c for c in generate_candidates(s) if c["text"] == "take part in"]
    assert hits, _texts(generate_candidates(s))
    assert hits[0]["source"] == "lexicon"
    assert s[hits[0]["span"][0] : hits[0]["span"][1]] == "take part in"


def test_idiom_and_slang_seed_hits():
    s = "No biggie, let's hit the books."
    by_text = {c["text"].lower(): c for c in generate_candidates(s)}
    assert by_text["no biggie"]["hint_type"] == "slang"
    assert by_text["hit the books"]["hint_type"] == "idiom"


def test_lexicon_priority_beats_word_hint_on_same_span():
    # "dude" 同时是 slang 种子与实词；同区间应保留高优先级（slang）
    cands = generate_candidates("Hey dude, what's up?")
    same = [c for c in cands if c["text"].lower() == "dude"]
    assert same and all(c["hint_type"] == "slang" for c in same)


# ---------------------------------------------------------------------------
# 实词 / 停用词
# ---------------------------------------------------------------------------


def test_content_words_kept_stopwords_and_short_dropped():
    texts = _lower_texts(generate_candidates("It is a big deal."))
    assert "big" in texts and "deal" in texts
    assert "is" not in texts and "a" not in texts and "it" not in texts


def test_contractions_are_not_content_word_candidates():
    texts = _lower_texts(generate_candidates("Don't worry about it."))
    assert "don't" not in texts
    assert "worry" in texts


# ---------------------------------------------------------------------------
# 结构与上限
# ---------------------------------------------------------------------------


def test_candidate_ids_unique_and_sequential():
    cands = generate_candidates("I will take part in the discussion.")
    ids = [c["candidate_id"] for c in cands]
    assert ids == [f"c{i}" for i in range(1, len(ids) + 1)]


def test_spans_are_unique():
    for s in SENTENCES:
        spans = [tuple(c["span"]) for c in generate_candidates(s)]
        assert len(spans) == len(set(spans)), s


def test_candidate_count_respects_max(monkeypatch):
    monkeypatch.setattr(config, "EXTENSION_CANDIDATE_MAX", 12)
    s = "The quick brown fox jumps over the lazy dog near the river bank every morning."
    cands = generate_candidates(s)
    assert len(cands) <= 12
    # word 桶受保护：截断后仍应有实词候选
    assert any(c["hint_type"] == "word" for c in cands)


def test_empty_and_whitespace_input():
    assert generate_candidates("") == []
    assert generate_candidates("   ") == []


# ---------------------------------------------------------------------------
# 缩写展开（仅 prompt 提示）
# ---------------------------------------------------------------------------


def test_expand_abbreviations_finds_speech_forms():
    got = dict(expand_abbreviations("I'm gonna figure it out, cuz I wanna know."))
    assert got["gonna"] == "going to"
    assert got["wanna"] == "want to"
    assert got["cuz"] == "because"


def test_expand_abbreviations_empty_when_none():
    assert expand_abbreviations("I will take part in the discussion.") == []
