"""单元测试：NC2 语料准备脚本（T2 / 设计稿 §3.1、§3.2、§8.4-1）

覆盖：课号表达式解析 / 内置 20 篇兜底 / 原文解析（best-effort）/
corpus 结构与 id 规范 / CLI 端到端（--out 落盘可被 loader 读回）。
"""
from __future__ import annotations

import pytest

from scripts.prepare_nc2_corpus import (
    SEED_LESSONS,
    _SEED_BY_NO,
    build_corpus,
    lessons_data,
    load_source,
    main,
    parse_lessons_arg,
    parse_source_text,
)
from services.learning.local_corpus import load_corpus


class TestParseLessonsArg:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("1-3", [1, 2, 3]),
            ("1,3,5", [1, 3, 5]),
            ("5-3", [3, 4, 5]),
            ("1-2,4", [1, 2, 4]),
            ("2,2,1", [1, 2]),
        ],
    )
    def test_ok(self, spec, expected):
        assert parse_lessons_arg(spec) == expected

    @pytest.mark.parametrize("spec", ["", " ", "a", "0"])
    def test_invalid(self, spec):
        with pytest.raises((ValueError, TypeError)):
            parse_lessons_arg(spec)


class TestSeedCorpus:
    def test_seed_has_20_lessons(self):
        assert len(SEED_LESSONS) == 20
        assert set(_SEED_BY_NO) == set(range(1, 21))

    def test_build_corpus_structure(self):
        corpus = build_corpus(lessons_data([1, 2]))
        assert corpus["book"]["textbook_id"] == "tb_nc2"
        assert len(corpus["chapters"]) == 1
        assert [l["lesson_id"] for l in corpus["lessons"]] == ["l_nc2_01", "l_nc2_02"]
        assert [l["title"] for l in corpus["lessons"]] == [
            "Lesson 1 A private conversation",
            "Lesson 2 Breakfast or lunch?",
        ]
        assert len(corpus["sentences"]) == 4
        first = corpus["sentences"][0]
        assert first["sentence_id"] == "s_nc2_01_01"
        assert first["text"] == "Last week I went to the theatre."
        assert first["translation"] == "上周我去看了戏。"
        assert first["order"] == 1
        assert corpus["groups"][0]["sentence_ids"] == ["s_nc2_01_01", "s_nc2_01_02"]
        assert corpus["groups"][0]["group_label"] == "L1 A private conversation"
        assert corpus["meta"]["lesson_count"] == 2
        assert corpus["meta"]["sentence_count"] == 4
        assert corpus["meta"]["group_count"] == 2

    def test_full_20_lessons_ids_are_consistent(self):
        corpus = build_corpus(lessons_data(list(range(1, 21))))
        assert len(corpus["lessons"]) == 20
        assert len(corpus["sentences"]) == 40
        assert corpus["lessons"][19]["lesson_id"] == "l_nc2_20"
        assert corpus["sentences"][-1]["sentence_id"] == "s_nc2_20_02"

    def test_unknown_lesson_gets_placeholder_sentences(self):
        corpus = build_corpus(lessons_data([21]))
        assert corpus["lessons"][0]["title"] == "Lesson 21 Practice 21"
        assert corpus["meta"]["sentence_count"] == 2
        assert corpus["groups"][0]["sentence_ids"] == ["s_nc2_21_01", "s_nc2_21_02"]


class TestParseSourceText:
    TEXT = (
        "Lesson 1 A private conversation\n"
        "Last week I went to the theatre. I had a very good seat.\n"
        "\n"
        "Lesson 2 Breakfast or lunch\n"
        "It was Sunday. I never get up early on Sundays.\n"
    )

    def test_parses_lessons_and_titles(self):
        parsed = parse_source_text(self.TEXT)
        assert set(parsed) == {1, 2}
        assert parsed[1]["title"] == "A private conversation"
        assert parsed[1]["sentences"] == [
            "Last week I went to the theatre.",
            "I had a very good seat.",
        ]

    def test_numeric_header_parses(self):
        parsed = parse_source_text("3\nPostcards always spoil my holidays. I love them.\n")
        assert 3 in parsed
        assert parsed[3]["sentences"] == [
            "Postcards always spoil my holidays.",
            "I love them.",
        ]

    def test_no_header_returns_empty(self):
        assert parse_source_text("just some free text without a lesson number") == {}

    def test_source_overrides_seed_sentences(self):
        parsed = parse_source_text(self.TEXT)
        data = lessons_data([1, 2], parsed)
        assert data[0]["source"] == "source"
        assert data[0]["sentences"][0]["text"] == "Last week I went to the theatre."
        # 未在原文中出现的课仍回落内置占位文本
        data2 = lessons_data([1, 3], parsed)
        assert data2[1]["source"] == "seed"

    def test_load_source_from_dir(self, tmp_path):
        (tmp_path / "a.txt").write_text(self.TEXT, encoding="utf-8")
        parsed = load_source(tmp_path)
        assert set(parsed) == {1, 2}


class TestCli:
    def test_main_writes_corpus_readable_by_loader(self, tmp_path):
        code = main(["--out", str(tmp_path), "--lessons", "1-3", "--quiet"])
        assert code == 0
        corpus = load_corpus(tmp_path / "corpus.json")
        assert corpus["meta"]["lesson_count"] == 3
        assert (tmp_path / "learners").is_dir()
        assert (tmp_path / "generated").is_dir()

    def test_main_with_source_file(self, tmp_path):
        source = tmp_path / "src.txt"
        source.write_text(TestParseSourceText.TEXT, encoding="utf-8")
        code = main(
            ["--out", str(tmp_path / "out"), "--lessons", "1-2", "--source", str(source), "--quiet"]
        )
        assert code == 0
        corpus = load_corpus(tmp_path / "out" / "corpus.json")
        assert corpus["sentences"][0]["text"] == "Last week I went to the theatre."
