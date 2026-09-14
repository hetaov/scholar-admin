"""单元测试：本地 NC2 语料加载器（T2 / 设计稿 §3.1、§3.2、§8.1-4）

覆盖：路径解析 / corpus 读写与结构校验 / 任务组迭代 / 场景构建 /
模拟学者读写与指标摘要。全程文件级，不触网不连库。
"""
from __future__ import annotations

import json

import pytest

from services.learning.local_corpus import (
    BOOK_ID,
    CHAPTER_ID,
    WEAK_MASTERY_THRESHOLD,
    build_ids,
    build_local_scenario,
    compute_weak_skills,
    corpus_path,
    empty_corpus,
    empty_learner,
    learner_path,
    learner_summary,
    lesson_sentence_ids,
    iter_lesson_groups,
    list_lessons,
    load_corpus,
    load_learner,
    project_root,
    resolve_data_dir,
    save_corpus,
    save_learner,
    sentence_index,
    try_load_corpus,
    validate_corpus,
)
from tests.fakes.seed_factory import seed_local_corpus


class TestBuildIds:
    def test_lesson_ids_are_zero_padded(self):
        ids = build_ids(3)
        assert ids["textbook_id"] == BOOK_ID
        assert ids["chapter_id"] == CHAPTER_ID
        assert ids["lesson_id"] == "l_nc2_03"
        assert ids["group_id"] == "g_nc2_03_a"
        assert ids["sentence_id"](2) == "s_nc2_03_02"

    def test_two_digit_lesson_keeps_number(self):
        assert build_ids(20)["lesson_id"] == "l_nc2_20"


class TestResolveDataDir:
    def test_relative_resolved_under_project_root(self):
        assert resolve_data_dir("data/nc2") == project_root() / "data" / "nc2"

    def test_absolute_kept(self, tmp_path):
        assert resolve_data_dir(tmp_path) == tmp_path


class TestCorpusIO:
    def test_missing_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_corpus(tmp_path / "corpus.json")

    def test_try_load_degrades_to_empty(self, tmp_path):
        corpus = try_load_corpus(tmp_path / "corpus.json")
        assert corpus == empty_corpus()

    def test_try_load_degrades_on_bad_json(self, tmp_path):
        bad = tmp_path / "corpus.json"
        bad.write_text("{not json", encoding="utf-8")
        assert try_load_corpus(bad) == empty_corpus()

    def test_validate_rejects_incomplete(self):
        with pytest.raises(ValueError):
            validate_corpus({"book": {}})

    def test_save_load_roundtrip(self, tmp_path):
        seeded = seed_local_corpus(tmp_path, lessons=(1, 2))
        loaded = load_corpus(seeded["corpus_path"])
        assert loaded["book"]["textbook_id"] == BOOK_ID
        assert len(loaded["lessons"]) == 2
        assert len(loaded["sentences"]) == 4
        assert len(loaded["groups"]) == 2

    def test_save_writes_utf8_json(self, tmp_path):
        corpus = seed_local_corpus(tmp_path, lessons=(1,))["corpus"]
        target = tmp_path / "x" / "corpus.json"
        save_corpus(corpus, target)
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["sentences"][0]["translation"] == "上周我去看了戏。"
        assert corpus_path(tmp_path) == tmp_path / "corpus.json"


class TestHierarchy:
    def test_sentence_index_and_lesson_ids(self, tmp_path):
        corpus = seed_local_corpus(tmp_path, lessons=(1, 2))["corpus"]
        index = sentence_index(corpus)
        assert set(index) == {
            "s_nc2_01_01",
            "s_nc2_01_02",
            "s_nc2_02_01",
            "s_nc2_02_02",
        }
        assert lesson_sentence_ids(corpus, "l_nc2_02") == ["s_nc2_02_01", "s_nc2_02_02"]

    def test_list_lessons_sorted_with_counts(self, tmp_path):
        corpus = seed_local_corpus(tmp_path, lessons=(1, 2))["corpus"]
        lessons = list_lessons(corpus)
        assert [x["lesson_id"] for x in lessons] == ["l_nc2_01", "l_nc2_02"]
        assert lessons[0]["sentence_count"] == 2
        assert lessons[0]["title"] == "Lesson 1 A private conversation"


class TestIterLessonGroups:
    def test_returns_task_group_shape(self, tmp_path):
        corpus = seed_local_corpus(tmp_path, lessons=(1, 2))["corpus"]
        groups = iter_lesson_groups(corpus)
        assert len(groups) == 2
        first = groups[0]
        assert first["lesson_id"] == "l_nc2_01"
        assert first["group_id"] == "g_nc2_01_a"
        assert first["group_label"] == "L1 A private conversation"
        assert first["sentences"] == [
            {"sentence_id": "s_nc2_01_01", "content": "Last week I went to the theatre."},
            {"sentence_id": "s_nc2_01_02", "content": "I had a very good seat."},
        ]

    def test_lesson_filter(self, tmp_path):
        corpus = seed_local_corpus(tmp_path, lessons=(1, 2, 3))["corpus"]
        groups = iter_lesson_groups(corpus, ["l_nc2_02"])
        assert [g["lesson_id"] for g in groups] == ["l_nc2_02"]

    def test_skips_groups_with_missing_or_empty_sentences(self, tmp_path):
        corpus = seed_local_corpus(tmp_path, lessons=(1,))["corpus"]
        corpus["groups"].append(
            {"group_id": "g_empty", "lesson_id": "l_nc2_01", "group_label": "空组",
             "sentence_ids": ["missing_id"]}
        )
        groups = iter_lesson_groups(corpus)
        assert [g["group_id"] for g in groups] == ["g_nc2_01_a"]

    def test_empty_corpus_yields_nothing(self):
        assert iter_lesson_groups(empty_corpus()) == []


class TestBuildLocalScenario:
    def test_uses_lesson_title_and_group_label(self, tmp_path):
        corpus = seed_local_corpus(tmp_path, lessons=(1,))["corpus"]
        scenario = build_local_scenario(corpus, "l_nc2_01")
        assert "Lesson 1 A private conversation" in scenario["background"]
        assert "L1 A private conversation" in scenario["goal"]

    def test_group_id_selects_label(self, tmp_path):
        corpus = seed_local_corpus(tmp_path, lessons=(1, 2))["corpus"]
        scenario = build_local_scenario(corpus, None, group_id="g_nc2_02_a")
        assert "L2 Breakfast or lunch?" in scenario["goal"]

    def test_unknown_lesson_falls_back(self):
        scenario = build_local_scenario(empty_corpus(), "l_x")
        assert scenario["background"]
        assert scenario["goal"] == "自然用出任务组句子"


class TestLearnerIO:
    def test_missing_returns_empty_learner(self, tmp_path):
        learner = load_learner("scholar_x", tmp_path)
        assert learner == empty_learner("scholar_x")

    def test_save_and_load(self, tmp_path):
        learner = empty_learner("scholar_x")
        learner["skill_states"] = [{"sentence_id": "s1", "mastery": 0.4}]
        target = save_learner(learner, tmp_path)
        assert target == learner_path("scholar_x", tmp_path)
        assert load_learner("scholar_x", tmp_path)["skill_states"][0]["mastery"] == 0.4

    def test_compute_weak_skills_below_threshold(self):
        learner = {
            "skill_states": [
                {"lesson_id": "l2", "mastery": 0.3},
                {"lesson_id": "l1", "mastery": 0.5},
                {"lesson_id": "l3", "mastery": 0.9},
            ]
        }
        assert compute_weak_skills(learner) == ["l2", "l1"]

    def test_learner_summary(self):
        learner = {
            "scholar_id": "s",
            "skill_states": [{"mastery": 0.5, "confidence": 0.6, "lesson_id": "l1"}],
            "attempts": [{"a": 1}, {"a": 2}],
            "weak_skills": ["l1"],
            "updated_at": 123,
        }
        summary = learner_summary(learner)
        assert summary["skill_state_count"] == 1
        assert summary["attempt_count"] == 2
        assert summary["avg_mastery"] == 0.5
        assert summary["avg_confidence"] == 0.6
        assert summary["weak_skills"] == ["l1"]
        assert summary["available"] is True

    def test_learner_summary_empty(self):
        summary = learner_summary(empty_learner("s"))
        assert summary["available"] is False
        assert summary["weak_skills"] == []

    def test_weak_threshold_constant_is_standard(self):
        assert WEAK_MASTERY_THRESHOLD == 0.6
