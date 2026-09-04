"""unit: scripts/group_lesson_sentences.py 纯函数（窗口/解析/校验/角色/prompt）"""
import pytest

from scripts.group_lesson_sentences import (
    GROUPABLE_TYPES,
    MAX_GROUP_SIZE,
    MIN_GROUP_SIZE,
    _is_ungrouped,
    build_window_prompt,
    collapse_by_text,
    compute_kept_sentence_ids,
    entry_sentence_ids,
    infer_role_in_group,
    normalize_window_groups,
    parse_llm_content,
    split_windows,
)


def _sent(sid, order, text="hello", translation="你好"):
    return {
        "sentence_id": sid,
        "order": order,
        "text": text,
        "translation": translation,
    }


class TestIsUngrouped:
    def test_grouped_false(self):
        assert _is_ungrouped({"group_id": "grp_x"}) is False

    def test_missing_group_true(self):
        assert _is_ungrouped({"sentence_id": "s1"}) is True

    def test_null_group_true(self):
        assert _is_ungrouped({"group_id": None}) is True

    def test_duplicate_sentence_skipped(self):
        # M5 防御：canonical_sentence_id 指向他句 → 不参与建组
        assert _is_ungrouped({
            "sentence_id": "s2",
            "group_id": None,
            "canonical_sentence_id": "s1",
        }) is False

    def test_self_canonical_ok(self):
        assert _is_ungrouped({
            "sentence_id": "s1",
            "canonical_sentence_id": "s1",
        }) is True


class TestEntrySentenceIds:
    def test_plain_sentence(self):
        assert entry_sentence_ids({"sentence_id": "s1"}) == ["s1"]

    def test_collapsed_entry(self):
        assert entry_sentence_ids({"sentence_ids": ["s1", "s2", "s3"]}) == ["s1", "s2", "s3"]

    def test_collapsed_filters_falsy(self):
        assert entry_sentence_ids({"sentence_ids": ["s1", "", None, "s2"]}) == ["s1", "s2"]

    def test_missing_id(self):
        assert entry_sentence_ids({"text": "x"}) == []

    def test_prefers_sentence_ids_over_sentence_id(self):
        assert entry_sentence_ids({
            "sentence_id": "s0",
            "sentence_ids": ["s1", "s2"],
        }) == ["s1", "s2"]


class TestCollapseByText:
    def test_empty(self):
        assert collapse_by_text([]) == []

    def test_dedup_normalized_text(self):
        # 标点/大小写差异 → 归一化文本相同 → 折叠成一个唯一句条目
        rows = [
            {"sentence_id": "s1", "order": 1, "text": "Is this your pen?"},
            {"sentence_id": "s2", "order": 2, "text": "is this your pen"},
            {"sentence_id": "s3", "order": 3, "text": "Yes, it is."},
        ]
        out = collapse_by_text(rows)
        assert len(out) == 2
        first, second = out
        assert first["sentence_ids"] == ["s1", "s2"]
        assert first["dup_count"] == 2
        assert first["order"] == 1
        assert first["sentence_id"] == "s1"  # 代表句 = 输入中首个
        assert second["sentence_ids"] == ["s3"]
        assert second["dup_count"] == 1

    def test_preserves_input_order(self):
        rows = [
            {"sentence_id": "b1", "order": 1, "text": "B?"},
            {"sentence_id": "a1", "order": 2, "text": "A!"},
            {"sentence_id": "b2", "order": 3, "text": "B?"},
        ]
        out = collapse_by_text(rows)
        # 代表句按最小 order 排序（保课内顺序）
        assert [e["sentence_id"] for e in out] == ["b1", "a1"]
        assert out[0]["sentence_ids"] == ["b1", "b2"]

    def test_empty_text_not_merged(self):
        rows = [
            {"sentence_id": "s1", "order": 1, "text": ""},
            {"sentence_id": "s2", "order": 2, "text": "  "},
        ]
        out = collapse_by_text(rows)
        assert len(out) == 2
        assert all(e["dup_count"] == 1 for e in out)

    def test_representative_keeps_other_fields(self):
        rows = [{
            "sentence_id": "s1", "order": 1, "text": "Hi!", "translation": "你好",
            "group_id": None,
        }]
        out = collapse_by_text(rows)
        assert out[0]["translation"] == "你好"
        assert out[0]["group_id"] is None


class TestComputeKept:
    def test_all_grouped(self):
        sents = [{"sentence_id": "s0", "sentence_ids": ["s0"]},
                 {"sentence_id": "s1", "sentence_ids": ["s1"]}]
        groups = [{"sentence_ids": ["s0", "s1"]}]
        assert compute_kept_sentence_ids(sents, groups) == []

    def test_collapsed_partial(self):
        sents = [
            {"sentence_id": "a1", "sentence_ids": ["a1", "a2"]},
            {"sentence_id": "b1", "sentence_ids": ["b1"]},
            {"sentence_id": "c1", "sentence_ids": ["c1", "c2", "c3"]},
        ]
        groups = [{"sentence_ids": ["a1", "a2", "b1"]}]
        assert compute_kept_sentence_ids(sents, groups) == ["c1", "c2", "c3"]

    def test_kept_preserves_expanded_order(self):
        sents = [
            {"sentence_id": "a1", "sentence_ids": ["a1", "a2"]},
            {"sentence_id": "b1", "sentence_ids": ["b1", "b2"]},
        ]
        assert compute_kept_sentence_ids(sents, []) == ["a1", "a2", "b1", "b2"]


class TestSplitWindows:
    def test_empty(self):
        assert split_windows([], 40) == []

    def test_small_single_window(self):
        sents = [_sent(f"s{i}", i) for i in range(5)]
        wins = split_windows(sents, 40)
        assert len(wins) == 1
        assert len(wins[0]) == 5

    def test_split_multi(self):
        sents = [_sent(f"s{i}", i) for i in range(10)]
        wins = split_windows(sents, 4)
        assert [len(w) for w in wins] == [4, 4, 2]
        assert [w[0]["sentence_id"] for w in wins] == ["s0", "s4", "s8"]

    def test_bad_window_clamped(self):
        sents = [_sent(f"s{i}", i) for i in range(3)]
        assert split_windows(sents, 0) == [sents]

    def test_after_collapse_unique_windows(self):
        # 折叠后切窗：同一文本（重复句）只出现在一个窗口
        rows = [
            {"sentence_id": f"s{i}", "order": i, "text": ("Hi!" if i % 2 == 0 else "Bye!")}
            for i in range(6)
        ]
        unique = collapse_by_text(rows)
        assert len(unique) == 2
        wins = split_windows(unique, 40)
        assert wins == [unique]


class TestParseLlmContent:
    def test_plain_json(self):
        r = parse_llm_content('{"groups": [], "skip_indices": []}')
        assert r["ok"] and r["obj"]["groups"] == []

    def test_fence_json(self):
        r = parse_llm_content('```json\n{"groups": [{"indices": [1, 2]}]}\n```')
        assert r["ok"] and r["obj"]["groups"][0]["indices"] == [1, 2]

    def test_trailing_text(self):
        r = parse_llm_content('好的：\n{"a": 1}\n以上。')
        assert r["ok"] and r["obj"]["a"] == 1

    def test_broken(self):
        r = parse_llm_content("no json here")
        assert r["ok"] is False

    def test_empty(self):
        assert parse_llm_content("")["ok"] is False


class TestNormalizeWindowGroups:
    def _win(self, n=6):
        return [_sent(f"s{i}", i) for i in range(n)]

    def test_valid_pairs(self):
        win = self._win(6)
        obj = {
            "groups": [
                {"indices": [1, 2], "type": "dialogue_pair", "title": "问答", "reason": "Q+A"},
                {"indices": [3, 4, 5], "type": "grammar_family", "title": "句型", "reason": "同型"},
            ],
            "skip_indices": [6],
        }
        r = normalize_window_groups(obj, win)
        assert len(r["groups"]) == 2
        g0 = r["groups"][0]
        assert g0["sentence_ids"] == ["s0", "s1"]
        assert g0["type"] == "dialogue_pair"
        assert r["invalid"] == []
        assert r["used"] == {1, 2, 3, 4, 5}

    def test_group_too_small_rejected(self):
        win = self._win(4)
        obj = {"groups": [{"indices": [1], "type": "vocab_family", "title": "单句"}]}
        r = normalize_window_groups(obj, win)
        assert r["groups"] == []
        assert len(r["invalid"]) == 1
        assert f"[{MIN_GROUP_SIZE},{MAX_GROUP_SIZE}]" in r["invalid"][0]["error"]

    def test_group_too_large_rejected(self):
        win = self._win(8)
        obj = {"groups": [{"indices": [1, 2, 3, 4, 5, 6, 7], "type": "vocab_family"}]}
        r = normalize_window_groups(obj, win)
        assert r["groups"] == []
        assert len(r["invalid"]) == 1

    def test_oob_rejected(self):
        win = self._win(4)
        obj = {"groups": [{"indices": [1, 9], "type": "dialogue_pair"}]}
        r = normalize_window_groups(obj, win)
        assert r["groups"] == [] and len(r["invalid"]) == 1
        assert "越界" in r["invalid"][0]["error"]

    def test_overlap_rejected(self):
        win = self._win(5)
        obj = {"groups": [
            {"indices": [1, 2], "type": "dialogue_pair"},
            {"indices": [2, 3], "type": "dialogue_pair"},
        ]}
        r = normalize_window_groups(obj, win)
        assert len(r["groups"]) == 1 and len(r["invalid"]) == 1
        assert "冲突" in r["invalid"][0]["error"]

    def test_bad_type_rejected(self):
        win = self._win(3)
        obj = {"groups": [{"indices": [1, 2], "type": "stand_alone"}]}
        r = normalize_window_groups(obj, win)
        assert r["groups"] == [] and len(r["invalid"]) == 1
        assert r["invalid"][0]["type"] == "stand_alone"

    def test_title_fallback(self):
        win = [{"sentence_id": "s0", "text": "Good morning!"}] + self._win(2)[1:]
        win[1] = {"sentence_id": "s1", "text": "Good morning to you!"}
        obj = {"groups": [{"indices": [1, 2], "type": "vocab_family"}]}
        r = normalize_window_groups(obj, win)
        assert r["groups"][0]["title"] == "Good morning!"[:20]

    def test_order_preserved_sorted(self):
        # LLM 乱序输出 → 组内按窗口原序重排
        win = self._win(4)
        obj = {"groups": [{"indices": [3, 1], "type": "dialogue_pair", "title": "t"}]}
        r = normalize_window_groups(obj, win)
        assert r["groups"][0]["indices"] == [1, 3]
        assert r["groups"][0]["sentence_ids"] == ["s0", "s2"]

    def test_groups_not_list(self):
        win = self._win(3)
        r = normalize_window_groups({"groups": "x"}, win)
        assert r["groups"] == [] and len(r["invalid"]) == 1

    def test_collapsed_entries_expand_sentence_ids(self):
        # 窗口条目为折叠后的唯一句（sentence_ids 含全部重复句）→ 组展开全量、保序
        win = [
            {"sentence_id": "a1", "text": "Is this your pen?",
             "sentence_ids": ["a1", "a2"], "order": 1, "dup_count": 2},
            {"sentence_id": "b1", "text": "Yes, it is.",
             "sentence_ids": ["b1"], "order": 2, "dup_count": 1},
            {"sentence_id": "c1", "text": "No, it is not.",
             "sentence_ids": ["c1", "c2", "c3"], "order": 3, "dup_count": 3},
        ]
        obj = {"groups": [{
            "indices": [1, 2], "type": "dialogue_pair", "title": "问答",
        }]}
        r = normalize_window_groups(obj, win)
        assert len(r["groups"]) == 1
        g = r["groups"][0]
        assert g["indices"] == [1, 2]
        assert g["sentence_ids"] == ["a1", "a2", "b1"]
        assert r["used"] == {1, 2}

    def test_collapsed_llm_unsorted_order_expanded(self):
        win = [
            {"sentence_id": "a1", "text": "Q", "sentence_ids": ["a1", "a2"]},
            {"sentence_id": "b1", "text": "A", "sentence_ids": ["b1", "b2"]},
        ]
        obj = {"groups": [{"indices": [2, 1], "type": "dialogue_pair"}]}
        r = normalize_window_groups(obj, win)
        assert r["groups"][0]["indices"] == [1, 2]
        assert r["groups"][0]["sentence_ids"] == ["a1", "a2", "b1", "b2"]


class TestInferRole:
    def test_dialogue_pair(self):
        assert infer_role_in_group("dialogue_pair", 0) == "question"
        assert infer_role_in_group("dialogue_pair", 1) == "answer_A"
        assert infer_role_in_group("dialogue_pair", 2) == "statement"

    def test_other_types(self):
        for t in ("grammar_family", "vocab_family"):
            assert infer_role_in_group(t, 0) == "statement"
            assert infer_role_in_group(t, 1) == "statement"


class TestBuildPrompt:
    def test_window_lines_ordered(self):
        task = {
            "textbook_title": "新概念英语第一册",
            "lesson_title": "Lesson 1",
            "sentences": [_sent("s0", 0, "Excuse me!"), _sent("s1", 1, "Yes?")],
        }
        msgs = build_window_prompt(task)
        assert msgs[0]["role"] == "system"
        user = msgs[1]["content"]
        assert "新概念英语第一册" in user
        assert "Lesson 1" in user
        assert '1. text: "Excuse me!"' in user
        assert '2. text: "Yes?"' in user
        assert 'translation: "你好"' in user
        assert len(msgs) == 2
