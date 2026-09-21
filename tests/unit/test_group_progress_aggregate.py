"""v4 组维度进度聚合单元测试 — services/english/group_view.py `aggregate_group_progress`

口径（契约 api-contract §3.1 组计数 / data-model-contract §4.22）：
- 句状态 = pick_state + status_to_int（与 summary.mastery 同源）
- **组状态 = 组内句状态取 min（木桶）**
- 已学习组 = 组状态 >= 1；已掌握组 = 组状态 >= 3
- 无 sentence_group 的课次 → 读兼容层逐句合成 legacy 组
- 与能力维度无关（不按 skill_code 过滤）

覆盖：
1. 木桶 + 两个计数（已掌握 / 只算已学习 / 全未学）
2. 多课次混合（一课有真实分组、一课无分组）
3. legacy 回退逐句成组
4. 边界：空输入 → 三计数皆 0；组内 sentence_ids 含未知句 id → 不计入、不崩
"""
from __future__ import annotations

from services.english.group_view import aggregate_group_progress

LESSON = "l1"


def _sentences(*sids, lesson_id: str = LESSON) -> list[dict]:
    return [{"sentence_id": sid, "lesson_id": lesson_id, "text": f"Text {sid}"} for sid in sids]


def _state(sid: str, status: str, mastery: int = 80) -> dict:
    return {
        "scholar_id": "scholar_1",
        "sentence_id": sid,
        "skill_code": "translation",
        "status": status,
        "mastery_score": mastery,
        "attempt_count": 1,
    }


def _group(gid: str, sids: list[str], lesson_id: str = LESSON) -> dict:
    return {
        "group_id": gid,
        "lesson_id": lesson_id,
        "sentence_ids": list(sids),
        "title": f"组 {gid}",
    }


class TestBucketAndCounts:
    def test_min_bucket_drives_two_counts(self):
        """三组：全掌握 / 掌握+learning（木桶=1）/ 全未学 → 2 已学习、1 已掌握。"""
        groups = [_group("g1", ["s1", "s2"]), _group("g2", ["s3", "s4"]), _group("g3", ["s5"])]
        sentences = _sentences("s1", "s2", "s3", "s4", "s5")
        states = [
            _state("s1", "mastered"),
            _state("s2", "mastered"),
            _state("s3", "mastered"),
            _state("s4", "learning", 40),
        ]

        out = aggregate_group_progress(groups=groups, sentences=sentences, states=states)

        assert out["learned_group_count"] == 2  # g1 + g2（g3 木桶 0）
        assert out["mastered_group_count"] == 1  # 仅 g1
        assert out["total_group_count"] == 3
        by_id = {r["group_id"]: r for r in out["groups"]}
        assert by_id["g1"]["group_status"] == 3
        assert by_id["g2"]["group_status"] == 1  # 木桶：最弱句 learning
        assert by_id["g3"]["group_status"] == 0  # 无 skill_state → 未学
        assert by_id["g1"]["sentence_count"] == 2
        assert by_id["g3"]["group_mastery"] == 0.0

    def test_learning_status_sentence_does_not_master_group(self):
        """组内全 learning（1）→ 已学习但未掌握（与「已掌握」口径分离，D2）。"""
        groups = [_group("g1", ["s1"])]
        out = aggregate_group_progress(
            groups=groups,
            sentences=_sentences("s1"),
            states=[_state("s1", "learning", 50)],
        )
        assert out["learned_group_count"] == 1
        assert out["mastered_group_count"] == 0

    def test_skill_code_isolated_states_do_not_matter(self):
        """多技能记录：pick_state 取乐观状态（与句级展示同源），不按 skill_code 分组。"""
        groups = [_group("g1", ["s1"])]
        states = [
            _state("s1", "mastered"),
            {**_state("s1", "not_started", 0), "skill_code": "listening"},
        ]
        out = aggregate_group_progress(groups=groups, sentences=_sentences("s1"), states=states)
        assert out["mastered_group_count"] == 1


class TestMultiLessonAndLegacy:
    def test_mixed_lessons_grouped_and_legacy(self):
        """l1 有真实分组、l2 无分组 → l2 逐句成组，两组一起计入。"""
        groups = [_group("g1", ["s1", "s2"], "l1")]
        sentences = _sentences("s1", "s2", "l1") + _sentences("s3", lesson_id="l2")
        states = [_state("s1", "mastered"), _state("s3", "learned", 70)]

        out = aggregate_group_progress(groups=groups, sentences=sentences, states=states)

        assert out["total_group_count"] == 2  # g1 + legacy_l2_s3
        assert out["learned_group_count"] == 1  # 仅 legacy_l2_s3
        assert out["mastered_group_count"] == 0
        by_id = {r["group_id"]: r for r in out["groups"]}
        assert "legacy_l2_s3" in by_id
        assert by_id["legacy_l2_s3"]["group_status"] == 2  # learned
        # g1 内 s2 无 state → 木桶 = min(3, 0) = 0 → 该组不计已学习/已掌握
        assert by_id["g1"]["group_status"] == 0

    def test_legacy_ids_follow_read_compat_layer(self):
        """无 sentence_group → 组 id 与 getLessonSentenceGroups 同款 legacy_{lesson}_{sentence}。"""
        out = aggregate_group_progress(
            groups=[],
            sentences=_sentences("s1", "s2"),
            states=[_state("s1", "mastered")],
        )
        ids = [r["group_id"] for r in out["groups"]]
        assert ids == ["legacy_l1_s1", "legacy_l1_s2"]
        assert out["learned_group_count"] == 1
        assert out["mastered_group_count"] == 1
        assert out["total_group_count"] == 2


class TestEdgeCases:
    def test_empty_inputs_all_zero(self):
        out = aggregate_group_progress(groups=[], sentences=[], states=[])
        assert out == {
            "learned_group_count": 0,
            "mastered_group_count": 0,
            "total_group_count": 0,
            "groups": [],
        }

    def test_unknown_sentence_id_in_group_is_ignored(self):
        """组内 sentence_ids 含不在 sentences 中的 id → 不计入句数、不崩。"""
        groups = [_group("g1", ["s1", "s_ghost"])]
        out = aggregate_group_progress(
            groups=groups,
            sentences=_sentences("s1"),
            states=[_state("s1", "mastered")],
        )
        assert out["total_group_count"] == 1
        assert out["groups"][0]["sentence_count"] == 1
        assert out["mastered_group_count"] == 1

    def test_group_without_usable_sentences_is_skipped(self):
        """组的 sentence_ids 全部未知 → 该组不产出（避免 min([]) 崩溃）。"""
        groups = [_group("g1", ["s_ghost"])]
        out = aggregate_group_progress(groups=groups, sentences=[], states=[])
        assert out["total_group_count"] == 0
