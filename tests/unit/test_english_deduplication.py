"""E-API-12 批量去重单元测试：dry_run 预览 / 执行级联清理 / canonical 保留

覆盖入口：
- deduplicateEnglishSentences（services/english/deduplication.py）
- 教材/课时不存在 → 404 异常
- canonical 保留口径（created_at 最早 / canonical_sentence_id 自指优先）
- 关联计数（related_data 5 表）
- 级联清理 + 审计写入（deduplicate_english_sentences）
"""
from __future__ import annotations

import asyncio

import pytest

from services.english import LessonNotFoundError, TextbookNotFoundError
from services.english.deduplication import deduplicateEnglishSentences
from tests.fakes.fake_db import FakeDB


def _run(coro):
    return asyncio.run(coro)


def _seed_textbook(db: FakeDB, *, textbook_id: str = "tb_dedup") -> None:
    """独立集合形态教材：textbook_v2 只有计数，层级在独立 chapter/lesson 集合。"""
    db.add(
        "textbook_v2",
        {
            "textbook_id": textbook_id,
            "title": "去重教材",
            "subject_type": "english",
            "chapter_count": 1,
            "lesson_count": 2,
            "sentence_count": 0,
        },
    )
    db.add(
        "chapter",
        {
            "chapter_id": "ch_1",
            "textbook_id": textbook_id,
            "title": "Ch1",
            "order": 1,
        },
    )
    for order, lid in enumerate(("ls_1", "ls_2"), start=1):
        db.add(
            "lesson",
            {
                "lesson_id": lid,
                "chapter_id": "ch_1",
                "textbook_id": textbook_id,
                "title": f"L{order}",
                "order": order,
            },
        )


def _add_sentence(
    db: FakeDB,
    *,
    sentence_id: str,
    text: str,
    lesson_id: str,
    created_at: int,
    textbook_id: str = "tb_dedup",
    extra: dict | None = None,
) -> None:
    doc = {
        "sentence_id": sentence_id,
        "text": text,
        "textbook_id": textbook_id,
        "lesson_id": lesson_id,
        "chapter_id": "ch_1",
        "created_at": created_at,
    }
    if extra:
        doc.update(extra)
    db.add("sentence_v2", doc)


# ===========================================================================
# dry_run 预览（零写入）
# ===========================================================================


class TestDryRunPreview:
    def test_preview_textbook_scope(self):
        db = FakeDB()
        _seed_textbook(db)
        # 2 组重复（normalize 后同 hash：标点/大小写不敏感）
        _add_sentence(db, sentence_id="s1", text="Hello!", lesson_id="ls_1", created_at=100)
        _add_sentence(db, sentence_id="s2", text="hello", lesson_id="ls_1", created_at=200)
        _add_sentence(db, sentence_id="s3", text="Bye!", lesson_id="ls_2", created_at=150)
        _add_sentence(db, sentence_id="s4", text="bye", lesson_id="ls_2", created_at=250)
        _add_sentence(db, sentence_id="s5", text="Unique.", lesson_id="ls_1", created_at=300)

        result = _run(
            deduplicateEnglishSentences(db, textbook_id="tb_dedup", dry_run=True)
        )

        assert result["scope"] == "textbook"
        assert result["dry_run"] is True
        assert result["total_groups"] == 2
        assert result["total_duplicates"] == 2
        assert result["deleted_count"] == 0

        g0 = next(g for g in result["groups"] if g["text_hash"])
        assert g0["text"] == "Hello!"
        assert g0["count"] == 2
        # canonical = created_at 最早（M3 建簇口径）
        assert g0["canonical_sentence_id"] == "s1"
        assert [d["sentence_id"] for d in g0["duplicates"]] == ["s2"]

        # 零写入：sentence_v2 / audit_log 均未变化
        assert len(db.all("sentence_v2")) == 5
        assert db.all("audit_log") == []

    def test_preview_lesson_scope(self):
        db = FakeDB()
        _seed_textbook(db)
        _add_sentence(db, sentence_id="s1", text="Hello!", lesson_id="ls_1", created_at=100)
        _add_sentence(db, sentence_id="s2", text="hello", lesson_id="ls_1", created_at=200)
        # 另一课同样重复——限定课时时不统计
        _add_sentence(db, sentence_id="s3", text="Bye!", lesson_id="ls_2", created_at=150)
        _add_sentence(db, sentence_id="s4", text="bye", lesson_id="ls_2", created_at=250)

        result = _run(
            deduplicateEnglishSentences(
                db, textbook_id="tb_dedup", lesson_id="ls_1", dry_run=True
            )
        )

        assert result["scope"] == "lesson"
        assert result["lesson_id"] == "ls_1"
        assert result["total_groups"] == 1
        assert result["groups"][0]["canonical_sentence_id"] == "s1"
        assert result["total_duplicates"] == 1

    def test_preview_related_data_counts(self):
        db = FakeDB()
        _seed_textbook(db)
        _add_sentence(db, sentence_id="s1", text="Hello!", lesson_id="ls_1", created_at=100)
        _add_sentence(db, sentence_id="s2", text="hello", lesson_id="ls_1", created_at=200)
        db.add("study_attempt", {"sentence_id": "s2", "scholar_id": "u1"})
        db.add("study_attempt", {"sentence_id": "s2", "scholar_id": "u2"})
        db.add("skill_state", {"sentence_id": "s2", "scholar_id": "u1"})

        result = _run(
            deduplicateEnglishSentences(db, textbook_id="tb_dedup", dry_run=True)
        )
        dup = result["groups"][0]["duplicates"][0]
        assert dup["related_data"]["study_attempt_count"] == 2
        assert dup["related_data"]["skill_state_count"] == 1
        assert dup["related_data"]["speech_evaluation_count"] == 0

    def test_no_duplicates(self):
        db = FakeDB()
        _seed_textbook(db)
        _add_sentence(db, sentence_id="s1", text="Hello!", lesson_id="ls_1", created_at=100)
        _add_sentence(db, sentence_id="s3", text="Bye!", lesson_id="ls_2", created_at=150)

        result = _run(
            deduplicateEnglishSentences(db, textbook_id="tb_dedup", dry_run=True)
        )
        assert result["total_groups"] == 0
        assert result["total_duplicates"] == 0
        assert result["groups"] == []

    def test_canonical_self_ref_priority(self):
        """组内已有 canonical_sentence_id 自指 → 优先保留（_pick_canonical_id 口径）。"""
        db = FakeDB()
        _seed_textbook(db)
        _add_sentence(
            db, sentence_id="s1", text="Hello!", lesson_id="ls_1", created_at=200,
            extra={"canonical_sentence_id": "s1"},
        )
        _add_sentence(db, sentence_id="s2", text="hello", lesson_id="ls_1", created_at=100)

        result = _run(
            deduplicateEnglishSentences(db, textbook_id="tb_dedup", dry_run=True)
        )
        g = result["groups"][0]
        assert g["canonical_sentence_id"] == "s1"  # 自指优先，非 created_at 最早
        assert g["duplicates"][0]["sentence_id"] == "s2"


# ===========================================================================
# 执行清理（dry_run=False）：级联删除 + canonical 保留 + 审计
# ===========================================================================


class TestExecute:
    def test_execute_deletes_duplicates_keeps_canonical(self):
        db = FakeDB()
        _seed_textbook(db)
        _add_sentence(db, sentence_id="s1", text="Hello!", lesson_id="ls_1", created_at=100)
        _add_sentence(db, sentence_id="s2", text="hello", lesson_id="ls_1", created_at=200)
        _add_sentence(db, sentence_id="s3", text="Bye!", lesson_id="ls_2", created_at=150)
        _add_sentence(db, sentence_id="s4", text="bye", lesson_id="ls_2", created_at=250)
        _add_sentence(db, sentence_id="s5", text="Unique.", lesson_id="ls_1", created_at=300)
        db.add("learning_attempt", {"sentence_id": "s2", "scholar_id": "u1"})
        db.add("speech_evaluation", {"sentence_id": "s4", "scholar_id": "u1"})

        result = _run(
            deduplicateEnglishSentences(
                db, textbook_id="tb_dedup", dry_run=False, editor_id="op_1"
            )
        )

        assert result["deleted_count"] == 2
        assert result["dry_run"] is False
        remaining = {r["sentence_id"] for r in db.all("sentence_v2")}
        assert remaining == {"s1", "s3", "s5"}  # canonical + 唯一句保留
        # 级联清理：关联记录被物理删
        assert db.all("learning_attempt") == []
        assert db.all("speech_evaluation") == []
        # 审计写入
        audits = db.all("audit_log")
        assert len(audits) == 1
        assert audits[0]["action"] == "deduplicate_english_sentences"
        assert audits[0]["object_ref"] == "tb_dedup"
        assert audits[0]["actor"] == "op_1"
        assert audits[0]["context"]["duplicates_deleted"] == 2
        assert audits[0]["context"]["groups"] == 2

    def test_execute_lesson_scope(self):
        db = FakeDB()
        _seed_textbook(db)
        _add_sentence(db, sentence_id="s1", text="Hello!", lesson_id="ls_1", created_at=100)
        _add_sentence(db, sentence_id="s2", text="hello", lesson_id="ls_1", created_at=200)
        _add_sentence(db, sentence_id="s3", text="Bye!", lesson_id="ls_2", created_at=150)
        _add_sentence(db, sentence_id="s4", text="bye", lesson_id="ls_2", created_at=250)

        result = _run(
            deduplicateEnglishSentences(
                db, textbook_id="tb_dedup", lesson_id="ls_1", dry_run=False
            )
        )

        assert result["deleted_count"] == 1
        remaining = {r["sentence_id"] for r in db.all("sentence_v2")}
        assert remaining == {"s1", "s3", "s4"}  # 仅 ls_1 的重复被清理

    def test_execute_no_duplicates_no_audit(self):
        db = FakeDB()
        _seed_textbook(db)
        _add_sentence(db, sentence_id="s1", text="Hello!", lesson_id="ls_1", created_at=100)

        result = _run(
            deduplicateEnglishSentences(db, textbook_id="tb_dedup", dry_run=False)
        )
        assert result["deleted_count"] == 0
        assert db.all("audit_log") == []  # 无重复不写审计


# ===========================================================================
# 异常语义（404）
# ===========================================================================


class TestErrors:
    def test_textbook_not_found(self):
        db = FakeDB()
        with pytest.raises(TextbookNotFoundError):
            _run(
                deduplicateEnglishSentences(
                    db, textbook_id="tb_nope", dry_run=True
                )
            )

    def test_lesson_not_found(self):
        db = FakeDB()
        _seed_textbook(db)
        with pytest.raises(LessonNotFoundError):
            _run(
                deduplicateEnglishSentences(
                    db, textbook_id="tb_dedup", lesson_id="ls_nope", dry_run=True
                )
            )
