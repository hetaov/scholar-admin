"""共享数据工厂（T1.5）：收敛各测试文件重复的 _seed_* / _payload helper。

约定：helper 接受与业务集合同构的配置，行为与迁移前完全一致。
新测试造数统一走这里，不再在测试文件内定义本地 seed。

默认参数对应最常见的 tb_1 内容层级（1 章 2 课 4 句），
需要差异（跨教材 tb_2 / 1 课 2 句 / 无文本回显）时显式传参覆盖。
"""

from __future__ import annotations

import math
import time

DEFAULT_SENTENCE_IDS = ("s1", "s2", "s3", "s4")
DEFAULT_LESSON_ID = ("l1", "l2")
DEFAULT_CHAPTER_ID = "c1"
DEFAULT_TEXTBOOK_ID = "tb_1"

# 语音评测请求 payload 的默认值（test_speech_eval 原 _payload）
SPEECH_PAYLOAD_DEFAULTS = {
    "scholar_id": "scholar_1001",
    "sentence_id": "sent_0001",
    "original_text": "The quick brown fox jumps over the lazy dog",
    "audio_base64": "ZmFrZS1tcDMtYnl0ZXM=",  # base64("fake-mp3-bytes")
    "voice_format": "mp3",
}


def seed_content(
    fake_db,
    *,
    textbook_id: str = DEFAULT_TEXTBOOK_ID,
    chapter_id: str = DEFAULT_CHAPTER_ID,
    lesson_ids: tuple[str, ...] = DEFAULT_LESSON_ID,
    sentence_ids: tuple[str, ...] = DEFAULT_SENTENCE_IDS,
    include_text: bool = True,
) -> None:
    """预置教材内容层级：1 章 N 课 M 句（sentence_v2 新集合）。

    lesson 默认挂靠 chapter_id / textbook_id；句子按 (sentence_id, lesson_id)
    顺序一一分配到各课（每课平均分摊）。include_text=False 时跳过
    text/translation 字段（学者×教材关联场景只需层级 id）。
    """
    fake_db.add("chapter", {
        "chapter_id": chapter_id, "textbook_id": textbook_id,
        "title": "Ch1", "order": 1,
    })
    for order, lid in enumerate(lesson_ids, start=1):
        fake_db.add("lesson", {
            "lesson_id": lid, "chapter_id": chapter_id,
            "textbook_id": textbook_id, "title": f"L{order}", "order": order,
        })
    for i, (sid, lid) in enumerate(sentence_ids_assign(sentence_ids, lesson_ids), start=1):
        doc = {
            "sentence_id": sid, "lesson_id": lid, "chapter_id": chapter_id,
            "textbook_id": textbook_id, "order": i,
        }
        if include_text:
            doc["text"] = f"Text {sid}"
            doc["translation"] = f"译{sid}"
        fake_db.add("sentence_v2", doc)


def sentence_ids_assign(
    sentence_ids: tuple[str, ...], lesson_ids: tuple[str, ...]
) -> list[tuple[str, str]]:
    """句子按顺序连续分块到各课（前 chunk 句到 l1，依此类推）。"""
    chunk = math.ceil(len(sentence_ids) / len(lesson_ids))
    return [(sid, lesson_ids[i // chunk]) for i, sid in enumerate(sentence_ids)]


def seed_skill_states(fake_db, states: list[dict]) -> None:
    """批量写入 skill_state（支持传入完整 dict 列表）。"""
    for st in states:
        fake_db.add("skill_state", st)


def seed_attempt(fake_db, scholar_id: str = "u1") -> str:
    """写入一条 learning_attempt，返回其 _id（评估证据链路用）。"""
    doc = fake_db.add(
        "learning_attempt",
        {
            "scholar_id": scholar_id,
            "sentence_id": "s1",
            "mode": "study",
            "original_text": "It is a watch.",
            "user_input": "it is a watch",
            "created_at": 1,
        },
    )
    return str(doc["_id"])


def seed_speech(fake_db, scholar_id: str = "u1") -> str:
    """写入一条 speech_evaluation（含 SOE-N parsed），返回其 _id。"""
    doc = fake_db.add(
        "speech_evaluation",
        {
            "scholar_id": scholar_id,
            "sentence_id": "s1",
            "original_text": "It is a watch.",
            "parsed": {
                "accuracy": 85.0,
                "fluency": 80.0,
                "completion": 95.0,
                "suggested_score": 88.0,
            },
            "raw": {},
            "created_at": 1,
        },
    )
    return str(doc["_id"])


def seed_task(fake_db, **overrides) -> dict:
    """写入一条 dialogue_task（默认 pending），返回写入的文档。

    覆盖 overrides 中的任意字段；未提供时 expires_at 按默认 TTL 24h 计算。
    """
    from services.dialogue_task import TASK_TTL_MS  # 延迟导入避免循环依赖

    now = int(time.time() * 1000)
    doc = {
        "task_id": "dt_test",
        "scholar_id": "s1",
        "sentence": "Hello",
        "status": "pending",
        "result": None,
        "is_question": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    doc.update(overrides)
    fake_db.add("dialogue_task", doc)
    return doc


def seed_translation_task(fake_db, **overrides) -> dict:
    """写入一条 translation_task（默认 pending），返回写入的文档。

    覆盖 overrides 中的任意字段；未提供时 expires_at 按默认 TTL 24h 计算。
    默认 ec（英译中）文字路径：original_text=英文原句，user_input=中文译文。
    """
    from services.translation_task import TASK_TTL_MS  # 延迟导入避免循环依赖

    now = int(time.time() * 1000)
    doc = {
        "task_id": "tr_test",
        "scholar_id": None,
        "sentence_id": None,
        "original_text": "It is a watch.",
        "user_input": "它是一块手表。",
        "audio_base64": None,
        "voice_format": "mp3",
        "input_mode": "text",
        "mode": "ec",
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    doc.update(overrides)
    fake_db.add("translation_task", doc)
    return doc


def speech_payload(**overrides) -> dict:
    """语音评测请求 payload，支持按字段覆盖。"""
    payload = dict(SPEECH_PAYLOAD_DEFAULTS)
    payload.update(overrides)
    return payload
