"""共享数据工厂（T1.5）：收敛各测试文件重复的 _seed_* / _payload helper。

约定：helper 接受与业务集合同构的配置，行为与迁移前完全一致。
新测试造数统一走这里，不再在测试文件内定义本地 seed。

默认参数对应最常见的 tb_1 内容层级（1 章 2 课 4 句），
需要差异（跨教材 tb_2 / 1 课 2 句 / 无文本回显）时显式传参覆盖。
"""

from __future__ import annotations

import json
import math
import time

DEFAULT_SENTENCE_IDS = ("s1", "s2", "s3", "s4")
DEFAULT_LESSON_ID = ("l1", "l2")
DEFAULT_CHAPTER_ID = "c1"
DEFAULT_TEXTBOOK_ID = "tb_1"

# 本地 NC2 对话生成语料默认值（T2 / 设计稿 §3.1、§8.1-4）
DIALOGUE_LESSON_IDS = ("l_nc2_01", "l_nc2_02")
DIALOGUE_SENTENCE_IDS = ("s_nc2_01_01", "s_nc2_01_02", "s_nc2_02_01", "s_nc2_02_02")
DIALOGUE_SENTENCE_TEXTS = {
    "s_nc2_01_01": "Last week I went to the theatre.",
    "s_nc2_01_02": "I had a very good seat.",
    "s_nc2_02_01": "It was Sunday.",
    "s_nc2_02_02": "I never get up early on Sundays.",
}
DIALOGUE_SENTENCE_TRANSLATIONS = {
    "s_nc2_01_01": "上周我去看了戏。",
    "s_nc2_01_02": "我的座位很好。",
    "s_nc2_02_01": "那天是星期天。",
    "s_nc2_02_02": "星期天我从不早起。",
}

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
    texts: dict[str, str] | None = None,
    translations: dict[str, str] | None = None,
) -> None:
    """预置教材内容层级：1 章 N 课 M 句（sentence_v2 新集合）。

    lesson 默认挂靠 chapter_id / textbook_id；句子按 (sentence_id, lesson_id)
    顺序一一分配到各课（每课平均分摊）。include_text=False 时跳过
    text/translation 字段（学者×教材关联场景只需层级 id）。
    texts / translations 可按 sentence_id 覆盖文本（对话生成/RAG 场景用真实语句）。
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
            doc["text"] = (texts or {}).get(sid, f"Text {sid}")
            doc["translation"] = (translations or {}).get(sid, f"译{sid}")
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


def seed_ai_session(fake_db, **overrides) -> dict:
    """写入一条 ai_session（沉浸式 AI 会话 v2，§4.19），返回写入的文档。

    默认 scholar_id=scholar_1 / 空 history / 空闲在途位 / expires_at 24h 后。
    覆盖 overrides 中的任意字段。
    """
    from services.learning.session_state import SESSION_TTL_MS  # 延迟导入避免循环依赖

    now = int(time.time() * 1000)
    doc = {
        "session_id": "s_test",
        "scholar_id": "scholar_1",
        "scenario": {"scene": "At the airport"},
        "roles": {
            "ai_role": {"name": "Airport Staff", "style": "kind"},
            "learner_role": {"name": "Passenger"},
        },
        "materials": [
            {
                "kind": "new",
                "sentences": [{"sentence_id": "sid_1", "content": "I'd like to check in."}],
            }
        ],
        "history": [],
        "assisted_count": 0,
        "pending_task": None,
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "expires_at": now + SESSION_TTL_MS,
    }
    doc.update(overrides)
    fake_db.add("ai_session", doc)
    return doc


def seed_ai_session_task(fake_db, **overrides) -> dict:
    """写入一条 ai_session_task（默认 pending，§4.18），返回写入的文档。

    覆盖 overrides 中的任意字段；未提供时 expires_at 按默认 TTL 24h 计算。
    """
    from services.learning.session_task import TASK_TTL_MS  # 延迟导入避免循环依赖

    now = int(time.time() * 1000)
    doc = {
        "task_id": "st_test",
        "scholar_id": "scholar_1",
        "session_id": "s_test",
        "mode": "start",
        "preferred_type": "auto",
        "status": "pending",
        "result": None,
        "error": None,
        "context": {"mode": "start", "scenario": {}, "roles": {}, "materials": [], "history": []},
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    doc.update(overrides)
    fake_db.add("ai_session_task", doc)
    return doc


def seed_ai_session_v3(fake_db, **overrides) -> dict:
    """写入一条 ai_session_v3（沉浸式 AI 会话 v3，§11.2），返回写入的文档。

    与 v2 `seed_ai_session` 同构，仅集合名不同（`ai_session_v3`）。
    默认 scholar_id=scholar_1 / 空 history / 空闲在途位 / expires_at 24h 后。
    """
    from services.learning.session_state_v3 import SESSION_TTL_MS  # 延迟导入

    now = int(time.time() * 1000)
    doc = {
        "session_id": "s_test",
        "scholar_id": "scholar_1",
        "scenario": {"scene": "At the airport"},
        "roles": {
            "ai_role": {"name": "Airport Staff", "style": "kind"},
            "learner_role": {"name": "Passenger"},
        },
        "materials": [
            {
                "kind": "new",
                "sentences": [{"sentence_id": "sid_1", "content": "I'd like to check in."}],
            }
        ],
        "history": [],
        "assisted_count": 0,
        "pending_task": None,
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "expires_at": now + SESSION_TTL_MS,
    }
    doc.update(overrides)
    fake_db.add("ai_session_v3", doc)
    return doc


def seed_ai_session_v3_task(fake_db, **overrides) -> dict:
    """写入一条 ai_session_v3_task（默认 pending，§11.2），返回写入的文档。

    与 v2 `seed_ai_session_task` 同构，仅集合名不同（`ai_session_v3_task`）。
    """
    from services.learning.session_task_v3 import TASK_TTL_MS  # 延迟导入

    now = int(time.time() * 1000)
    doc = {
        "task_id": "st_test",
        "scholar_id": "scholar_1",
        "session_id": "s_test",
        "mode": "start",
        "preferred_type": "auto",
        "status": "pending",
        "result": None,
        "error": None,
        "context": {"mode": "start", "scenario": {}, "roles": {}, "materials": [], "history": []},
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    doc.update(overrides)
    fake_db.add("ai_session_v3_task", doc)
    return doc


def seed_dialogue_gen_task(fake_db, **overrides) -> dict:
    """写入一条 ai_dialogue_task（批量对话生成，§6.1，默认 pending），返回写入的文档。

    覆盖 overrides 中的任意字段；未提供时 expires_at 按默认 TTL 24h 计算。
    """
    from services.learning.dialogue_gen_task import TASK_TTL_MS  # 延迟导入避免循环依赖

    now = int(time.time() * 1000)
    doc = {
        "task_id": "dg_test",
        "scholar_id": "scholar_1",
        "status": "pending",
        "result": None,
        "error": None,
        "context": {
            "task_group": {
                "lesson_id": "l_nc2_01",
                "group_id": "g_nc2_01_a",
                "group_label": "L1 叙述组",
                "sentences": [
                    {"sentence_id": "s_nc2_01_01", "content": "Last week I went to the theatre."}
                ],
            },
            "scenario": {"background": "两个同学在讨论上周末的活动"},
            "roles": [
                {"code": "A", "name": "Tom", "identity": "student"},
                {"code": "B", "name": "Lily", "identity": "classmate"},
            ],
            "recall": {"enabled": True, "top_k": 4},
            "metrics": {"enabled": True, "weak_skills": ["past_tense"]},
            "prompt_lang": "zh",
            "preferred_type": "auto",
        },
        "checkpoint_id": None,
        "retry_count": 0,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    doc.update(overrides)
    fake_db.add("ai_dialogue_task", doc)
    return doc


# ---------------------------------------------------------------------------
# T2：本地语料 / 指标造数（设计稿 §3.1、§8.1-4）
# ---------------------------------------------------------------------------


def dialogue_gen_v2_flow_gen(base):
    """把 T1 直连 fake generator 包装成 v2 全流程 fake（每句评估/总结分流）。

    识别 v2 新增节点的 system prompt（单句评估 / 背景总结）直接返回合法 JSON，
    其余调用透传给 `base` —— 计数类断言只统计「对话生成」调用，与 v1 行为一致。
    """
    import re

    async def gen(messages):
        sys_msg = messages[0]["content"]
        if "能否在真实人际对话里自然地用出" in sys_msg:
            m = re.search(r"\[([^\]]+)\]", messages[1]["content"])
            sid = m.group(1) if m else ""
            return json.dumps(
                {
                    "viable": True,
                    "recalled_used": [],
                    "naturalness": 0.85,
                    "turns": [
                        {"speaker": "A", "text": "line one", "target_sentence_id": sid},
                        {"speaker": "B", "text": "line two", "target_sentence_id": None},
                    ],
                    "reason": "ok",
                },
                ensure_ascii=False,
            )
        if "反推" in sys_msg:
            return json.dumps(
                {
                    "background": "周末下午，两位同学在咖啡店复习功课。",
                    "roles": [
                        {"code": "A", "name": "Tom", "identity": "student"},
                        {"code": "B", "name": "Lily", "identity": "classmate"},
                    ],
                },
                ensure_ascii=False,
            )
        return await base(messages)

    return gen


def seed_dialogue_corpus(
    fake_db,
    *,
    textbook_id: str = "tb_nc2",
    chapter_id: str = "ch_nc2_1",
    lesson_ids: tuple[str, ...] = DIALOGUE_LESSON_IDS,
    sentence_ids: tuple[str, ...] = DIALOGUE_SENTENCE_IDS,
    texts: dict[str, str] | None = None,
    translations: dict[str, str] | None = None,
) -> None:
    """预置对话生成教材层级（textbook_v2/chapter/lesson/sentence_v2，复用 seed_content）。"""
    merged_texts = {**DIALOGUE_SENTENCE_TEXTS, **(texts or {})}
    merged_translations = {**DIALOGUE_SENTENCE_TRANSLATIONS, **(translations or {})}
    seed_content(
        fake_db,
        textbook_id=textbook_id,
        chapter_id=chapter_id,
        lesson_ids=lesson_ids,
        sentence_ids=sentence_ids,
        texts=merged_texts,
        translations=merged_translations,
    )


def seed_skill_states_with_metrics(
    fake_db,
    *,
    scholar_id: str = "scholar_debug_01",
    sentence_ids: tuple[str, ...] = DIALOGUE_SENTENCE_IDS,
    lesson_id: str | None = "l_nc2_01",
    skill_code: str = "translation",
    mastery: float = 0.5,
    mastery_score: float | None = None,
    confidence: float = 0.7,
    stability: float = 0.3,
    difficulty: int = 2,
    attempt_count: int = 1,
    status: str = "learning",
    states: list[dict] | None = None,
    **overrides,
) -> list[dict]:
    """写入带 mastery/confidence 的 skill_state（RAG 候选 / 指标卡场景）。

    传 `states` 时按原样批量写入（支持逐句差异化指标），否则按 `sentence_ids`
    统一生成。返回写入的文档列表。
    """
    if states is not None:
        docs = [dict(s) for s in states]
        for doc in docs:
            doc.setdefault("scholar_id", scholar_id)
            fake_db.add("skill_state", doc)
        return docs

    score = mastery_score if mastery_score is not None else round(mastery * 100, 2)
    docs = []
    for sid in sentence_ids:
        doc = {
            "scholar_id": scholar_id,
            "sentence_id": sid,
            "lesson_id": lesson_id,
            "skill_code": skill_code,
            "status": status,
            "mastery_score": score,
            "mastery": mastery,
            "confidence": confidence,
            "stability": stability,
            "difficulty": difficulty,
            "attempt_count": attempt_count,
            "last_outcome": "correct",
            "stable_streak": 1,
        }
        doc.update(overrides)
        doc["_id"] = f"{scholar_id}_{sid}_{skill_code}"
        doc["state_id"] = doc["_id"]
        fake_db.add("skill_state", doc)
        docs.append(doc)
    return docs


def seed_local_corpus(
    data_dir,
    *,
    lessons: tuple[int, ...] = (1, 2),
    scholar_id: str = "scholar_debug_01",
    learner: dict | None = None,
    write_learner: bool = True,
    **learner_overrides,
) -> dict:
    """生成 §3.1 的本地 JSON 语料（corpus.json + learners/<scholar>.json）。

    语料结构由 `scripts.prepare_nc2_corpus` 生成，保证与生产脚本产物一致；
    默认写一份带指标的最小 learner，供 loader / 接口测试直接使用。

    Returns:
        {data_dir, corpus_path, learner_path, corpus, learner}
    """
    from pathlib import Path as _Path  # 延迟导入避免顶层依赖

    from scripts.prepare_nc2_corpus import build_corpus, lessons_data

    data_dir = _Path(data_dir)
    corpus = build_corpus(lessons_data(list(lessons)))

    from services.learning.local_corpus import (
        compute_weak_skills,
        corpus_path,
        empty_learner,
        learner_path,
        now_ms,
        save_corpus,
        save_learner,
    )

    save_corpus(corpus, corpus_path(data_dir))

    result_learner = None
    if write_learner:
        result_learner = learner or {
            **empty_learner(scholar_id),
            "skill_states": [
                {
                    "scholar_id": scholar_id,
                    "sentence_id": "s_nc2_01_01",
                    "lesson_id": "l_nc2_01",
                    "skill_code": "translation",
                    "status": "learning",
                    "mastery_score": 52.0,
                    "mastery": 0.52,
                    "confidence": 0.66,
                    "stability": 0.2,
                    "difficulty": 2,
                    "attempt_count": 3,
                    "last_outcome": "fail",
                    "stable_streak": 1,
                }
            ],
            "attempts": [],
            "updated_at": now_ms(),
        }
        result_learner.update(learner_overrides)
        result_learner["weak_skills"] = compute_weak_skills(result_learner)
        save_learner(result_learner, data_dir)

    return {
        "data_dir": data_dir,
        "corpus_path": corpus_path(data_dir),
        "learner_path": learner_path(scholar_id, data_dir),
        "corpus": corpus,
        "learner": result_learner,
    }
