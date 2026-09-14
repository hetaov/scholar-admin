"""本地 NC2 语料加载器（T2 / 设计稿 §3.1、§3.2、§6.3）。

职责：把「文件化的教材层级 + 已学语句 + 模拟指标」读成与线上集合同构的 dict，
供**脚本 / 接口 / 测试**三方共用（避免各自解析 JSON 而漂移）。

目录规范（§3.1）::

    data/nc2/                          # 本地语料根（DIALOGUE_CORPUS_DIR）
    ├── corpus.json                    # 教材层级：book/chapter/lesson/sentence/group
    ├── learners/<scholar_id>.json     # 模拟学者：skill_states + attempts
    └── generated/                     # 生成产物落盘（对话 + 指标），人工抽检

零侵入（§9-7）：本模块只读写本地 JSON 文件，**不触真实库、不触网**；
输出结构与 `textbook_v2 / chapter / lesson / sentence_v2 / skill_state / learning_attempt`
同构，保证「本地能跑、线上字段也对」。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("scholar-admin.local_corpus")

# ---------------------------------------------------------------------------
# 常量与路径
# ---------------------------------------------------------------------------

BOOK_ID = "tb_nc2"
CHAPTER_ID = "ch_nc2_1"
CHAPTER_TITLE = "Lesson 1-20"

CORPUS_FILENAME = "corpus.json"
LEARNERS_DIRNAME = "learners"
GENERATED_DIRNAME = "generated"

DEFAULT_SCHOLAR_ID = "scholar_debug_01"

# 弱项判定线（对齐 §5.4 mastery_threshold / learned_threshold 口径）
WEAK_MASTERY_THRESHOLD = 0.6


def project_root() -> Path:
    """项目根目录（scholar-admin/）：services/learning/local_corpus.py → parents[2]。"""
    return Path(__file__).resolve().parents[2]


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """解析语料根目录；相对路径按项目根解析（不受进程 cwd 影响）。"""
    if data_dir is None:
        from config import DIALOGUE_CORPUS_DIR

        data_dir = DIALOGUE_CORPUS_DIR
    path = Path(data_dir)
    return path if path.is_absolute() else project_root() / path


def corpus_path(data_dir: str | Path | None = None) -> Path:
    """`corpus.json` 路径。"""
    return resolve_data_dir(data_dir) / CORPUS_FILENAME


def learner_path(scholar_id: str, data_dir: str | Path | None = None) -> Path:
    """`learners/<scholar_id>.json` 路径。"""
    return resolve_data_dir(data_dir) / LEARNERS_DIRNAME / f"{scholar_id}.json"


def generated_dir(data_dir: str | Path | None = None) -> Path:
    """生成产物落盘目录 `generated/`（不存在则创建）。"""
    path = resolve_data_dir(data_dir) / GENERATED_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 层级 id 规范（脚本 / 测试共用，避免 id 命名散落）
# ---------------------------------------------------------------------------


def build_ids(lesson_no: int) -> dict:
    """按课号生成层级 id（§3.1 命名：`l_nc2_01` / `s_nc2_01_01` / `g_nc2_01_a`）。"""
    no = int(lesson_no)
    return {
        "textbook_id": BOOK_ID,
        "chapter_id": CHAPTER_ID,
        "lesson_id": f"l_nc2_{no:02d}",
        "group_id": f"g_nc2_{no:02d}_a",
        "sentence_id": lambda idx: f"s_nc2_{no:02d}_{int(idx):02d}",
    }


# ---------------------------------------------------------------------------
# corpus.json 读写
# ---------------------------------------------------------------------------


def empty_corpus() -> dict:
    """空语料骨架（结构合法但无内容；缺失语料时的降级返回）。"""
    return {
        "book": {"textbook_id": BOOK_ID, "title": "新概念英语 2", "level": "B1"},
        "chapters": [],
        "lessons": [],
        "sentences": [],
        "groups": [],
    }


def validate_corpus(corpus: Any) -> None:
    """最小结构校验；不合法抛 ValueError（调用方按需降级）。"""
    if not isinstance(corpus, dict):
        raise ValueError("corpus 必须是 object")
    for key in ("book", "chapters", "lessons", "sentences", "groups"):
        if key not in corpus:
            raise ValueError(f"corpus 缺少字段：{key}")
    if not isinstance(corpus["sentences"], list):
        raise ValueError("corpus.sentences 必须是 array")
    if not isinstance(corpus["groups"], list):
        raise ValueError("corpus.groups 必须是 array")


def load_corpus(path: str | Path | None = None) -> dict:
    """读取 corpus.json（path 缺省取 DIALOGUE_CORPUS_DIR/corpus.json）。

    Raises:
        FileNotFoundError: 语料文件不存在（脚本场景需显式失败）。
        ValueError: 结构不合法。
    """
    target = Path(path) if path else corpus_path()
    if not target.exists():
        raise FileNotFoundError(f"本地语料不存在：{target}（先跑 scripts/prepare_nc2_corpus.py）")
    with target.open("r", encoding="utf-8") as fh:
        corpus = json.load(fh)
    validate_corpus(corpus)
    return corpus


def try_load_corpus(path: str | Path | None = None) -> dict:
    """容错读取：缺失/损坏 → 空骨架（接口热路径降级，不阻断页面）。"""
    try:
        return load_corpus(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("[local_corpus] 语料读取降级（空语料）: %s", exc)
        return empty_corpus()


def save_corpus(corpus: dict, path: str | Path | None = None) -> Path:
    """写回 corpus.json（父目录自动创建）。"""
    validate_corpus(corpus)
    target = Path(path) if path else corpus_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return target


# ---------------------------------------------------------------------------
# 层级索引 / 任务组迭代
# ---------------------------------------------------------------------------


def sentence_index(corpus: dict) -> dict[str, dict]:
    """`{sentence_id: sentence}` 索引（原样引用 corpus 内文档）。"""
    return {
        str(s.get("sentence_id")): s
        for s in corpus.get("sentences") or []
        if s.get("sentence_id")
    }


def list_lessons(corpus: dict) -> list[dict]:
    """按 order 升序列出课本课（附 sentence_count），供页面/脚本展示。"""
    counts: dict[str, int] = {}
    for s in corpus.get("sentences") or []:
        lid = str(s.get("lesson_id") or "")
        if lid:
            counts[lid] = counts.get(lid, 0) + 1
    lessons = []
    for lesson in corpus.get("lessons") or []:
        lid = str(lesson.get("lesson_id") or "")
        lessons.append(
            {
                "lesson_id": lid,
                "title": lesson.get("title"),
                "order": lesson.get("order"),
                "sentence_count": counts.get(lid, 0),
            }
        )
    lessons.sort(key=lambda item: (item.get("order") or 0, item["lesson_id"]))
    return lessons


def lesson_sentence_ids(corpus: dict, lesson_id: str) -> list[str]:
    """某课全部句子 id（按 order 升序）。"""
    rows = [
        s
        for s in corpus.get("sentences") or []
        if str(s.get("lesson_id") or "") == str(lesson_id)
    ]
    rows.sort(key=lambda s: (s.get("order") or 0, str(s.get("sentence_id"))))
    return [str(s.get("sentence_id")) for s in rows]


def iter_lesson_groups(
    corpus: dict, lesson_ids: Optional[Iterable[str]] = None
) -> list[dict]:
    """产出任务组下拉数据（§5 入参 `task_group` 结构，与页面/接口同构）。

    Returns:
        `[{lesson_id, group_id, group_label, sentences: [{sentence_id, content}]}]`
        ——按 lesson.order → group 出现顺序排列；空句子组被跳过。
    """
    allowed = {str(x) for x in lesson_ids} if lesson_ids is not None else None
    order_by_lesson = {
        str(lesson.get("lesson_id")): (lesson.get("order") or 0)
        for lesson in corpus.get("lessons") or []
    }
    index = sentence_index(corpus)

    groups: list[tuple[tuple, dict]] = []
    for group in corpus.get("groups") or []:
        lesson_id = str(group.get("lesson_id") or "")
        if allowed is not None and lesson_id not in allowed:
            continue
        sentences: list[dict] = []
        for sid in group.get("sentence_ids") or []:
            sent = index.get(str(sid))
            if not sent:
                continue
            content = str(sent.get("text") or "").strip()
            if not content:
                continue
            sentences.append({"sentence_id": str(sid), "content": content})
        if not sentences:
            continue
        sort_key = (order_by_lesson.get(lesson_id, 0), str(group.get("group_id") or ""))
        groups.append(
            (
                sort_key,
                {
                    "lesson_id": lesson_id,
                    "group_id": str(group.get("group_id") or ""),
                    "group_label": str(group.get("group_label") or ""),
                    "sentences": sentences,
                },
            )
        )
    groups.sort(key=lambda pair: pair[0])
    return [payload for _, payload in groups]


def build_local_scenario(
    corpus: dict,
    lesson_id: Optional[str] = None,
    *,
    group_id: Optional[str] = None,
) -> dict:
    """按课/任务组生成默认会话背景（页面「任务组选择后自动带出背景」用）。"""
    lesson = next(
        (
            item
            for item in corpus.get("lessons") or []
            if str(item.get("lesson_id") or "") == str(lesson_id or "")
        ),
        None,
    )
    title = str((lesson or {}).get("title") or "").strip()
    label = ""
    for group in corpus.get("groups") or []:
        if group_id and str(group.get("group_id") or "") != str(group_id):
            continue
        if not group_id and str(group.get("lesson_id") or "") != str(lesson_id or ""):
            continue
        label = str(group.get("group_label") or "").strip()
        break
    scene = f"围绕「{title}」的日常话题" if title else "日常英语交流场景"
    return {
        "background": f"{scene}，两位/多位角色自然交谈，把本组句子用进对话。",
        "goal": f"自然用出「{label}」中的句子" if label else "自然用出任务组句子",
    }


# ---------------------------------------------------------------------------
# learners/<scholar_id>.json 读写（模拟学者：已学语句 + 指标）
# ---------------------------------------------------------------------------


def empty_learner(scholar_id: str = DEFAULT_SCHOLAR_ID) -> dict:
    """空模拟学者文档（结构对齐 §3.1 learners 样例）。"""
    return {
        "scholar_id": scholar_id,
        "skill_states": [],
        "attempts": [],
        "weak_skills": [],
        "updated_at": None,
    }


def load_learner(
    scholar_id: str = DEFAULT_SCHOLAR_ID, data_dir: str | Path | None = None
) -> dict:
    """读取模拟学者文件；不存在返回空文档（不抛错，首跑即可用）。"""
    target = learner_path(scholar_id, data_dir)
    if not target.exists():
        return empty_learner(scholar_id)
    with target.open("r", encoding="utf-8") as fh:
        learner = json.load(fh)
    if not isinstance(learner, dict):
        return empty_learner(scholar_id)
    learner.setdefault("scholar_id", scholar_id)
    learner.setdefault("skill_states", [])
    learner.setdefault("attempts", [])
    learner.setdefault("weak_skills", [])
    return learner


def save_learner(
    learner: dict,
    data_dir: str | Path | None = None,
) -> Path:
    """写回模拟学者文件（父目录自动创建）。"""
    scholar_id = str(learner.get("scholar_id") or DEFAULT_SCHOLAR_ID)
    target = learner_path(scholar_id, data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(learner, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return target


def compute_weak_skills(learner: dict, *, limit: int = 5) -> list[str]:
    """由模拟指标派生弱项（mastery < 0.6 的句子所属 lesson_id，按掌握度升序）。

    线上 `metrics.weak_skills` 为「能力/语法弱项」；本地无标签语料，用
    `lesson_id` 作为可解释的弱项标识（页面/生成只做难度约束，不写进台词）。
    """
    states = [s for s in learner.get("skill_states") or [] if isinstance(s, dict)]
    weak = [
        s
        for s in states
        if float(s.get("mastery") or 0.0) < WEAK_MASTERY_THRESHOLD
    ]
    weak.sort(key=lambda s: float(s.get("mastery") or 0.0))
    result: list[str] = []
    for state in weak:
        code = str(state.get("lesson_id") or state.get("sentence_id") or "").strip()
        if code and code not in result:
            result.append(code)
    return result[:limit]


def learner_summary(learner: dict) -> dict:
    """模拟学者指标摘要（页面指标卡/表单预填用，与线上 metrics 口径一致）。"""
    states = [s for s in learner.get("skill_states") or [] if isinstance(s, dict)]
    confidences = [float(s.get("confidence") or 0.0) for s in states if s.get("confidence") is not None]
    masteries = [float(s.get("mastery") or 0.0) for s in states if s.get("mastery") is not None]
    return {
        "scholar_id": learner.get("scholar_id"),
        "skill_state_count": len(states),
        "attempt_count": len(learner.get("attempts") or []),
        "avg_mastery": round(sum(masteries) / len(masteries), 4) if masteries else 0.0,
        "avg_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        ),
        "weak_skills": list(learner.get("weak_skills") or compute_weak_skills(learner)),
        "updated_at": learner.get("updated_at"),
        "available": bool(states),
    }


def now_ms() -> int:
    """当前毫秒时间戳（本地文件时间戳统一口径）。"""
    return int(time.time() * 1000)
