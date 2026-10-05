"""语言点造句「多轮」出题面（中文情景句装配 / 校验 / 去重 / 重出）

契约：api-contract.md §3.18（E7 / E8「中文情景句规格」）
设计：docs_v1/扩展/第三期-语言点造句多轮-v1.md §3.3（出题规格 + 去重兜底）
账本：docs_v1/扩展/第三期-语言点造句多轮-任务拆分与断点-v1.md（B05）

职责（**出题 + 判分决策**，落库与编排在 B08）：
- `build_round_prompt_messages`：按上下文装配出题 messages（目标点 / 原句译文 /
  上一轮 `user_input` + `errors` / 已用情景 / 语域 / 难度），system prompt 取自 B04；
- `validate_prompt_item`：`banned_check` —— 中文情景句里**不得出现目标点英文原文**
  （契约硬约束②，命中即判本轮失败，R13：不消耗轮次）；
- `is_repeat_prompt`：与 `used_prompts[]` 比对是否重复（指纹相等 / 文本相似 /
  同场景且较相似）；
- `generate_round_prompt`：出题主流程（解析 → banned → 去重 → **重出 1 次** →
  仍重复则打 `repeat_risk=True` **放行不阻断**）；
- `judge_round_turn`（B06）：复用既有 L2 rubric（`services/english/extension.py`），
  多轮时**额外**带中文情景与 `reference_en` 作语义锚点；`passed` **由后端按
  `total >= PASS_SCORE 且 must_use_hit` 判定**，不信 LLM 自述；
- `decide_next_action`（B06）：达标 / 未达标且 k<N / k=N / LLM 失败 四分支；
- `build_summary`（B06）：会话小结 `{ turns_used, passed, best_score, top_errors[],
  model_sentences[] }`。

边界：本模块**不写库**（`used_prompts[]` 由调用方传入、新一轮由 B08 落库），
不写 mastery（R10 / R11）。
"""
from __future__ import annotations

import difflib
import logging
import re

import config
from services.english.extension import _call_l2_rubric_llm, grade_l2_rubric
from services.learning.extension_round import build_prompt_fingerprint
from services.providers.extension_llm import STAGE_PARSE, ExtensionError
from services.providers.extension_round_llm import (
    PROMPT_VERSION,
    ROUND_SYSTEM_PROMPT,
    call_round_prompt_llm,
    parse_round_prompt,
)

logger = logging.getLogger("scholar-admin.english.extension_round")

# 语域偏好（E7 入参 register；auto = 交给模型按情景定）
REGISTER_AUTO = "auto"
REGISTERS = (REGISTER_AUTO, "neutral", "formal", "informal")

# 难度（后续轮：same = 换场景保持语域；harder = 换场景并升语域）
DIFFICULTY_SAME = "same"
DIFFICULTY_HARDER = "harder"

# 去重阈值（纯函数、可单测）：文本相似度 ≥ 高阈值 → 重复；
# 场景标签完全相同且相似度 ≥ 低阈值 → 判重复（设计：场景 / 人物 / 动作至少换一项）
REPEAT_SIMILARITY_THRESHOLD = 0.7
REPEAT_SCENE_SIMILARITY_THRESHOLD = 0.5

# 重出时追加的强化指令（去重兜底第 1 次重出用）
_RETRY_INSTRUCTION = (
    "【重出要求】上一版情景与已用情景重复。必须**换一个场景**（人物 / 场所 / 动作至少换一项），"
    "不得沿用上一版的场景标签，也不要只换同义词。"
)

_WS_RE = re.compile(r"\s+")

# 业务码（api-contract §3.18）
ERR_LLM_PARSE_ERROR = "LLM_PARSE_ERROR"

# 轮次推进动作（设计 §3.4 决策矩阵）
ACTION_FINISH_PASSED = "finish_passed"
ACTION_CONTINUE = "continue"
ACTION_FINISH_MAX = "finish_max"
ACTION_FAILED = "failed"

# 小结里「常错 Top N」条数
TOP_ERRORS_LIMIT = 3


def _normalize_term(text: str) -> str:
    """词表比对用归一化：小写 + 弯撇号归一 + 压缩空白（同 extension._normalize_term）。"""
    return " ".join(str(text or "").strip().lower().replace("’", "'").split())


def _split_fingerprint(entry: str) -> tuple[str, list[str]]:
    """拆分 `used_prompts[]` 条目（指纹 `文本#tag1,tag2` 或原始 `prompt_zh`）。"""
    raw = str(entry or "")
    if "#" in raw:
        text, _, tags = raw.partition("#")
        return text, [t for t in tags.split(",") if t]
    return raw, []


def _word_hit(term: str, text_zh: str) -> bool:
    """目标点英文原文是否出现在中文情景句里（banned_check 的单项判定）。

    - 多词短语：子串命中即算（如 `take up`）；
    - 单词：按词边界命中（避免 `art` 命中 `start` 这类误伤）。
    """
    term = _normalize_term(term)
    if not term:
        return False
    hay = _WS_RE.sub(" ", str(text_zh or "")).lower()
    if " " in term:
        return term in hay
    return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", hay) is not None


def must_use_texts(points: list[dict]) -> list[str]:
    """本轮必须使用的语言点文本（**只取 `origin != 'mine'`**，同 E2 口径）。"""
    return [
        str(p.get("text") or "").strip()
        for p in (points or [])
        if str(p.get("text") or "").strip() and p.get("origin", "ai") != "mine"
    ]


# ===========================================================================
# 出题 messages 装配
# ===========================================================================


def build_round_prompt_messages(
    *,
    points: list[dict],
    original: str = "",
    translation: str = "",
    used_prompts: list[str] | None = None,
    prev_turn: dict | None = None,
    difficulty: str = DIFFICULTY_SAME,
    register: str = REGISTER_AUTO,
    turn_index: int = 1,
    extra_instruction: str = "",
) -> list[dict]:
    """装配出题 messages（system 取自 B04 的 `ROUND_SYSTEM_PROMPT`）。

    Args:
        points: 目标语言点（`points_snapshot` 中的勾选项；只用
            `text / type / meaning_zh / register / cefr / collocations`）。
        used_prompts: 已用情景指纹列表（`extension_round.used_prompts[]`）。
        prev_turn: 上一轮 `{ user_input, errors[] }`；第 1 轮传 None。
        difficulty: `same` / `harder`，只影响第 2 轮及以后。
        register: `auto` / `neutral` / `formal` / `informal`。
        extra_instruction: 重出时追加的强化指令（去重兜底）。
    """
    targets = [
        p for p in (points or []) if p.get("origin", "ai") != "mine"
    ] or list(points or [])

    lines: list[str] = []
    lines.append("## 目标语言点（必须使用）")
    for p in targets:
        collocations = p.get("collocations") or []
        coll = "、".join(str(c) for c in collocations[:3]) if collocations else ""
        lines.append(
            f"- {p.get('text', '')}（{p.get('type', '')}）"
            f" 释义：{p.get('meaning_zh', '')}"
            f"｜语域：{p.get('register', '') or 'neutral'}"
            f"｜CEFR：{p.get('cefr', '') or '—'}"
            + (f"｜常见搭配：{coll}" if coll else "")
        )

    if original or translation:
        lines.append("")
        lines.append("## 教材原句（第 1 轮的语境来源）")
        if original:
            lines.append(f"英文：{original}")
        if translation:
            lines.append(f"中文：{translation}")

    lines.append("")
    lines.append(f"## 当前是第 {max(1, int(turn_index))} 轮（difficulty={difficulty}）")
    if turn_index <= 1:
        lines.append("- 第 1 轮：情景贴近上面的教材原句语境；`focus_hint` 恒为空串。")
    elif difficulty == DIFFICULTY_HARDER:
        lines.append("- 换一个场景，并**升一级语域**（neutral → formal 或 informal，视情景而定）。")
    else:
        lines.append("- 换一个场景，语域保持不变。")

    prev = prev_turn or {}
    if prev.get("user_input"):
        lines.append("")
        lines.append("## 上一轮作答与错误")
        lines.append(f"作答：{prev.get('user_input')}")
        errors = prev.get("errors") or []
        if errors:
            lines.append("错误（focus_hint 必须针对第 1 条）：")
            for e in errors[:5]:
                lines.append(f"  - {e}")
        else:
            lines.append("错误：无（`focus_hint` 填空串）")

    used = [u for u in (used_prompts or []) if str(u or "").strip()]
    lines.append("")
    lines.append("## 已用情景（**不得重复**；场景 / 人物 / 动作至少换一项）")
    if used:
        for u in used[-6:]:
            text, tags = _split_fingerprint(u)
            lines.append(f"- {text}" + (f"（场景标签：{','.join(tags)}）" if tags else ""))
    else:
        lines.append("- （无，这是第 1 轮）")

    if register and register != REGISTER_AUTO:
        lines.append("")
        lines.append(f"## 语域偏好：{register}（`register` 字段按此填写）")

    if extra_instruction:
        lines.append("")
        lines.append(extra_instruction)

    lines.append("")
    lines.append("请输出 JSON（只含 prompt_zh / register / focus_hint / scene_tag / reference_en）。")

    return [
        {"role": "system", "content": ROUND_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


# ===========================================================================
# 出题结果校验（banned_check）
# ===========================================================================


def validate_prompt_item(item: dict, must_use: list[str] | None = None) -> dict:
    """出题结果校验：`banned_check` —— 中文句里不得出现目标点英文原文（契约硬约束②）。

    Returns:
        `{ ok, hits[], reason, reference_en_hit }`：
        - `ok=False` → 命中禁用原文（调用方应判本轮失败，**不消耗轮次**，R13）；
        - `reference_en_hit`：**观测位** —— `reference_en` 是否**原样包含**全部目标点文本
          （设计 §3.3 要求参考句包含目标点）。参考句用屈折形式（如 `take up` → `took up`）
          时会判 False，**不判失败**，只供实验页观测；示范句仍优先取 `reference_en`。
    """
    prompt_zh = str((item or {}).get("prompt_zh") or "")
    terms = [t for t in (must_use or []) if str(t or "").strip()]
    hits = [t for t in terms if _word_hit(t, prompt_zh)]

    reference_en = str((item or {}).get("reference_en") or "").lower()
    reference_hit = bool(reference_en) and all(
        _normalize_term(t) in reference_en for t in terms
    )

    return {
        "ok": not hits,
        "hits": hits,
        "reason": (
            f"中文情景句里出现了目标语言点英文原文：{'、'.join(hits)}" if hits else ""
        ),
        "reference_en_hit": reference_hit,
    }


# ===========================================================================
# 去重（纯函数）
# ===========================================================================


def is_repeat_prompt(
    prompt_zh: str,
    scene_tag: list[str] | None = None,
    used_prompts: list[str] | None = None,
) -> bool:
    """新情景是否与 `used_prompts[]` 重复（**纯函数**，供 B08 与单测直接调用）。

    判据（满足任一即重复）：
    1. 指纹相等（`build_prompt_fingerprint`，文本归一化 + 场景标签排序归一）；
    2. 文本相似度 ≥ `REPEAT_SIMILARITY_THRESHOLD`（0.7）；
    3. 场景标签**完全相同**且相似度 ≥ `REPEAT_SCENE_SIMILARITY_THRESHOLD`（0.5）
       —— 对应设计「场景 / 人物 / 动作至少换一项」。
    """
    used = [u for u in (used_prompts or []) if str(u or "").strip()]
    if not used:
        return False
    fingerprint = build_prompt_fingerprint(prompt_zh, scene_tag)
    if fingerprint in used:
        return True

    new_text = _WS_RE.sub("", str(prompt_zh or "")).lower()
    if not new_text:
        return False
    new_tags = sorted(t.strip().lower() for t in (scene_tag or []) if str(t or "").strip())

    for entry in used:
        used_text, used_tags = _split_fingerprint(entry)
        used_text = _WS_RE.sub("", used_text).lower()
        if not used_text:
            continue
        ratio = difflib.SequenceMatcher(None, new_text, used_text).ratio()
        if ratio >= REPEAT_SIMILARITY_THRESHOLD:
            return True
        if new_tags and new_tags == sorted(used_tags) and ratio >= REPEAT_SCENE_SIMILARITY_THRESHOLD:
            return True
    return False


# ===========================================================================
# 出题主流程（解析 → banned → 去重 → 重出 1 次）
# ===========================================================================


async def generate_round_prompt(
    *,
    points: list[dict],
    original: str = "",
    translation: str = "",
    used_prompts: list[str] | None = None,
    prev_turn: dict | None = None,
    difficulty: str = DIFFICULTY_SAME,
    register: str = REGISTER_AUTO,
    turn_index: int = 1,
    max_retry: int | None = None,
) -> dict:
    """生成一轮中文情景句（**不写库**），返回可直接落 `turns[]` 的字段 + 观测位。

    流程：调用 LLM → `parse_round_prompt` 强校验 → `banned_check`
    （命中 → 本轮失败，抛错，**不消耗轮次**）→ `is_repeat_prompt`
    （重复 → 追加「必须换场景」**重出 1 次**，`EXTENSION_ROUND_REPEAT_MAX_RETRY`）
    → 仍重复 → `repeat_risk=True` **放行不阻断**（契约：实验页可观测）。

    Returns:
        `{ prompt_zh, register, focus_hint, scene_tag, reference_en,
           must_use, banned_check, repeat_risk, prompt_version, attempts }`

    Raises:
        ExtensionError: LLM_TIMEOUT / PROVIDER_UNAVAILABLE（LLM 通道）；
        LLM_PARSE_ERROR（解析失败或 `banned_check` 命中）。
    """
    retry = (
        config.EXTENSION_ROUND_REPEAT_MAX_RETRY if max_retry is None else max(0, int(max_retry))
    )
    used = list(used_prompts or [])
    must_use = must_use_texts(points)

    attempts = 0
    while True:
        messages = build_round_prompt_messages(
            points=points,
            original=original,
            translation=translation,
            used_prompts=used,
            prev_turn=prev_turn,
            difficulty=difficulty,
            register=register,
            turn_index=turn_index,
            extra_instruction=_RETRY_INSTRUCTION if attempts > 0 else "",
        )
        content = await call_round_prompt_llm(messages)
        item = parse_round_prompt(content)
        attempts += 1

        banned = validate_prompt_item(item, must_use)
        if not banned["ok"]:
            # 契约硬约束②：命中即判本轮失败（R13：turn_index 不前进）
            logger.warning(
                f"[extension_round] banned_check 命中 → {banned['reason']}"
            )
            raise ExtensionError(
                ERR_LLM_PARSE_ERROR,
                STAGE_PARSE,
                f"出题未通过 banned_check：{banned['reason']}",
            )

        repeat = is_repeat_prompt(item["prompt_zh"], item["scene_tag"], used)
        if repeat and attempts <= retry:
            logger.info(
                f"[extension_round] 情景重复 → 重出第 {attempts} 次"
                f"（turn_index={turn_index}）"
            )
            continue

        return {
            "prompt_zh": item["prompt_zh"],
            "register": item["register"],
            "focus_hint": item["focus_hint"],
            "scene_tag": item["scene_tag"],
            "reference_en": item["reference_en"],
            "must_use": must_use,
            "banned_check": banned,
            "repeat_risk": bool(repeat),
            "prompt_version": PROMPT_VERSION,
            "attempts": attempts,
        }


# ===========================================================================
# 判分（B06）：复用既有 L2 rubric，多轮时补「中文情景 + 参考句」语义锚点
# ===========================================================================


async def judge_round_turn(
    *,
    points: list[dict],
    user_input: str,
    prompt_zh: str = "",
    reference_en: str = "",
) -> dict:
    """判一轮作答（**不写库**），返回形态同 E2 的 `l2_sentence` 结果。

    与 E2 的两点差异（设计 §3.4 + 决策 D5）：
    1. `semantics` 维有了锚点 —— 把 `prompt_zh` 与 `reference_en` 一起喂 rubric
       （`_call_l2_rubric_llm` 的 `round_context`，**E2 不传 → 走原分支**）；
    2. `passed` **由后端判定**：`total >= EXTENSION_ROUND_PASS_SCORE` **且**
       `must_use_hit`（后者由 `grade_l2_rubric` 独立判定，**不信 LLM 自述**）。

    `model_sentence`：LLM 给了示范句就用它；没给时用 `reference_en` 兜底（D5）。
    """
    must_info = grade_l2_rubric(points, user_input)
    round_context = None
    if prompt_zh or reference_en:
        round_context = {"prompt_zh": prompt_zh, "reference_en": reference_en}

    rubric = await _call_l2_rubric_llm(points, user_input, round_context)
    scores = rubric.get("scores", {}) or {}
    total = rubric.get("total", sum(scores.values()))
    try:
        total = int(total)
    except (TypeError, ValueError):
        total = 0

    must_use_hit = bool(must_info.get("must_use_hit"))
    passed = total >= config.EXTENSION_ROUND_PASS_SCORE and must_use_hit

    model_sentence = str(rubric.get("model_sentence") or "").strip()
    if not model_sentence:
        model_sentence = str(reference_en or "").strip()

    return {
        "must_use_hit": must_use_hit,
        "used_points": must_info.get("used_points", []),
        "score": total,
        "passed": passed,
        "rubric_scores": scores,
        "errors": rubric.get("errors", []) or [],
        "suggestion": rubric.get("suggestion", "") or "",
        "model_sentence": model_sentence,
    }


# ===========================================================================
# 推进决策与小结（B06）
# ===========================================================================


def decide_next_action(
    *,
    turn_index: int,
    max_turns: int,
    score: int,
    must_use_hit: bool,
    failed: bool = False,
) -> dict:
    """本轮判完后决定下一步（设计 §3.4 决策矩阵，**纯函数**）。

    | 条件 | `action` | 含义 |
    |---|---|---|
    | LLM 失败 | `failed` | 本轮作废、**`turn_index` 不前进**（R13） |
    | 达标（`total >= PASS_SCORE` 且 `must_use_hit`） | `finish_passed` | 立即结束，**不再出题** |
    | 未达标且 `k < N` | `continue` | 出第 k+1 轮新情景 |
    | 未达标且 `k = N` | `finish_max` | 出小结（可「再来一组」） |

    Returns: `{ action, passed, reason }`
    """
    if failed:
        return {
            "action": ACTION_FAILED,
            "passed": False,
            "reason": "本轮出题或判分失败，不消耗轮次",
        }

    passed = int(score or 0) >= config.EXTENSION_ROUND_PASS_SCORE and bool(must_use_hit)
    if passed:
        return {
            "action": ACTION_FINISH_PASSED,
            "passed": True,
            "reason": f"达标（score={score} ≥ {config.EXTENSION_ROUND_PASS_SCORE} 且 must_use_hit）",
        }

    if int(turn_index or 0) < int(max_turns or 0):
        return {
            "action": ACTION_CONTINUE,
            "passed": False,
            "reason": f"未达标（score={score}），还有轮次（{turn_index}/{max_turns}）",
        }

    return {
        "action": ACTION_FINISH_MAX,
        "passed": False,
        "reason": f"轮次用尽（{turn_index}/{max_turns}）仍未达标",
    }


def build_summary(
    turns: list[dict],
    *,
    passed: bool | None = None,
) -> dict:
    """会话小结（`extension_round.summary`，形态见 data-model §4.28）。

    `{ turns_used, passed, best_score, top_errors[], model_sentences[] }`：
    - `turns_used`：有判分结果的轮数（未判分的作废轮不计）；
    - `best_score`：各轮最高分；
    - `top_errors`：跨轮错误按出现次数取 **Top3**（次数相同按首次出现序）；
    - `model_sentences`：按轮序的示范句（LLM 未给则 `reference_en` 兜底）。

    `passed` 缺省（None）时取**末轮** `result.passed`（B08 亦可按终态显式传入）。
    """
    judged = [t for t in (turns or []) if (t or {}).get("result")]
    scores = [int((t["result"] or {}).get("score") or 0) for t in judged]
    best_score = max(scores) if scores else 0

    counts: dict[str, int] = {}
    order: list[str] = []
    for t in judged:
        for e in (t["result"] or {}).get("errors") or []:
            text = str(e or "").strip()
            if not text:
                continue
            if text not in counts:
                counts[text] = 0
                order.append(text)
            counts[text] += 1
    top_errors = [
        e for e in sorted(order, key=lambda x: (-counts[x], order.index(x)))
    ][:TOP_ERRORS_LIMIT]

    model_sentences = []
    for t in judged:
        result = t["result"] or {}
        sentence = str(result.get("model_sentence") or "").strip()
        if not sentence:
            sentence = str(t.get("reference_en") or "").strip()
        if sentence:
            model_sentences.append(sentence)

    if passed is None:
        passed = bool(judged and (judged[-1]["result"] or {}).get("passed"))

    return {
        "turns_used": len(judged),
        "passed": bool(passed),
        "best_score": best_score,
        "top_errors": top_errors,
        "model_sentences": model_sentences,
    }
