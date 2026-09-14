"""模拟学习脚本：本地语料 → 模拟作答 → 评测指标（设计稿 §3.3 / §8.4-2）

行为：对随机/指定句子模拟学习者作答 → 调 `evaluate_text`（默认注入 **L1 规则**，
**不触网**）→ 累计 `skill_state`（attempt_count / mastery / confidence / stability）→
写回 `data/nc2/learners/<scholar>.json`（**不落真实 evaluation 证据、不写真实库**）。

约束（对齐既有评测域口径 §9-2）：
- `confidence < EVAL_CONFIDENCE_THRESHOLD(0.6)` 不回写（低置信不沉淀为能力证据）；
- 证据不可改、评价可重算：本地脚本只做近似模拟，attempts 仅作本地留痕。

用法（项目根目录）::

    python scripts/simulate_learning.py --scholar scholar_debug_01 --n 40
    python scripts/simulate_learning.py --mode conversation --n 60 --seed 7
    python scripts/simulate_learning.py --judge          # 用真实 Judge（触网，默认不开）
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import EVAL_CONFIDENCE_THRESHOLD, MIN_EVIDENCE  # noqa: E402
from services.learning.local_corpus import (  # noqa: E402
    DEFAULT_SCHOLAR_ID,
    compute_weak_skills,
    corpus_path,
    empty_learner,
    load_corpus,
    load_learner,
    now_ms,
    resolve_data_dir,
    save_learner,
    sentence_index,
)
from services.evaluation_engine import l1_rule_evaluate  # noqa: E402
from services.models_learning import (  # noqa: E402
    DEFAULT_SKILL_CODE,
    compute_next_review_at,
    skill_state_id,
    to_mastery_score,
    update_confidence,
    update_difficulty,
    update_stability,
)

logger = logging.getLogger("scholar-admin.simulate_learning")

# 作答形态概率（study 更贴近原句；conversation 更口语化/更易残缺）
_INPUT_PROFILES = {
    "study": {"exact": 0.40, "partial": 0.35, "weak": 0.25},
    "conversation": {"exact": 0.25, "partial": 0.40, "weak": 0.35},
}

# 明显答非所问的占位作答（触发 L1「不相关」档 → 低置信 → 不回写）
_WEAK_INPUTS = (
    "I am not sure about this.",
    "Sorry, I do not know.",
    "I forgot how to say it.",
)

MASTERED_THRESHOLD_SCORE = 80.0
PASS_THRESHOLD_SCORE = 60.0


def clamp(value: float, low: float, high: float) -> float:
    """数值裁剪（与 models/learning.clamp 同口径）。"""
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# 模拟作答
# ---------------------------------------------------------------------------


def simulate_input(text: str, rng: random.Random, mode: str = "study") -> str:
    """按 mode 概率生成一次模拟作答（exact / partial / weak 三档）。

    - exact：完全复现（L1 → 100 分）；
    - partial：随机截取一段（保留 45%~80% 词，制造 40~90 分区间）；
    - weak：明显答非所问（L1 → 低置信，触发「不回写」门控，§8.4-2）。
    """
    profile = _INPUT_PROFILES.get(mode, _INPUT_PROFILES["study"])
    roll = rng.random()
    words = str(text or "").split()
    if roll < profile["exact"]:
        return str(text or "").strip()
    if roll < profile["exact"] + profile["partial"] and len(words) >= 3:
        keep = max(2, min(len(words) - 1, int(round(len(words) * rng.uniform(0.45, 0.8)))))
        start = 0 if rng.random() < 0.5 else rng.randint(0, max(0, len(words) - keep))
        return " ".join(words[start : start + keep])
    return rng.choice(_WEAK_INPUTS)


def pick_sentence(
    candidates: list[dict], states: dict[str, dict], rng: random.Random
) -> dict:
    """加权抽取下一句：弱项（mastery 低）更高概率被复练（贴近真实学习行为）。"""
    weights: list[float] = []
    for sentence in candidates:
        state = states.get(str(sentence.get("sentence_id")))
        if state is None:
            weights.append(1.0)  # 未练过 → 先覆盖
        else:
            mastery = float(state.get("mastery") or 0.0)
            weights.append(0.5 + 2.5 * (1.0 - mastery))
    return rng.choices(candidates, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# 指标累计（本地近似：复用评测域 confidence/stability/difficulty 口径）
# ---------------------------------------------------------------------------


def accumulate_mastery(
    prev_score: float | None, new_score: float, attempt_count: int, *, weight: float = 1.0
) -> float:
    """mastery_score 累计（对齐 §5.6.2 证据稀疏保护口径，本地近似）。

    - 首次：直接以本次分初始化（避免冷启动被 1/MIN_EVIDENCE 拉低到失真）；
    - 之后：增量 = (新分 - 旧分) × min(1, attempt/MIN_EVIDENCE) × weight。
    """
    new_score = clamp(float(new_score), 0.0, 100.0)
    if prev_score is None:
        return round(new_score, 2)
    attempt = max(int(attempt_count), 1)
    evidence = min(1.0, attempt / MIN_EVIDENCE) if MIN_EVIDENCE > 0 else 1.0
    delta = (new_score - float(prev_score)) * evidence * float(weight)
    return round(clamp(float(prev_score) + delta, 0.0, 100.0), 2)


def build_or_update_state(
    prev: dict | None,
    *,
    scholar_id: str,
    sentence: dict,
    verdict: dict,
    now_seconds: int,
    last_studied_ms: int,
) -> dict:
    """把一次评估结果累计进 skill_state（字段对齐线上 + §3.1 样例）。"""
    mastery_score = to_mastery_score(verdict.get("score"), None)
    mastery_value = accumulate_mastery(
        (prev or {}).get("mastery_score"),
        mastery_score or 0.0,
        int((prev or {}).get("attempt_count") or 0) + 1,
    )
    attempt_count = int((prev or {}).get("attempt_count") or 0) + 1
    outcome = "correct" if (mastery_score or 0) >= PASS_THRESHOLD_SCORE else "fail"
    # 上次方向按线上口径（success/fail）参与稳定性比较；本地存储为 correct/fail（§3.1 样例）
    prev_outcome = (prev or {}).get("last_outcome")
    prev_direction = None
    if prev_outcome:
        prev_direction = "success" if prev_outcome == "correct" else "fail"
    stability, last_outcome, streak = update_stability(
        (prev or {}).get("stability"),
        "success" if outcome == "correct" else "fail",
        prev_direction,
        int((prev or {}).get("stable_streak") or 0),
    )
    confidence = update_confidence(
        (prev or {}).get("confidence"), float(verdict.get("confidence") or 0.0), attempt_count
    )
    difficulty = update_difficulty(
        (prev or {}).get("difficulty"), sentence.get("difficulty")
    )
    sentence_id = str(sentence.get("sentence_id"))
    state_id = skill_state_id(scholar_id, sentence_id, DEFAULT_SKILL_CODE)
    return {
        "_id": state_id,
        "state_id": state_id,
        "scholar_id": scholar_id,
        "sentence_id": sentence_id,
        "lesson_id": sentence.get("lesson_id"),
        "skill_code": DEFAULT_SKILL_CODE,
        "status": "mastered" if mastery_value >= MASTERED_THRESHOLD_SCORE else "learning",
        "mastery_score": mastery_value,
        "mastery": round(mastery_value / 100.0, 4),
        "attempt_count": attempt_count,
        "confidence": confidence,
        "stability": stability,
        "difficulty": difficulty,
        "last_outcome": outcome,
        "stable_streak": streak,
        "last_studied_at": now_seconds,
        "next_review_at": compute_next_review_at(
            now_seconds, attempt_count, mastery_value
        ),
        "created_at": int((prev or {}).get("created_at") or last_studied_ms),
        "updated_at": last_studied_ms,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_simulation(
    *,
    corpus: dict,
    learner: dict,
    scholar_id: str,
    n: int,
    mode: str = "study",
    seed: int = 42,
    lesson_ids: list[str] | None = None,
    evaluator=l1_rule_evaluate,
) -> dict:
    """执行 n 次模拟学习，返回更新后的 learner（就地更新 skills/attempts）。"""
    rng = random.Random(seed)
    index = sentence_index(corpus)
    allowed = {str(x) for x in lesson_ids} if lesson_ids else None
    candidates = [
        s
        for s in index.values()
        if str(s.get("text") or "").strip()
        and (allowed is None or str(s.get("lesson_id")) in allowed)
    ]
    if not candidates:
        raise ValueError("语料中没有可模拟的句子（检查 --lessons / corpus.json）")

    states: dict[str, dict] = {
        str(s.get("sentence_id")): s for s in learner.get("skill_states") or []
    }
    attempts: list[dict] = list(learner.get("attempts") or [])
    written = 0
    skipped_low_confidence = 0
    skipped_not_meaningful = 0

    for _ in range(max(0, int(n))):
        sentence = pick_sentence(candidates, states, rng)
        original = str(sentence.get("text") or "").strip()
        user_input = simulate_input(original, rng, mode=mode)
        verdict = evaluator(original, user_input)
        ts_ms = now_ms()
        ts_s = ts_ms // 1000

        attempts.append(
            {
                "scholar_id": scholar_id,
                "sentence_id": sentence.get("sentence_id"),
                "lesson_id": sentence.get("lesson_id"),
                "mode": mode,
                "original_text": original,
                "user_input": user_input,
                "score": verdict.get("score"),
                "meaningful": verdict.get("meaningful"),
                "faithfulness": verdict.get("faithfulness"),
                "anomaly": verdict.get("anomaly"),
                "confidence": verdict.get("confidence"),
                "level": verdict.get("level"),
                "source": "local_simulate",
                "created_at": ts_ms,
            }
        )

        # 门控（§9-2）：低置信 / 未达意 不回写 SkillState
        if not verdict.get("meaningful"):
            skipped_not_meaningful += 1
            continue
        if float(verdict.get("confidence") or 0.0) < EVAL_CONFIDENCE_THRESHOLD:
            skipped_low_confidence += 1
            continue

        sid = str(sentence.get("sentence_id"))
        states[sid] = build_or_update_state(
            states.get(sid),
            scholar_id=scholar_id,
            sentence=sentence,
            verdict=verdict,
            now_seconds=ts_s,
            last_studied_ms=ts_ms,
        )
        written += 1

    learner["scholar_id"] = scholar_id
    learner["skill_states"] = list(states.values())
    learner["attempts"] = attempts
    learner["weak_skills"] = compute_weak_skills(learner)
    learner["updated_at"] = now_ms()
    learner["stats"] = {
        "attempted": len(attempts),
        "written_back": written,
        "skipped_low_confidence": skipped_low_confidence,
        "skipped_not_meaningful": skipped_not_meaningful,
        "confidence_threshold": EVAL_CONFIDENCE_THRESHOLD,
        "mode": mode,
    }
    logger.info(
        "[simulate_learning] attempts=%d, 回写=%d, 低置信跳过=%d, 未达意跳过=%d",
        len(attempts),
        written,
        skipped_low_confidence,
        skipped_not_meaningful,
    )
    return learner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地模拟学习 → 评测指标（设计稿 §3.3）")
    parser.add_argument("--corpus", default=None, help="corpus.json 路径（缺省 DIALOGUE_CORPUS_DIR/corpus.json）")
    parser.add_argument("--scholar", default=DEFAULT_SCHOLAR_ID, help="模拟学者 ID")
    parser.add_argument("--mode", default="study", choices=["study", "conversation"])
    parser.add_argument("--n", type=int, default=40, help="模拟作答次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    parser.add_argument("--lessons", default=None, help="仅模拟指定课，如 1-5")
    parser.add_argument("--out", default=None, help="数据目录（缺省 DIALOGUE_CORPUS_DIR，或 --corpus 所在目录）")
    parser.add_argument("--reset", action="store_true", help="先清空该学者的历史模拟数据")
    parser.add_argument("--judge", action="store_true", help="用真实 evaluate_text（触网，默认仅 L1 规则）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    lesson_ids = None
    if args.lessons:
        from scripts.prepare_nc2_corpus import parse_lessons_arg
        from services.learning.local_corpus import build_ids

        lesson_ids = [build_ids(no)["lesson_id"] for no in parse_lessons_arg(args.lessons)]

    data_dir = resolve_data_dir(args.out) if args.out else None
    corpus_file = Path(args.corpus) if args.corpus else corpus_path(data_dir)
    if data_dir is None:
        # 未显式指定时，learner 与 corpus 同目录（保证 --corpus 自洽）
        data_dir = corpus_file.parent
    corpus = load_corpus(corpus_file)
    if args.reset:
        learner = empty_learner(args.scholar)
    else:
        learner = load_learner(args.scholar, data_dir)

    evaluator = None
    if args.judge:
        from services.evaluation_engine import evaluate_text

        evaluator = evaluate_text

    learner = run_simulation(
        corpus=corpus,
        learner=learner,
        scholar_id=args.scholar,
        n=args.n,
        mode=args.mode,
        seed=args.seed,
        lesson_ids=lesson_ids,
        **({"evaluator": evaluator} if evaluator else {}),
    )
    target = save_learner(learner, data_dir)
    print(
        f"✅ 模拟学习完成：{target}\n"
        f"   作答 {learner['stats']['attempted']} / 回写 {learner['stats']['written_back']} / "
        f"低置信跳过 {learner['stats']['skipped_low_confidence']} / 未达意跳过 {learner['stats']['skipped_not_meaningful']}\n"
        f"   弱项：{learner['weak_skills'] or '（无，均已达标）'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
