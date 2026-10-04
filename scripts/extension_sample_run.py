"""O02 样本跑量：跑真实 E1 / E2 / E3 / E4' / E6 链路并产出量化指标。

对应：docs_v1/扩展/第一期-英文语句扩展-任务拆分与断点-v1.md §3.4 O02 / O03
指标口径：docs_v1/扩展/第一期-scholar-admin接口与admin-web实验页-v1.md §6.1 / §6.2

与「实验页手工跑」的差异（本脚本的价值与局限）：
  - 价值：机器可复核、可重跑、指标自动统计；cover 全区 batch。
  - 局限：①「学习者校对」由脚本按显式规则代打（见 `_judge_review`），不是真人判断，
         因此「语言点人工接受率」在本轮口径为「规则初审接受率」，须人工在实验页抽查复核；
       ② UI 结论（小屏呈现 / 交互观感）无法由本脚本产出，仍须人工走查。

用法::

    python scripts/extension_sample_run.py --limit 3              # 先 pilot
    python scripts/extension_sample_run.py --limit 30             # 正式跑量
    python scripts/extension_sample_run.py --limit 30 --skip-eval # 只跑抽取

产物（data/extension_o02_*）:
  - extension_o02_report.json   汇总指标（O03 判据在此）
  - extension_o02_samples.json  逐句明细（含抽取结果、耗时、 crowd ）
  - extension_o02_diff.json     学习者校对 diff 日志（等同实验页「导出 diff JSON」）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from services.dependencies import get_db  # noqa: E402
from services.english.extension import (  # noqa: E402
    resolve_effective_point,
    run_evaluate_pipeline,
    run_extract_pipeline,
)
from services.english.extension_candidates import (  # noqa: E402
    IDIOM_SEED,
    PHRASE_SEED,
    SLANG_SEED,
    STOPWORDS,
)
from services.english.extension_review import (  # noqa: E402
    get_review,
    normalize_point_text,
    save_review,
)
from services.models_content import compute_text_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data"

DEFAULT_BOOKS = ["tb_e6bce2b577554d02", "tb_db3e2209b3cc4e9e"]  # NCE2 / NCE1
DEFAULT_SCHOLAR = "o02-runner"

# 规则初审：这类点即使 AI 抽出也不值得练（超高频功能词 / 极基础词）
_TOO_BASIC = STOPWORDS | {
    "got", "get", "go", "went", "come", "came", "said", "say", "see", "saw",
    "know", "knew", "think", "thought", "want", "like", "make", "made", "take",
    "took", "put", "give", "gave", "tell", "told", "ask", "asked", "look",
    "looked", "seem", "seemed", "become", "became", "find", "found", "one",
    "two", "time", "day", "year", "way", "thing", "man", "men", "people",
}

_QUOTE_RE = re.compile(r"[\"'“”‘’]")


# ---------------------------------------------------------------------------
# 样本选取
# ---------------------------------------------------------------------------


def _buckets(text: str) -> list[str]:
    """给句子打类别标签（§6.1 要求覆盖：对话句 / 长难句 / 俚语习语句 / 短句）。"""
    words = text.split()
    tags: list[str] = []
    n = len(words)
    if n <= 6:
        tags.append("short")
    elif n >= 14 or ("," in text and n >= 10):
        tags.append("long")
    else:
        tags.append("mid")
    if _QUOTE_RE.search(text):
        tags.append("dialogue")
    low = " " + " ".join(words).lower() + " "
    if any(f" {t} " in low for t in SLANG_SEED | IDIOM_SEED):
        tags.append("slang_idiom")
    return tags


async def load_sentences(db, textbook_id: str, limit: int = 600) -> list[dict]:
    rows: list[dict] = []
    for offset in range(0, limit, 200):
        res = await db.query(
            "sentence_v2",
            where={"textbook_id": textbook_id},
            order=[{"field": "order", "direction": "asc"}],
            offset=offset,
            limit=200,
        )
        recs = res.get("records", [])
        rows.extend(recs)
        if len(recs) < 200:
            break
    return rows


def pick_samples(rows: list[dict], n: int, tag_quota: dict[str, int]) -> list[dict]:
    """按类别配额抽句：先均摊到全语料，再按配额取，剩余补齐。

    均摊（每 `step` 句取一）是为了让样本覆盖靠后的课时，而不是扎堆在前几课。
    """
    step = max(1, len(rows) // 400)
    rows = rows[::step]
    picked: list[dict] = []
    used: set[str] = set()
    for tag, quota in tag_quota.items():
        cnt = 0
        for r in rows:
            if cnt >= quota:
                break
            sid = r.get("sentence_id")
            if sid in used or len((r.get("text") or "").split()) < 4:
                continue
            if tag in _buckets(r.get("text", "")):
                picked.append(r)
                used.add(sid)
                cnt += 1
    # 补齐
    for r in rows:
        if len(picked) >= n:
            break
        sid = r.get("sentence_id")
        if sid in used or len((r.get("text") or "").split()) < 4:
            continue
        picked.append(r)
        used.add(sid)
    return [p for p in picked if "[" not in (p.get("text") or "")][:n]


# ---------------------------------------------------------------------------
# 学习者校对：规则初审（代打）
# ---------------------------------------------------------------------------


def _lexicon_misses(original: str, ai_points: list[dict]) -> list[str]:
    """内置词表里命中句子、但 AI 未抽出的固定表达 —— 候选召回缺口的代理指标。"""
    low = original.lower()
    ai_texts = {normalize_point_text(p.get("text", "")) for p in ai_points}
    misses: list[str] = []
    for term in sorted(SLANG_SEED | IDIOM_SEED | PHRASE_SEED, key=len, reverse=True):
        if " " not in term:
            continue  # 单词类不在词表回填范围（word 由 n-gram 兜底）
        if term in low and term not in ai_texts:
            misses.append(term)
    return misses[: config.EXTENSION_REVIEW_MAX_ADDED]


def judge_review(original: str, ai_points: list[dict]) -> tuple[list[str], list[dict]]:
    """规则初审：给出 removed_texts / added（脚本口径，非真人判断）。"""
    removed = [
        p.get("text", "")
        for p in ai_points
        if normalize_point_text(p.get("text", "")) in _TOO_BASIC
    ]
    added = [
        {"type": "phrase", "text": t, "meaning_zh": "", "note": "O02 脚本初审：AI 未抽出的固定表达"}
        for t in _lexicon_misses(original, ai_points)
    ]
    return removed, added


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------


def check_spans(original: str, points: list[dict]) -> list[dict]:
    """R7 复检：`original[span[0]:span[1]] == text`，返回不一致项（应为 0）。"""
    bad = []
    for p in points:
        span = p.get("span") or []
        if len(span) != 2:
            bad.append({"text": p.get("text"), "reason": "span_missing"})
            continue
        if original[span[0] : span[1]] != p.get("text", ""):
            bad.append({"text": p.get("text"), "span": span, "reason": "mismatch"})
    return bad


def check_quota(points: list[dict]) -> list[str]:
    """配额校验（§3.5）：word≤3 / phrase≤2 / slang≤1 / idiom≤1 / 合计≤5。"""
    caps = {"word": 3, "phrase": 2, "slang": 1, "idiom": 1}
    errs: list[str] = []
    counts: dict[str, int] = {}
    for p in points:
        t = p.get("type", "?")
        counts[t] = counts.get(t, 0) + 1
    for t, c in counts.items():
        cap = caps.get(t)
        if cap is not None and c > cap:
            errs.append(f"{t}={c}>{cap}")
    if len(points) > config.EXTENSION_MAX_POINTS_PER_SENTENCE:
        errs.append(f"total={len(points)}>{config.EXTENSION_MAX_POINTS_PER_SENTENCE}")
    return errs


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def run_sentence(db, sent: dict, scholar_id: str, with_review: bool) -> dict:
    """单句：冷抽取 → 缓存复跑 → 规则初审校对 → E6 回读。"""
    original = sent.get("text", "")
    translation = sent.get("translation", "") or ""
    sid = sent.get("sentence_id")
    common = dict(
        sentence_id=sid,
        textbook_id=sent.get("textbook_id"),
        lesson_id=sent.get("lesson_id"),
        original=original,
        translation=translation,
        scholar_id=scholar_id,
        enable_fallback=True,
    )

    t0 = time.perf_counter()
    cold = await run_extract_pipeline(db, force_refresh=True, **common)
    cold_ms = int((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    warm = await run_extract_pipeline(db, force_refresh=False, **common)
    warm_ms = int((time.perf_counter() - t1) * 1000)

    points = cold.get("points", []) or []
    meta = cold.get("meta", {}) or {}
    ch = meta.get("content_hash", "")

    review_summary = None
    removed: list[str] = []
    added: list[dict] = []
    if with_review:
        removed, added = judge_review(original, points)
        await save_review(
            db,
            scholar_id=scholar_id,
            sentence_id=sid,
            content_hash=ch,
            removed_texts=removed,
            added=added,
            note="O02 跑量 · 规则初审",
        )
        got = await get_review(db, scholar_id=scholar_id, sentence_id=sid)
        review_summary = got.get("review")
        review_summary["history_count"] = len(got.get("history", []) or [])

    return {
        "sentence_id": sid,
        "textbook_id": sent.get("textbook_id"),
        "lesson_id": sent.get("lesson_id"),
        "original": original,
        "translation": translation,
        "buckets": _buckets(original),
        "word_count": len(original.split()),
        "cold_ms": cold_ms,
        "cached_ms": warm_ms,
        "cached_hit": bool(warm.get("cached")),
        "source": meta.get("source"),
        "attempts": meta.get("attempts"),
        "fallback_reason": meta.get("fallback_reason") or "",
        "prompt_version": meta.get("prompt_version"),
        "model": meta.get("model"),
        "content_hash": ch,
        "point_count": len(points),
        "count_by_type": meta.get("count_by_type", {}),
        "excluding_count": sum(1 for p in points if p.get("excluding")),
        "risk_count": sum(1 for p in points if (p.get("risk") or "")),
        "l1_missing": sum(1 for p in points if not p.get("l1")),
        "l1_bad_options": sum(
            1 for p in points if p.get("l1") and len((p["l1"].get("options") or [])) != 4
        ),
        "span_violations": check_spans(original, points),
        "quota_violations": check_quota(points),
        "points": [
            {
                "id": p.get("id"),
                "type": p.get("type"),
                "text": p.get("text"),
                "meaning_zh": p.get("meaning_zh", ""),
                "register": p.get("register", ""),
                "cefr": p.get("cefr", ""),
                "confidence": p.get("confidence"),
                "risk": p.get("risk", ""),
                "excluding": bool(p.get("excluding")),
                "span": p.get("span"),
            }
            for p in points
        ],
        "review": review_summary,
        "removed_texts": removed,
        "added_texts": [a["text"] for a in added],
    }


async def run_l1(db, samples: list[dict], runs: int = 10) -> dict:
    """L1 判分：8 次正解 + 2 次错解（均要求确定性），另同答案重复 3 次测抖动。"""
    usable = [s for s in samples if s["points"] and s["points"][0].get("text")]
    if not usable:
        return {"runs": 0, "error": "无可做题面"}
    results = []
    for i in range(min(runs, len(usable))):
        s = usable[i]
        rec = await resolve_effective_point(db, s["sentence_id"])
        if not rec:
            continue
        points = rec.get("points", [])
        target = next((p for p in points if p.get("l1")), None)
        if not target:
            continue
        # 正解 = 该点文本；错解 = 故意改词
        answer = target["text"] if i < runs - 2 else (target["text"] + "zz")
        try:
            r = await run_evaluate_pipeline(
                db,
                task_type="l1_fill",
                selected_ids=[target["id"]],
                points_snapshot=points,
                input_mode="text",
                user_input=answer,
            )
        except Exception as e:  # 单笔失败不中断
            print(f"  [l1] ✗ {s['sentence_id']}: {e}", flush=True)
            continue
        results.append(
            {
                "sentence_id": s["sentence_id"],
                "expected_passed": i < runs - 2,
                "answer": answer,
                "passed": r.get("passed"),
                "score": r.get("score"),
                "answer_key": r.get("answer_key"),
            }
        )
    # 同答案重复三次（确定性）
    if usable:
        s = usable[0]
        rec = await resolve_effective_point(db, s["sentence_id"])
        points = rec.get("points", []) if rec else []
        target = next((p for p in points if p.get("l1")), None)
        repeats = []
        if target:
            for _ in range(3):
                r = await run_evaluate_pipeline(
                    db,
                    task_type="l1_fill",
                    selected_ids=[target["id"]],
                    points_snapshot=points,
                    input_mode="text",
                    user_input=target["text"],
                )
                repeats.append({"score": r.get("score"), "passed": r.get("passed")})
    else:
        repeats = []
    spread = 0
    if repeats:
        scores = {r["score"] for r in repeats}
        spread = max(scores) - min(scores)
    correct = sum(1 for r in results if bool(r["passed"]) == r["expected_passed"])
    return {
        "runs": len(results),
        "correct_rate": correct / len(results) if results else 0,
        "repeat_triple_spread": spread,
        "repeats": repeats,
        "detail": results,
    }


async def run_l2(db, samples: list[dict], runs: int = 10) -> dict:
    """L2 rubric：run-3 同答案重复（占 3 次），其余单次；作答用「原句本身」（必含目标点）。"""
    usable = [s for s in samples if s["points"]]
    if not usable:
        return {"runs": 0, "error": "无可做题面"}
    results = []
    latencies: list[int] = []

    async def _once(s: dict) -> dict:
        rec = await resolve_effective_point(db, s["sentence_id"])
        points = rec.get("points", []) if rec else []
        if not points:
            return {}
        ids = [p["id"] for p in points[:2]]
        t0 = time.perf_counter()
        try:
            r = await run_evaluate_pipeline(
                db,
                task_type="l2_sentence",
                selected_ids=ids,
                points_snapshot=points,
                input_mode="text",
                user_input=s["original"],
            )
        except Exception as e:  # 单笔失败不中断
            print(f"  [l2] ✗ {s['sentence_id']}: {e}", flush=True)
            return {}
        latencies.append(int((time.perf_counter() - t0) * 1000))
        return {
            "sentence_id": s["sentence_id"],
            "selected_ids": ids,
            "score": r.get("score"),
            "passed": r.get("passed"),
            "must_use_hit": r.get("must_use_hit"),
            "rubric_scores": r.get("rubric_scores", {}),
            "errors": r.get("errors", []),
            "model_sentence": r.get("model_sentence", "")[:120],
        }

    singles = max(runs - 3, 0)
    for i in range(singles):
        s = usable[i % len(usable)]
        r = await _once(s)
        if r:
            results.append(r)
    triples: list[dict] = []
    if usable:
        for _ in range(3):
            r = await _once(usable[0])
            if r:
                triples.append(r)
                results.append(r)
    scores = [r.get("score") or 0 for r in results]
    triple_scores = [r.get("score") or 0 for r in triples]
    return {
        "runs": len(results),
        "pass_rate": sum(1 for r in results if r.get("passed")) / len(results) if results else 0,
        "must_use_hit_rate": (
            sum(1 for r in results if r.get("must_use_hit")) / len(results) if results else 0
        ),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "triple_scores": triple_scores,
        "triple_spread": (max(triple_scores) - min(triple_scores)) if triple_scores else None,
        "latency_ms": latencies,
        "error_detail_sample": [r.get("errors") for r in results[:3]],
        "detail": results,
    }


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = int(round((len(ordered) - 1) * p / 100))
    return ordered[k]


def build_report(details: list[dict], l1: dict, l2: dict, elapsed_ms: int) -> dict:
    n = len(details)
    cold = [d["cold_ms"] for d in details]
    warm = [d["cached_ms"] for d in details]
    ai_points = sum(d["point_count"] for d in details)
    removed = sum(len(d.get("removed_texts") or []) for d in details)
    added = sum(len(d.get("added_texts") or []) for d in details)
    type_totals: dict[str, int] = {}
    missing_type = {"word": 0, "phrase": 0, "slang": 0, "idiom": 0}
    for d in details:
        cbt = d.get("count_by_type") or {}
        for t, c in cbt.items():
            type_totals[t] = type_totals.get(t, 0) + c
        for t in missing_type:
            if not cbt.get(t):
                missing_type[t] += 1
    return {
        "generated_at": int(time.time() * 1000),
        "config": {
            "prompt_version": config.EXTENSION_PROMPT_VERSION,
            "model": config.EXTENSION_LLM_MODEL,
            "max_points": config.EXTENSION_MAX_POINTS_PER_SENTENCE,
            "candidate_max": config.EXTENSION_CANDIDATE_MAX,
            "thinking_disabled": config.EXTENSION_THINKING_DISABLED,
            "rule_fallback_enabled": config.EXTENSION_RULE_FALLBACK_ENABLED,
            "langgraph": config.EXTENSION_USE_LANGGRAPH,
        },
        "sample": {
            "sentences": n,
            "by_book": {b: sum(1 for d in details if d["textbook_id"] == b) for b in
                        {d["textbook_id"] for d in details}},
            "by_bucket": {
                b: sum(1 for d in details if b in d["buckets"])
                for b in ("short", "mid", "long", "dialogue", "slang_idiom")
            },
            "avg_word_count": round(statistics.mean([d["word_count"] for d in details]), 1) if n else 0,
        },
        "extract": {
            "cold_ms_p50": pct(cold, 50),
            "cold_ms_p90": pct(cold, 90),
            "cold_ms_max": max(cold) if cold else 0,
            "cached_ms_p50": pct(warm, 50),
            "cached_ms_p90": pct(warm, 90),
            "cache_hit_rate_raw": (sum(1 for d in details if d["cached_hit"]) / n) if n else 0,
            "cache_call_ratio": 0.5,  # 本跑法每句 1 冷 1 缓存，脚本口径固定 50%
            "source_counts": {
                s: sum(1 for d in details if d["source"] == s)
                for s in {d["source"] for d in details}
            },
            "parse_fail_rate": (
                sum(1 for d in details if (d["attempts"] or 1) > 1
                    or d["fallback_reason"] == "LLM_PARSE_ERROR") / n
            ) if n else 0,
            "fallback_reasons": {
                r: sum(1 for d in details if d["fallback_reason"] == r)
                for r in {d["fallback_reason"] for d in details if d["fallback_reason"]}
            },
            "empty_rate": (sum(1 for d in details if d["point_count"] == 0) / n) if n else 0,
            "avg_points": round(ai_points / n, 2) if n else 0,
        },
        "quality": {
            "ai_points_total": ai_points,
            "span_violation_rate": (
                sum(len(d["span_violations"]) for d in details) / ai_points if ai_points else 0
            ),
            "span_violation_total": sum(len(d["span_violations"]) for d in details),
            "quota_violation_sentences": sum(1 for d in details if d["quota_violations"]),
            "type_totals": type_totals,
            "missing_type_sentence_rate": {t: c / n for t, c in missing_type.items()} if n else {},
            "excluding_count": sum(d["excluding_count"] for d in details),
            "risk_count": sum(d["risk_count"] for d in details),
            "l1_missing_points": sum(d["l1_missing"] for d in details),
            "l1_bad_option_points": sum(d["l1_bad_options"] for d in details),
            "review_accept_rate": (1 - removed / ai_points) if ai_points else None,
            "review_removed": removed,
            "review_added": added,
            "recall_gap_rate": (added / (ai_points + added)) if (ai_points + added) else 0,
        },
        "l1": l1,
        "l2": l2,
        "wall_clock_ms": elapsed_ms,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30, help="样本句数")
    ap.add_argument("--books", default=",".join(DEFAULT_BOOKS))
    ap.add_argument("--scholar", default=DEFAULT_SCHOLAR)
    ap.add_argument("--skip-review", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--tag-quota", default="short:6,long:8,dialogue:4,slang_idiom:4")
    args = ap.parse_args()

    tag_quota = {}
    for kv in args.tag_quota.split(","):
        if not kv.strip():
            continue
        k, v = kv.split(":")
        tag_quota[k.strip()] = int(v)

    db = get_db()
    books = [b.strip() for b in args.books.split(",") if b.strip()]
    rows: list[dict] = []
    for b in books:
        got = await load_sentences(db, b)
        print(f"[sample] {b}: 载入 {len(got)} 句", flush=True)
        rows.extend(got)
    samples = pick_samples(rows, args.limit, tag_quota)
    print(f"[sample] 选出 {len(samples)} 句", flush=True)
    for s in samples:
        print(f"   - {s['sentence_id']} [{'+'.join(_buckets(s['text']))}] {s['text'][:70]}", flush=True)

    t_start = time.perf_counter()
    details: list[dict] = []
    for i, s in enumerate(samples, 1):
        try:
            d = await run_sentence(db, s, args.scholar, with_review=not args.skip_review)
        except Exception as e:  # 单句失败不中断整轮
            print(f"[{i}/{len(samples)}] ✗ {s.get('sentence_id')} {e}", flush=True)
            continue
        details.append(d)
        print(
            f"[{i}/{len(samples)}] cold={d['cold_ms']}ms cache={d['cached_ms']}ms "
            f"points={d['point_count']} src={d['source']} try={d['attempts']} "
            f"fb={d['fallback_reason'] or '-'} span_bad={len(d['span_violations'])}",
            flush=True,
        )
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    l1 = await run_l1(db, details) if not args.skip_eval else {"runs": 0, "skipped": True}
    l2 = await run_l2(db, details) if not args.skip_eval else {"runs": 0, "skipped": True}

    report = build_report(details, l1, l2, elapsed_ms)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "extension_o02_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "extension_o02_samples.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    diff = [
        {
            "sentence_id": d["sentence_id"],
            "scholar_id": args.scholar,
            "ai_points": [p["text"] for p in d["points"]],
            "my_removed": d.get("removed_texts", []),
            "my_added": d.get("added_texts", []),
            "stale": (d.get("review") or {}).get("stale"),
            "at": (d.get("review") or {}).get("updated_at"),
        }
        for d in details
    ]
    (OUT_DIR / "extension_o02_diff.json").write_text(
        json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    k = report["extract"]
    q = report["quality"]
    print("\n===== O02 汇总 =====", flush=True)
    print(f"样本 {report['sample']['sentences']} 句，AI 点 {q['ai_points_total']} 个", flush=True)
    print(f"抽取 P50={k['cold_ms_p50']}ms P90={k['cold_ms_p90']}ms max={k['cold_ms_max']}ms", flush=True)
    print(f"缓存 P50={k['cached_ms_p50']}ms P90={k['cached_ms_p90']}ms", flush=True)
    print(f"LLM 失败率 {k['parse_fail_rate']:.1%}  空结果率 {k['empty_rate']:.1%}  source={k['source_counts']}", flush=True)
    print(f"span 违例 {q['span_violation_total']}  配额违例句 {q['quota_violation_sentences']}", flush=True)
    print(f"规则初审接受率 {q['review_accept_rate']}  自建/漏抽 {q['review_added']} 条", flush=True)
    print(f"L1 runs={l1.get('runs')} 正确率={l1.get('correct_rate')} 同答案抖动={l1.get('repeat_triple_spread')}", flush=True)
    print(f"L2 runs={l2.get('runs')} 通过率={l2.get('pass_rate')} must_use命中={l2.get('must_use_hit_rate')} "
          f"同答案分差={l2.get('triple_spread')} P90={pct(l2.get('latency_ms') or [], 90)}ms", flush=True)
    print(f"产出：data/extension_o02_report.json / _samples.json / _diff.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
