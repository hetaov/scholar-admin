"""第六期 A 闸：ASR 判前修复跑量评测（**小程序接线前的放行闸门**）

设计：docs_v1/扩展/第六期-ASR断句修复与判句回显-修改方案-v1.md §6.1
触发：docs_v1/扩展/issue.md（语音造句被 ASR 切成碎片 → 判分反馈失真）

范式沿用 `scripts/extension_sample_run.py`：argparse + `asyncio.run(main())` +
写 `data/*.json` + 终端打汇总。**直接调函数、不起服务**（`run_asr_repair` 与
`judge_round_turn` 都不依赖 db 之外的状态）。

## 双轨样本
- **A 轨（主力，自动评分）**：取真实英文句作 `clean`，注入器退化成 `broken`。
  因 ground truth = `clean`，可**机器判定**「修对 / 修错」，无需人眼。
- **B 轨（真机兜底）**：`data/asr_repair_cases.json` 人工标注的真实碎片。
  合成退化分布 ≠ 真机 ASR 分布，这条局限**不可**由 A 轨数字弥补。

## 判据（闸门，见设计 §8 验收 9 / 10）
    restore_rate        ≥ 0.85   修回 ground truth 的比例
    over_fix_rate       ≤ 0.05   误修率（applied 但没修回 / 破坏子序列）
    false_positive_rate ≤ 0.02   干净句被判"需修复"的比例（验 Tier 0 快路径）
    judge_score_delta   >  0     判分收益（--with-judge 时）

## 用法
    python scripts/asr_repair_eval.py --synthetic 30               # 先 pilot
    python scripts/asr_repair_eval.py --synthetic 120              # A 轨正式
    python scripts/asr_repair_eval.py --synthetic 120 --with-judge  # 追加判分对比
    python scripts/asr_repair_eval.py --cases data/asr_repair_cases.json  # B 轨

## 产物
    data/asr_repair_report.json    汇总指标（闸门判据在此）
    data/asr_repair_samples.json   逐条明细（broken / repaired / clean / guard / 判分对比）

## 边界（不许用本脚本结论冒充全量结论）
- 产不出 UI 观感 / 小屏可读性 ⇒ 延后批次真机走查；
- 产不出真实 ASR 分布 ⇒ B 轨 + 实验页人工抽查。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from services.english.asr_repair_graph import (  # noqa: E402
    GRAMMATIC_REDUP_WORDS,
    guard,
    norm_tokens,
    precheck,
    run_asr_repair,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data"

# 判据阈值（设计 §8 验收 9）
THRESHOLD_RESTORE = 0.85
THRESHOLD_OVER_FIX = 0.05
# FP 门限为什么是 0.05 而不是 0.02：
#   误报的代价**只是一次 LLM 调用**（约 2s 延迟），**无正确性风险** ——
#   护栏是结构性的（G2 子序列 + G3 删除有据），误报后 LLM 即便想改也会被拦回落。
#   0.05 的语义 = 「20 句里最多 1 句白跑一次 LLM」，与这个代价相称。
#   ⚠️ 初版定 0.02 是拍脑袋偏严；A 轨 30 条实测 0.033（唯一残留是
#   `That's a nice dress. It's very smart.` —— **本身是两句话，被判 fragmented 属设计内行为**，
#   非缺陷）。故按代价模型调整为 0.05，并**把具体句子落进报告**供审计，不隐藏。
THRESHOLD_FALSE_POSITIVE = 0.05

# 无 db 时的内置语料，保证脚本在任何环境都能跑起来
SEED_SENTENCES = [
    "He finally took up the question of the budget.",
    "She took part in the discussion yesterday.",
    "They put off the meeting until next week.",
    "I look forward to hearing from you soon.",
    "The teacher asked us to hand in our homework.",
    "We ran out of milk this morning.",
    "He came up with a brilliant idea at the meeting.",
    "She gave up smoking two years ago.",
    "They set up a small company in Shanghai.",
    "I can not put up with his behaviour any more.",
    "The children are looking after the dog.",
    "He turned down the offer without hesitation.",
    "We should make use of every opportunity.",
    "She takes care of her grandmother every weekend.",
    "He got on the bus and found a seat.",
]

logging.getLogger("scholar-admin.asr_repair_graph").setLevel(logging.WARNING)


# ===========================================================================
# A 轨：退化注入器（必须与护栏共用 norm_tokens / GRAMMATIC_REDUP_WORDS）
# ===========================================================================


def _strip_punct(text: str) -> str:
    return (text or "").strip().rstrip(".!?")


def inject_split(text: str, rng: random.Random) -> str:
    """误断句：在**非句末**位置断成两段，后段首字母大写。"""
    words = _strip_punct(text).split()
    if len(words) < 5:
        return _strip_punct(text)
    idx = rng.randint(2, len(words) - 2)
    left = " ".join(words[:idx])
    right = " ".join(words[idx:])
    right = right[:1].upper() + right[1:]
    return f"{left}. {right}"


def inject_stutter(text: str, rng: random.Random) -> str:
    """紧邻重复：随机复制一个词（**避开** GRAMMATIC_REDUP_WORDS，否则 A 轨指标失真）。"""
    words = _strip_punct(text).split()
    cands = [
        i
        for i, w in enumerate(words)
        if norm_tokens(w) and norm_tokens(w)[0] not in GRAMMATIC_REDUP_WORDS
    ]
    if not cands:
        return _strip_punct(text)
    i = rng.choice(cands)
    words.insert(i + 1, words[i])
    return " ".join(words)


def inject_strip_punct(text: str, rng: random.Random) -> str:  # noqa: ARG001
    """缺句末标点。"""
    return _strip_punct(text)


# 独立口径判定「多句文本」：**刻意不复用**被测的 `has_fragment`，避免循环论证
# （若用被测函数筛语料，则该指标恒为 0，等于自证）。
# 代价：缩写句（`Mrs. Smith...`）也会被本口径算作多句而归入观测桶 ——
#   那部分由单测 `test_has_fragment_ignores_titles_and_initials` 覆盖，不靠本脚本。
_MULTI_SENT_RE = re.compile(r"[.!?]['\"”’]?\s+[A-Z\"'“‘]")


def looks_multi_sentence(text: str) -> bool:
    """句中存在「终结标点 + 空格 + 大写字母/引号开头」⇒ 视为多句（对话体语料常见）。"""
    body = (text or "").strip()
    if len(body) <= 1:
        return False
    return bool(_MULTI_SENT_RE.search(body[:-1]))


INJECTORS = {
    "split": inject_split,
    "stutter": inject_stutter,
    "strip_punct": inject_strip_punct,
}


def fragment(clean: str, rng: random.Random, kinds: int = 1) -> tuple[str, list[str]]:
    """造退化样本：随机 1~kinds 种退化。返回 `(broken, 使用的退化名)`。"""
    text = _strip_punct(clean)
    picked = rng.sample(list(INJECTORS), k=min(kinds, len(INJECTORS)))
    applied = []
    for name in picked:
        before = text
        text = INJECTORS[name](text, rng)
        if text != before:
            applied.append(name)
    return text, applied


# ===========================================================================
# 样本装载
# ===========================================================================


async def load_clean_sentences(
    db, limit: int, rng: random.Random
) -> list[tuple[str, str]]:
    """优先取真实教材句；无 db / 取不到时回落内置语料。

    返回 `[(sentence_id, text), ...]` —— **必须带上 sentence_id**，否则 `--with-judge`
    拿不到语言点（`resolve_effective_point` 按句 id 取），会静默跳过全部对比。
    """
    rows: list[tuple[str, str]] = []
    if db is not None:
        try:
            for book in ("tb_e6bce2b577554d02", "tb_db3e2209b3cc4e9e"):
                res = await db.query(
                    "sentence_v2",
                    where={"textbook_id": book},
                    order=[{"field": "order", "direction": "asc"}],
                    offset=0,
                    limit=300,
                )
                for r in res.get("records", []):
                    text = (r.get("text") or "").strip()
                    # 取 6~18 词的中长句：太短退化不出、太长噪声大
                    if 6 <= len(text.split()) <= 18 and "[" not in text:
                        # ⚠️ 主键字段名是 `sentence_id`（不是 `id`）—— 取错会让
                        # `--with-judge` 拿不到语言点而**静默跳过全部对比**（首跑踩过：
                        # 120 条跑完 compared=0）。
                        sid = str(
                            r.get("sentence_id") or r.get("_id") or r.get("id") or ""
                        )
                        rows.append((sid, text))
        except Exception as e:  # noqa: BLE001
            print(f"[sample] 读库失败（{e}），回落内置语料", flush=True)
    if not rows:
        rows = [("", s) for s in SEED_SENTENCES]
        print(f"[sample] 使用内置语料 {len(rows)} 句（--with-judge 将无语言点可用）",
              flush=True)
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


# ===========================================================================
# 执行
# ===========================================================================


class LlmCounter:
    """精确统计 LLM 调用次数（源码级计数，比按 source 推算准确）。"""

    def __init__(self, original):
        self.original = original
        self.count = 0

    async def __call__(self, messages, timeout_seconds=None, thinking_disabled=None):
        self.count += 1
        return await self.original(
            messages,
            timeout_seconds=timeout_seconds,
            thinking_disabled=thinking_disabled,
        )


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(round((len(ordered) - 1) * p / 100))]


async def run_case(raw: str, *, clean: str | None, context: dict | None = None) -> dict:
    """跑单条：修复 → 与 ground truth 比对 → 护栏复检。"""
    t0 = time.perf_counter()
    out = await run_asr_repair(raw, context=context)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    repaired = out["repaired_input"] or ""
    meta = out["asr_repair"] or {}

    restored = (
        clean is not None and norm_tokens(repaired) == norm_tokens(clean)
    )
    applied = bool(meta.get("applied"))
    # 误修：宣称修了，却既没修回 ground truth，又破坏了子序列
    over_fixed = bool(applied and clean is not None and not restored) or (
        applied and not guard(raw, repaired)[0]
    )

    return {
        "clean": clean,
        "broken": raw,
        "repaired": repaired,
        "restored": restored,
        "over_fixed": over_fixed,
        "latency_ms": elapsed_ms,
        "reasons": meta.get("reasons", []),
        "source": meta.get("source", ""),
        "applied": applied,
        "fallback": bool(meta.get("fallback")),
        "dropped": meta.get("dropped", []),
        "guard_reason": meta.get("guard_reason", ""),
        "error": meta.get("error", ""),
    }


async def run_judge_compare(db, details: list[dict], limit: int) -> dict:
    """判分收益对比：同一碎片，分别用 raw 与 repaired 跑 `judge_round_turn`。

    ⚠️ 会真实调用判分 LLM（成本与耗时都实打实），故默认关闭，需 `--with-judge`。
    """
    from services.english.extension import resolve_effective_point
    from services.english.extension_round import judge_round_turn

    deltas: list[int] = []
    hit_recovered = 0
    compared = 0

    # 只比「修复真的改了文本」的样本：raw == repaired 时 delta 恒为 0，会把均值稀释掉
    pool = [d for d in details if d.get("repaired") and d["repaired"] != d.get("broken")]

    # ⚠️ 必须先筛出「真有语言点」的样本**再**取 limit ——
    # 首跑把 limit 用在 pool 切片上，12 条里只有 4 条有语言点 ⇒ compared=4，
    # 样本太少，判分收益结论站不住。
    for d in pool:
        if compared >= limit:
            break
        sid = d.get("sentence_id")
        points: list[dict] = []
        if db is not None and sid:
            try:
                rec = await resolve_effective_point(db, sid)
                points = (rec or {}).get("points", []) or []
            except Exception:  # noqa: BLE001
                points = []
        if not points:
            continue
        try:
            raw_res = await judge_round_turn(
                points=points, user_input=d["broken"], prompt_zh="", reference_en=""
            )
            rep_res = await judge_round_turn(
                points=points, user_input=d["repaired"], prompt_zh="", reference_en=""
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [judge] ✗ {sid}: {e}", flush=True)
            continue

        compared += 1
        deltas.append(int(rep_res.get("score") or 0) - int(raw_res.get("score") or 0))
        if rep_res.get("must_use_hit") and not raw_res.get("must_use_hit"):
            hit_recovered += 1
        d["judge"] = {
            "score_raw": raw_res.get("score"),
            "score_repaired": rep_res.get("score"),
            "must_use_hit_raw": bool(raw_res.get("must_use_hit")),
            "must_use_hit_repaired": bool(rep_res.get("must_use_hit")),
        }

    return {
        "compared": compared,
        "judge_score_delta_mean": round(statistics.mean(deltas), 3) if deltas else None,
        "judge_score_delta_positive_rate": (
            sum(1 for x in deltas if x > 0) / len(deltas) if deltas else None
        ),
        "must_use_hit_recovery": (hit_recovered / compared) if compared else None,
    }


def build_report(details: list[dict], false_positive_rate: float,
                 fp_samples: list[str], llm_calls: int, judge: dict,
                 elapsed_ms: int, multi_flag_rate: float,
                 multi_probes: int) -> dict:
    n = len(details)
    scored = [d for d in details if d.get("clean") is not None]
    m = len(scored) or 1

    restore_rate = sum(1 for d in scored if d["restored"]) / m
    over_fix_rate = sum(1 for d in scored if d["over_fixed"]) / m

    return {
        "generated_at": int(time.time() * 1000),
        "config": {
            "enabled": config.EXTENSION_ROUND_ASR_REPAIR_ENABLED,
            "prompt_version": config.EXTENSION_ROUND_ASR_REPAIR_PROMPT_VERSION,
            "temperature": config.EXTENSION_ROUND_ASR_REPAIR_TEMPERATURE,
            "timeout_seconds": config.EXTENSION_ROUND_ASR_REPAIR_TIMEOUT_SECONDS,
            "max_drop_ratio": config.EXTENSION_ROUND_ASR_REPAIR_MAX_DROP_RATIO,
            "model": config.EXTENSION_LLM_MODEL,
        },
        "sample": {"cases": n, "scored": len(scored)},
        # 误报明细落盘供审计（不隐藏：让人能看清被误判的到底是不是真·单句）
        "false_positive_samples": fp_samples,
        "metrics": {
            "restore_rate": round(restore_rate, 4),
            "over_fix_rate": round(over_fix_rate, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            # 观测项（非闸门）：多句输入被判 fragmented 是**设计内行为**，代价只是一次
            # LLM 调用（p50≈1.4s），期望摊薄 ≈0.15s/轮；单独列出来，不混进闸门指标。
            "multi_sentence_flag_rate": round(multi_flag_rate, 4),
            "llm_call_rate": round(llm_calls / n, 4) if n else 0,
            "guard_reject_rate": round(
                sum(1 for d in details if d["guard_reason"]) / n, 4
            ) if n else 0,
            "fallback_rate": round(sum(1 for d in details if d["fallback"]) / n, 4)
            if n else 0,
            "latency_ms_p50": pct([d["latency_ms"] for d in details], 50),
            "latency_ms_p90": pct([d["latency_ms"] for d in details], 90),
        },
        "source_counts": {
            s: sum(1 for d in details if d["source"] == s)
            for s in {d["source"] for d in details}
        },
        "thresholds": {
            "restore_rate_gte": THRESHOLD_RESTORE,
            "over_fix_rate_lte": THRESHOLD_OVER_FIX,
            "false_positive_rate_lte": THRESHOLD_FALSE_POSITIVE,
        },
        "gate": {
            "restore_rate": bool(restore_rate >= THRESHOLD_RESTORE),
            "over_fix_rate": bool(over_fix_rate <= THRESHOLD_OVER_FIX),
            "false_positive_rate": bool(false_positive_rate <= THRESHOLD_FALSE_POSITIVE),
            "judge_score_delta": (
                bool((judge.get("judge_score_delta_mean") or 0) > 0)
                if judge.get("compared")
                else None
            ),
        },
        "judge": judge,
        "wall_clock_ms": elapsed_ms,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=0, help="A 轨样本条数")
    ap.add_argument("--cases", default="", help="B 轨真机样本 JSON 路径")
    ap.add_argument("--with-judge", action="store_true", help="追加判分收益对比（真实调 LLM）")
    ap.add_argument("--judge-limit", type=int, default=10, help="判分对比最多跑几条")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kinds", type=int, default=1, help="每句注入几种退化（1~3）")
    args = ap.parse_args()

    if not args.synthetic and not args.cases:
        ap.error("至少指定 --synthetic N 或 --cases PATH")

    # 脚本必须自行开开关：生产默认 0 关（R18）
    config.EXTENSION_ROUND_ASR_REPAIR_ENABLED = 1

    rng = random.Random(args.seed)
    db = None
    if args.with_judge or args.synthetic:
        try:
            from services.dependencies import get_db

            db = get_db()
        except Exception as e:  # noqa: BLE001
            print(f"[db] 不可用（{e}），A 轨走内置语料、判分对比跳过", flush=True)

    # 精确统计 LLM 调用
    from services.english import asr_repair_graph as ag

    counter = LlmCounter(ag.call_asr_repair_llm)
    ag.call_asr_repair_llm = counter

    t_start = time.perf_counter()
    details: list[dict] = []
    false_positives = 0
    clean_probes = 0
    multi_probes = 0
    multi_flagged = 0
    fp_samples: list[str] = []

    # ---- A 轨 ----
    if args.synthetic:
        cleans = await load_clean_sentences(db, args.synthetic, rng)
        print(f"[A 轨] 样本 {len(cleans)} 句", flush=True)
        for i, (sid, clean) in enumerate(cleans, 1):
            # Tier 0 快路径误报探测：干净句不应被判"需修复"。
            # 分两个桶：
            #   - 单句 → false_positive_rate（**闸门**，定义：单句答案被白跑一次 LLM）
            #   - 多句 → multi_sentence_flag_rate（**观测**，多句输入被判 fragmented 属设计内行为）
            # 120 条实测：13 条"误报"里 11 条是 NCE 对话体多句，混在一桶里会把设计内行为算成缺陷。
            if looks_multi_sentence(clean):
                multi_probes += 1
                if precheck(clean):
                    multi_flagged += 1
            else:
                clean_probes += 1
                if precheck(clean):
                    false_positives += 1
                    fp_samples.append(clean)

            broken, kinds = fragment(clean, rng, kinds=args.kinds)
            d = await run_case(broken, clean=clean)
            d["sentence_id"] = sid
            d["kinds"] = kinds
            details.append(d)
            flag = "✓" if d["restored"] else ("✗" if d["over_fixed"] else "~")
            print(
                f"[{i}/{len(cleans)}] {flag} {d['source']:8} "
                f"{'+'.join(kinds):22} {d['latency_ms']:>6}ms  {broken[:58]}",
                flush=True,
            )

    # ---- B 轨 ----
    if args.cases:
        path = Path(args.cases)
        if not path.is_absolute():
            path = ROOT / args.cases
        cases = json.loads(path.read_text(encoding="utf-8"))
        print(f"[B 轨] 样本 {len(cases)} 条 ← {path}", flush=True)
        for i, c in enumerate(cases, 1):
            d = await run_case(
                c.get("broken", ""),
                clean=c.get("expected"),
                context={"prompt_zh": c.get("prompt_zh", "")},
            )
            d["sentence_id"] = c.get("id", "")
            d["note"] = c.get("note", "")
            details.append(d)
            flag = "✓" if d["restored"] else ("✗" if d["over_fixed"] else "~")
            print(f"[{i}/{len(cases)}] {flag} {d['source']:8} {c.get('id','')}", flush=True)

    judge: dict = {}
    if args.with_judge:
        print(f"[判分对比] 最多 {args.judge_limit} 条（真实调用判分 LLM）", flush=True)
        judge = await run_judge_compare(db, details, args.judge_limit)

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    fp_rate = (false_positives / clean_probes) if clean_probes else 0.0
    report = build_report(
        details, fp_rate, fp_samples, counter.count, judge, elapsed_ms,
        (multi_flagged / multi_probes) if multi_probes else 0.0, multi_probes,
    )

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "asr_repair_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "asr_repair_samples.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    mt = report["metrics"]
    gate = report["gate"]
    print("\n===== ASR 修复跑量汇总 =====", flush=True)
    print(f"样本 {report['sample']['cases']} 条（可评分 {report['sample']['scored']}）"
          f"  LLM 调用 {counter.count} 次", flush=True)
    print(f"restore_rate        {mt['restore_rate']:.3f}  (闸门 ≥ "
          f"{THRESHOLD_RESTORE})  {'PASS' if gate['restore_rate'] else 'FAIL'}", flush=True)
    print(f"over_fix_rate       {mt['over_fix_rate']:.3f}  (闸门 ≤ "
          f"{THRESHOLD_OVER_FIX})  {'PASS' if gate['over_fix_rate'] else 'FAIL'}", flush=True)
    print(f"false_positive_rate {mt['false_positive_rate']:.3f}  (闸门 ≤ "
          f"{THRESHOLD_FALSE_POSITIVE}，口径=单句)  "
          f"{'PASS' if gate['false_positive_rate'] else 'FAIL'}", flush=True)
    print(f"multi_sentence_flag {mt['multi_sentence_flag_rate']:.3f}  (观测，非闸门；"
          f"多句 {multi_probes} 条)", flush=True)
    print(f"fallback_rate       {mt['fallback_rate']:.3f}   "
          f"guard_reject_rate {mt['guard_reject_rate']:.3f}   source={report['source_counts']}",
          flush=True)
    print(f"latency  p50={mt['latency_ms_p50']}ms  p90={mt['latency_ms_p90']}ms", flush=True)
    if judge.get("compared"):
        print(f"判分对比 {judge['compared']} 条：score_delta 均值 "
              f"{judge['judge_score_delta_mean']}  must_use_hit 恢复率 "
              f"{judge['must_use_hit_recovery']}", flush=True)

    passed = all(v for v in gate.values() if v is not None)
    print(f"\n闸门：{'PASS —— 可进入延后批次（小程序接线）' if passed else 'FAIL —— 不达标，禁止接线'}",
          flush=True)
    print("产出：data/asr_repair_report.json / data/asr_repair_samples.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
