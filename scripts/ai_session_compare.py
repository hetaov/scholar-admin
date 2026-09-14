#!/usr/bin/env python3
"""沉浸式会话 v2(legacy) ↔ v3(新引擎) 双跑对照脚本（设计稿 §11.8 / T8）

同一 context 分别调用两套生成引擎，做**字段级差异摘要**，落盘
`data/nc2/compare/compare_<timestamp>.json`，作为小程序切换（T9）的决策依据。

- legacy：`services.providers.session_gen.generate_session_reply`（/ai/session/v2 生成核）
- v3    ：`services.learning.dialogue_engine.generate_session_reply`（/ai/session/v3 生成核）

**只读、不写线上**（§11.8）：
- v3 引擎以 `db=None` 运行 → 不接 checkpointer、不写 `ai_session_v3*`；
- legacy 引擎无任何持久化；
- 两引擎共用同一 context，产物仅落入本地对照 JSON。

用法：
  # 真实 LLM 双跑（需 VOLCANO_* 凭据；成本 = 轮次数 × 2 次 LLM）
  python scripts/ai_session_compare.py

  # 离线确定性（注入 fake 生成器，不触网；用于本地/CI 冒烟）
  python scripts/ai_session_compare.py --fake

  # 只跑开场 / 指定落盘目录
  python scripts/ai_session_compare.py --modes start --out-dir data/nc2/compare

退出码：0=两引擎在所选轮次均产出；1=任一引擎失败
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# 保证 `python scripts/xxx.py` 可运行（项目根入 sys.path）
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from services.learning import dialogue_engine  # noqa: E402
from services.providers import session_gen  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ai_session_compare")

# ---------------------------------------------------------------------------
# 对照 context（对齐 §11.8：同一 context 跑 v2 / v3）
#   —— 与 scripts/ai_session_eval.py 的 SESSION_V2_* 同口径，便于三处互相印证。
# ---------------------------------------------------------------------------

SCENARIO = {
    "scene_id": "negotiation",
    "title": "商务谈判 · 折扣条件",
    "scene": "Learner is a sales representative negotiating an order discount with a buyer.",
    "goal": "Secure a discount by offering order-volume terms",
    "constraints": "Stay in role; guide the learner to produce target sentences.",
}
ROLES = {
    "ai_role": {
        "name": "Buyer",
        "identity": "Procurement lead of a large client",
        "style": "polite but firm",
        "goal": "Nudge the learner to justify a discount request",
    },
    "learner_role": {"name": "Sales Rep", "identity": "Vendor representative"},
}
# 2 个新句（必用）+ 3 个复习句（召回落在 [2,6] 内，不触发 recall_insufficient）
GROUPS = [
    {
        "kind": "new",
        "sentences": [
            {"sentence_id": "tg_discount_1", "content": "We are willing to consider a discount if the order volume is substantial."},
            {"sentence_id": "tg_discount_2", "content": "A larger order would let us offer better payment terms."},
        ],
    },
    {
        "kind": "review",
        "sentences": [
            {"sentence_id": "rev_payment", "content": "We need to clarify the payment terms."},
            {"sentence_id": "rev_delivery", "content": "Could you confirm the delivery schedule?"},
            {"sentence_id": "rev_quality", "content": "Our quality control process is quite strict."},
        ],
    },
]

TURN_HISTORY = [
    {
        "role": "ai",
        "text": "Buyer: We're evaluating several vendors. What can you offer on a large volume?",
    },
]
TURN_USER_INPUT = "If you increase the volume, we can talk about a discount."


def _build_context(mode: str) -> dict:
    """构造 start / turn 的 context 快照（对齐 §4.18 / §11.4 字段）。"""
    if mode == "start":
        return {
            "mode": "start",
            "scenario": SCENARIO,
            "roles": ROLES,
            "materials": GROUPS,
            "history": [],
            "user_input": None,
            "assisted": False,
            "target_sentence_ids": [],
        }
    return {
        "mode": "turn",
        "scenario": SCENARIO,
        "roles": ROLES,
        "materials": GROUPS,
        "history": TURN_HISTORY,
        "user_input": TURN_USER_INPUT,
        "assisted": False,
        "target_sentence_ids": [],
    }


# ---------------------------------------------------------------------------
# 离线 fake（--fake：不触网，验证对照链路与差异摘要）
# ---------------------------------------------------------------------------

_FAKE_JSON_TEMPLATE = (
    '{{"content_type": "{content_type}", '
    '"ai_text": "（fake）Buyer: Let\'s talk numbers — how large is the order?", '
    '"hint": {{"levels": ["折扣：d...（线索）", "If you ___ the volume, we can ___ a discount.", '
    '"如果订单量足够大，我们可以谈折扣。"]}}, '
    '"suggested_targets": ["tg_discount_1"]}}'
)


def _fake_output_for(messages: list[dict]) -> str:
    """按引擎实际选择的 content_type 回显 fake 输出（反映 choose_content_type 差异）。"""
    system = ""
    for m in messages:
        if m.get("role") == "system":
            system = str(m.get("content") or "")
    match = re.search(r'"content_type":\s*"(dialogue|fill)"', system)
    content_type = match.group(1) if match else "dialogue"
    return _FAKE_JSON_TEMPLATE.format(content_type=content_type)


async def _fake_generator(messages: list[dict]) -> str:
    """注入 v3 引擎的 fake LLM（dialogue_engine.generate_session_reply(generator=...)）。"""
    return _fake_output_for(messages)


async def _fake_call_session_llm(messages: list[dict], timeout_seconds: int | None = None) -> str:
    """替换 legacy 引擎的 `session_gen.call_session_llm`（--fake 模式）。"""
    return _fake_output_for(messages)


# ---------------------------------------------------------------------------
# 单引擎执行 + 差异摘要
# ---------------------------------------------------------------------------


async def _run_engine(engine: str, coro) -> dict:
    """执行单个引擎调用，异常降级为结构化记录（不抛出，保证双跑都能落盘）。"""
    t0 = time.perf_counter()
    try:
        result = await coro
        return {
            "engine": engine,
            "status": "success",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            "result": result,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "engine": engine,
            "status": "failed",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            "error_code": getattr(e, "error_code", None) or type(e).__name__,
            "error": str(getattr(e, "detail", e))[:300],
        }


def _hint_meta(result: dict | None) -> dict:
    """hint 档位摘要：是否存在 / 定长档数 / 非空档数 / max_level。"""
    hint = (result or {}).get("hint")
    if not isinstance(hint, dict):
        return {"present": False, "max_level": None, "levels_count": 0, "nonempty_levels": 0}
    levels = hint.get("levels") or []
    return {
        "present": True,
        "max_level": hint.get("max_level"),
        "levels_count": len(levels),
        "nonempty_levels": sum(1 for x in levels if str(x).strip()),
    }


def _diff_summary(
    legacy: dict,
    v3: dict,
    *,
    required_ids: list[str],
    recall_ids: list[str],
) -> dict:
    """字段级差异摘要：content_type / hint 档位 / suggested_targets / 覆盖情况。"""
    legacy_reply = legacy.get("result") or {}
    v3_reply = v3.get("result") or {}
    legacy_targets = list(legacy_reply.get("suggested_targets") or [])
    v3_targets = list(v3_reply.get("suggested_targets") or [])
    legacy_hint = _hint_meta(legacy_reply)
    v3_hint = _hint_meta(v3_reply)

    def _covered(targets: list[str]) -> list[str]:
        return [t for t in targets if t in required_ids]

    return {
        "content_type": {
            "legacy": legacy_reply.get("content_type"),
            "v3": v3_reply.get("content_type"),
            "same": legacy_reply.get("content_type") == v3_reply.get("content_type"),
        },
        "ai_text_len": {
            "legacy": len(str(legacy_reply.get("ai_text") or "")),
            "v3": len(str(v3_reply.get("ai_text") or "")),
        },
        "hint_档位": {
            "legacy": legacy_hint,
            "v3": v3_hint,
            "same_levels": legacy_hint["nonempty_levels"] == v3_hint["nonempty_levels"],
        },
        "suggested_targets": {
            "legacy": legacy_targets,
            "v3": v3_targets,
            "same": legacy_targets == v3_targets,
        },
        "覆盖情况": {
            "required_ids": required_ids,
            "recall_ids": recall_ids,
            "legacy_covered_required": _covered(legacy_targets),
            "v3_covered_required": _covered(v3_targets),
            "legacy_hit_recall": [t for t in legacy_targets if t in recall_ids],
            "v3_hit_recall": [t for t in v3_targets if t in recall_ids],
        },
    }


async def _run_mode(mode: str, *, timeout: float | None, fake: bool) -> dict:
    """同一 context 下跑 legacy 与 v3，并输出差异摘要。"""
    context = _build_context(mode)
    normalized, required_ids, recall_ids, notes = dialogue_engine.normalize_session_materials(
        context.get("materials") or []
    )
    print(f"\n----- [双跑] mode={mode} -----")
    print(f"    required(new)={required_ids} recall(clamp后)={recall_ids} notes={notes or '无'}")

    legacy_kwargs = {"context": context, "preferred_type": "auto"}
    v3_kwargs = {
        "session_id": f"s_cmp_{mode}",
        "context": context,
        "preferred_type": "auto",
    }
    if timeout is not None:
        legacy_kwargs["timeout_seconds"] = int(timeout)
        v3_kwargs["timeout_seconds"] = int(timeout)
    if fake:
        v3_kwargs["generator"] = _fake_generator

    legacy = await _run_engine(
        "legacy", session_gen.generate_session_reply(**legacy_kwargs)
    )
    v3 = await _run_engine(
        "v3", dialogue_engine.generate_session_reply(**v3_kwargs)
    )

    for r in (legacy, v3):
        if r["status"] == "success":
            result = r.get("result") or {}
            print(
                f"    [{r['engine']:<6}] {r['elapsed_ms']}ms "
                f"content_type={result.get('content_type')} "
                f"hint={'有' if result.get('hint') else '无'} "
                f"targets={result.get('suggested_targets')}"
            )
        else:
            print(f"    [{r['engine']:<6}] FAILED {r.get('error_code')}: {r.get('error')}")

    diff = _diff_summary(legacy, v3, required_ids=required_ids, recall_ids=recall_ids)
    same = all(
        (
            diff["content_type"]["same"],
            diff["hint_档位"]["same_levels"],
            diff["suggested_targets"]["same"],
        )
    )
    if legacy["status"] == "success" and v3["status"] == "success":
        mark = "一致" if same else "有差异"
        print(f"    → 字段对照: content_type={diff['content_type']} "
              f"hint档位={diff['hint_档位']['legacy']['nonempty_levels']}"
              f" vs {diff['hint_档位']['v3']['nonempty_levels']} "
              f"targets={diff['suggested_targets']} [{mark}]")

    return {
        "mode": mode,
        "context_digest": {
            "scenario": SCENARIO,
            "roles": ROLES,
            "source": {
                "new_sentence_ids": required_ids,
                "review_sentence_ids": [
                    s["sentence_id"]
                    for g in GROUPS
                    if g.get("kind") == "review"
                    for s in g.get("sentences") or []
                ],
            },
            "v3_normalized": {
                "required_ids": required_ids,
                "recall_ids": recall_ids,
                "notes": notes,
                "recall_clamped": len(recall_ids) != sum(
                    len(g.get("sentences") or [])
                    for g in GROUPS
                    if g.get("kind") == "review"
                ),
            },
        },
        "legacy": legacy,
        "v3": v3,
        "diff": diff,
    }


async def _main_async(args: argparse.Namespace) -> int:
    if args.fake:
        # 离线：替换 legacy 引擎的 LLM 调用（v3 侧经 generator 注入）
        session_gen.call_session_llm = _fake_call_session_llm  # type: ignore[assignment]
        print("[模式] --fake：注入 fake 生成器，不触网（仅验证对照链路与差异摘要）")
    else:
        print("[模式] 真实 LLM 双跑（需 VOLCANO_* 凭据）")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    invalid = [m for m in modes if m not in ("start", "turn")]
    if invalid:
        print(f"[ERROR] --modes 非法：{invalid}（仅支持 start / turn）")
        return 1

    runs = {}
    for mode in modes:
        runs[mode] = await _run_mode(mode, timeout=args.timeout, fake=args.fake)

    any_failed = any(
        run["legacy"]["status"] != "success" or run["v3"]["status"] != "success"
        for run in runs.values()
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "fake" if args.fake else "live",
        "engines": {
            "legacy": "services.providers.session_gen.generate_session_reply",
            "v3": "services.learning.dialogue_engine.generate_session_reply",
        },
        "note": "只读对照：两引擎均未写入任何集合（v3 db=None 不接 checkpointer）。",
        "runs": runs,
        "any_failed": any_failed,
    }

    out_dir = args.out_dir if args.out_dir.is_absolute() else (HERE / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"compare_{ts}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[落盘] {out_path}")

    print("\n===== 双跑对照汇总 =====")
    for mode, run in runs.items():
        legacy_ok = run["legacy"]["status"] == "success"
        v3_ok = run["v3"]["status"] == "success"
        print(f"  {mode}: legacy={'✓' if legacy_ok else '✗'} v3={'✓' if v3_ok else '✗'}")
    print(f"  → {'FAIL（存在引擎失败）' if any_failed else 'PASS（两引擎均产出）'}")
    return 1 if any_failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="沉浸式会话 v2(legacy) ↔ v3(新引擎) 双跑对照（§11.8 / T8）"
    )
    parser.add_argument("--modes", default="start,turn",
                        help="对照轮次（逗号分隔）：start / turn（默认两者）")
    parser.add_argument("--fake", action="store_true",
                        help="离线确定性：注入 fake 生成器，不触网（本地/CI 冒烟）")
    parser.add_argument("--timeout", type=float, default=None,
                        help="单次 LLM 超时秒（缺省用各引擎默认 SESSION/DIALOGUE_LLM_TIMEOUT_SECONDS）")
    parser.add_argument("--out-dir", type=Path, default=Path("data/nc2/compare"),
                        help="对照 JSON 落盘目录（默认 data/nc2/compare；相对项目根）")
    args = parser.parse_args()
    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
