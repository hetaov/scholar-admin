"""POST /math/practice-sheet 功能问题点探针（离线，不触网不连真库）

背景：
- 小程序 callContainer 超时上限 15s（wx-knowlege-graph/services/tcb.js 已用满 15000ms），
  超时即表现为「请求失败: POST /math/practice-sheet timeout」。
- generatePracticeSheet 在请求内同步等待 LLM 出题（最多 3 知识点 × (基础 + 奥数) = 6 次调用），
  若串行执行且单次调用较慢，总时长线性叠加，极易超过 15s。

本脚本用 FakeDB + 假 LLM 客户端（可注入单次时延）离线复现 generatePracticeSheet，
量化三类问题点并验证修复效果：
  P1 单次 LLM 调用是否禁用 thinking（推理模型不禁用单次可 >60s）
  P2 知识点间 LLM 调用是否串行（调用数 × 单次时延 = 总时长），实测最大并发度
  P3 无整体超时预算（OpenAI client 60s 上限 ≫ 15s 容器上限）

用法示例（scholar-admin 根目录，服务进程同环境）：
    python3 scripts/math_practice_sheet_probe.py               # 时延 0：只测结构/调用数/契约
    python3 scripts/math_practice_sheet_probe.py --latency 3   # 模拟单次 LLM 3s：实测墙钟 vs 15s
    python3 scripts/math_practice_sheet_probe.py --case C --latency 4  # 单场景
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

# 允许直接以脚本方式运行（scripts/ 下无包结构）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import services.math.practice_sheet as ps  # noqa: E402
from tests.fakes.fake_db import FakeDB  # noqa: E402

SCHOLAR = "scholar_probe_001"
CLIENT_CAP_SEC = 15.0  # 小程序 callContainer 上限（tcb.js）
LLM_CALL_CAP_SEC = 13.0  # 留 ~2s 给 DB/落库/审计
_PROMPT_COUNT_RE = re.compile(r"生成 (\d+) 道")


# ---------------------------------------------------------------------------
# 假 LLM：记录每次调用的起止/重叠并发，可注入单次时延 / 提前失败 / 题量不足
# ---------------------------------------------------------------------------
class FakeLLM:
    def __init__(self, latency: float = 0.0, partial: bool = False):
        self.latency = latency
        self.partial = partial
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        self._active = 0
        self.peak_concurrency = 0

    def _create(self, **kwargs):
        prompt = ""
        for msg in kwargs.get("messages") or []:
            if msg.get("role") == "user":
                prompt = msg.get("content") or ""
        m = _PROMPT_COUNT_RE.search(prompt)
        n = int(m.group(1)) if m else 2
        if self.partial and n > 1:
            n -= 1  # 题量不足场景：请求 2 道只回 1 道
        with self._lock:
            started = self._active
            self._active += 1
            self.peak_concurrency = max(self.peak_concurrency, self._active)
            rec = {"n": n, "started": time.perf_counter(), "overlap": started}
            self.calls.append(rec)
        time.sleep(self.latency)
        with self._lock:
            self._active -= 1
        items = [
            {
                "question": f"probe-题{i}-{rec['n']}",
                "answer": f"probe-答{i}",
                "difficulty": 3,
                "hint_card": "家长提示",
            }
            for i in range(n)
        ]
        content = json.dumps({"items": items}, ensure_ascii=False)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeOpenAI:
    """client.chat.completions.create(...) 替身"""

    def __init__(self, llm: FakeLLM):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=llm._create))


# ---------------------------------------------------------------------------
# 种子数据
# ---------------------------------------------------------------------------
def _node(node_id: str, code: str, title: str, kp_name: str) -> dict:
    return {
        "node_id": node_id,
        "code": code,
        "title": title,
        "grade": "五年级",
        "semester": "up",
        "unit_title": "分数",
        "lesson_title": title,
        "textbook_id": "TB_MATH_5",
        "description_version": 1,
        "ai_summary": {
            "status": "success",
            "knowledge_points": [
                {
                    "name": kp_name,
                    "summary": f"{kp_name} 一句话总结",
                    "ability_dimensions": ["arithmetic"],
                    "source_node_id": node_id,
                    "source_lesson_id": "",
                }
            ],
            "extended_points": [
                {
                    "name": f"{kp_name}·奥数",
                    "summary": "奥数扩展点说明",
                    "difficulty_band": "入门",
                    "related_knowledge_name": kp_name,
                    "source_lesson_id": "",
                }
            ],
        },
    }


def _seed_ai_nodes(db: FakeDB, kps: list[str]) -> list[dict]:
    nodes = []
    for i, kp in enumerate(kps):
        nid, code = f"n_probe_{i}", f"c_probe_{i}"
        nodes.append(_node(nid, code, f"课时{i}", kp))
        db.add("curriculum_node", nodes[-1])
    return nodes


def _seed_error_records(db: FakeDB, codes: list[str], occurrence: int = 3) -> None:
    for i, code in enumerate(codes):
        db.add(
            "error_record",
            {
                "scholar_id": SCHOLAR,
                "node_code": code,
                "occurrence": occurrence,
                "primary_error": "concept" if i % 2 == 0 else "calculation",
            },
        )


# ---------------------------------------------------------------------------
# 场景执行与断言
# ---------------------------------------------------------------------------
def _check_contract(out: dict, *, source: str, expected_items: int) -> list[str]:
    """契约字段校验（api-contract §3.10 出参）：返回问题清单（空 = 通过）"""
    problems: list[str] = []
    if out.get("source") != source:
        problems.append(f"source 回显不符: {out.get('source')!r} != {source!r}")
    if out.get("status") != "generated":
        problems.append(f"status 应为 generated: {out.get('status')!r}")
    got = len(out.get("items") or [])
    if got != expected_items:
        problems.append(f"题量不符: 期望 {expected_items} 实际 {got}")
    for it in out.get("items") or []:
        if "answer" in it:
            problems.append("出参泄漏 answer（防背题契约）")
            break
        if "hint_card" in it:
            problems.append("出参泄漏 hint_card")
            break
    if len(out.get("nodes") or []) > 3:
        problems.append("nodes 超过 3（A4 篇幅上限）")
    return problems


async def _run_case(
    *,
    name: str,
    source: str,
    latency: float,
    knowledge_points: list | None = None,
    include_ext: bool = False,
    seed_fn=None,
) -> dict:
    db = FakeDB()
    if seed_fn:
        seed_fn(db)
    llm = FakeLLM(latency=latency)
    fake = FakeOpenAI(llm)
    orig_client, orig_model, orig_schedule = (
        ps._get_llm_client,
        ps.LLM_SUMMARY_MODEL,
        ps._schedule_render,
    )
    ps._get_llm_client = lambda: fake
    ps.LLM_SUMMARY_MODEL = "probe-llm-model"
    ps._schedule_render = lambda _db, _sid: None  # 探针不渲染（不触 playwright）
    try:
        # 第一次生成
        t0 = time.perf_counter()
        out = await ps.generatePracticeSheet(
            db,
            scholar_id=SCHOLAR,
            source=source,
            knowledge_points=knowledge_points,
            include_extended_points=include_ext,
        )
        first_wall = time.perf_counter() - t0
        calls_first = len(llm.calls)

        # 幂等复跑（同参数 10 分钟内）：不应再调 LLM / 不应再落库
        t1 = time.perf_counter()
        out2 = await ps.generatePracticeSheet(
            db,
            scholar_id=SCHOLAR,
            source=source,
            knowledge_points=knowledge_points,
            include_extended_points=include_ext,
        )
        idem_wall = time.perf_counter() - t1
        calls_idem = len(llm.calls) - calls_first
        sheets = len(db.all("practice_sheet"))

        problems = _check_contract(out, source=source, expected_items=len(out["items"]))
        if calls_idem != 0:
            problems.append(f"幂等复跑仍调了 {calls_idem} 次 LLM")
        if sheets != 1:
            problems.append(f"幂等复跑重复落库 practice_sheet={sheets} 份")
        # 按文档口径人工核对 expected 题量（调用数 × 每点 2 题 + 奥数每点 1 题）
        n_basic = max(1, (knowledge_points or []).__len__()) if knowledge_points else 0
        if source == "wrong_book":
            # wrong_book 题量 = 聚合知识点数 × 2（种子 3 条记录 → 3 节点）
            return {
                "name": name,
                "source": source,
                "llm_calls": calls_first,
                "items": len(out["items"]),
                "first_wall_s": round(first_wall, 3),
                "idem_wall_s": round(idem_wall, 3),
                "idem_llm_calls": calls_idem,
                "peak_concurrency": llm.peak_concurrency,
                "problems": problems,
            }
        expected = n_basic * 2 + (n_basic if include_ext else 0)
        if len(out["items"]) != expected:
            problems.append(f"题量不符: 期望 {expected} 实际 {len(out['items'])}")
        return {
            "name": name,
            "source": source,
            "llm_calls": calls_first,
            "items": len(out["items"]),
            "first_wall_s": round(first_wall, 3),
            "idem_wall_s": round(idem_wall, 3),
            "idem_llm_calls": calls_idem,
            "peak_concurrency": llm.peak_concurrency,
            "problems": problems,
        }
    finally:
        ps._get_llm_client = orig_client
        ps.LLM_SUMMARY_MODEL = orig_model
        ps._schedule_render = orig_schedule


def _thinking_disabled() -> bool:
    src = inspect.getsource(ps._call_chat_sync)
    return "thinking" in src


async def main(args: argparse.Namespace) -> int:
    kps3 = [f"知识点甲", f"知识点乙", f"知识点丙"]

    def seed_wrong3(db):
        _seed_error_records(db, ["c_probe_0", "c_probe_1", "c_probe_2"])

    def seed_ai3(db):
        _seed_ai_nodes(db, kps3)

    cases = {
        "A": dict(name="A·wrong_book×3节点", source="wrong_book", seed_fn=seed_wrong3),
        "B": dict(name="B·ai_knowledge×3点", source="ai_knowledge", knowledge_points=[{"name": k} for k in kps3], seed_fn=seed_ai3),
        "C": dict(name="C·ai_knowledge×3+奥数", source="ai_knowledge", knowledge_points=[{"name": k} for k in kps3], include_ext=True, seed_fn=seed_ai3),
    }
    picks = cases if args.case == "all" else {args.case: cases[args.case]}

    print(f"== POST /math/practice-sheet 探针 == 单次 LLM 时延={args.latency}s，容器上限 {CLIENT_CAP_SEC}s")
    print(f"P1 出题 LLM 调用是否禁用 thinking: {'是' if _thinking_disabled() else '否（推理模型单次可>60s，超时主因之一）'}")
    print()

    header = (
        f"{'场景':<28}{'LLM次数':>8}{'题量':>5}{'墙钟(s)':>9}{'幂等墙钟':>9}"
        f"{'并发峰值':>8}  契约问题"
    )
    print(header)
    print("-" * len(header))
    worst_serial: list[tuple[str, int]] = []
    for key, kw in picks.items():
        res = await _run_case(latency=args.latency, **kw)
        worst_serial.append((res["name"], res["llm_calls"]))
        flag = ""
        # 串行模型下最坏墙钟 ≈ N×latency + 实测开销（latency=0 的墙钟已含 DB/组装）
        if args.latency > 0 and res["first_wall_s"] >= CLIENT_CAP_SEC:
            flag = " ⚠ >15s 容器上限"
        print(
            f"{res['name']:<28}{res['llm_calls']:>8}{res['items']:>5}"
            f"{res['first_wall_s']:>9}{res['idem_wall_s']:>9}{res['peak_concurrency']:>8}"
            f"  {'; '.join(res['problems']) or '通过'}{flag}"
        )
    print()

    print("== 超时预算推演（串行 N×L vs 15s 上限；并发优化后 ≈ 1~2 波 × L） ==")
    overhead = 1.0  # 探针实测 DB/组装开销量级（毫秒~百毫秒，此处保守取 1s 说明问题）
    print(f"{'场景':<28}" + "".join(f"{L:>5}s" for L in (4, 6, 8, 10)))
    for name, n in worst_serial:
        row = f"{name:<28}"
        for L in (4, 6, 8, 10):
            est = overhead + n * L
            row += f"{est:>5.0f}{'⚠' if est >= CLIENT_CAP_SEC else ''}"
        print(row)
    print(f"（结论：LLM 调用数 {max((n for _, n in worst_serial), default=1)} 时，"
          f"串行在单次 >{(CLIENT_CAP_SEC - overhead) / max((n for _, n in worst_serial), default=1):.1f}s 即超时；"
          f"必须并行 + 单次限时 + 禁用 thinking）")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POST /math/practice-sheet 问题点探针")
    parser.add_argument("--latency", type=float, default=0.0, help="模拟单次 LLM 时延（秒），0=只测结构与调用数")
    parser.add_argument("--case", choices=("A", "B", "C", "all"), default="all", help="场景：A wrong_book / B ai / C ai+奥数")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
