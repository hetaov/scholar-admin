#!/usr/bin/env python3
"""数学错题拍照链路产物 · 混元评估脚本（LLM-as-Judge）

承接链路脚本 `math_wrong_photo_chain_verify.py` 的产物做质量评估：
  被评对象 = 一次错题拍照全链路产物（1 张练习卷照片
    → OCR 文本 → Judge 归类(needs_review) → correct 人工修正落库 error_record）。
  Judge ≠ Generator：产物由火山 vision / Judge（生成侧）产出，
  评分固定用混元（HUNYUAN_EVAL_MODEL=hy3，OpenAI 兼容网关，配置见 .env HUNYUAN_*）。

评分面（每面一次混元调用，按权重维度打分 0~1，阈值 HUNYUAN_EVAL_PASS_THRESHOLD=0.7）：
  ocr            —— OCR 文本质量（完整/数值保真/可切题/自洽）
  judge_classify —— Judge 切题 + 知识点定位 + 错因四分类 + 置信度门控（对齐 B1.5a：
                    候选外高置信先语义就近改名挂正式（error_record.original_kp_name
                    保留原判名，L2 计 auto_direct_renamed），无就近候选才 EXTRA_AI
                    直落（auto_direct_extra_ai）；操作符互斥/仅共享模板后缀不得改名，
                    护栏拦下保持 EXTRA_AI 不算漏改名；items 按 classify_method 反链标 gate）
  manual_correct —— correct 归链质量（正式节点锚定 / EXTRA_AI 图谱外新建复用 / 就近修正命名，B1.5 语义）

数据源：直接读 TCB DB（.env TENCENTCLOUD_SECRETID/SECRETKEY/TCB_ENV_ID），不依赖本地服务：
  math_scan_upload（scan_id → ocr_text / ocr_status / classify_result[]）
  error_record（scan_upload_id=scan_id → correct 后落库记录）
  scholar_book + curriculum_node(ai_summary) → Judge 实际可见的知识点候选集

用法：
  # 直接给 scan_id（推荐，含 OCR 全文）
  python scripts/math_wrong_photo_eval.py \
      --scan-id scan_1788577070114_43537a0d \
      --output /tmp/math_wrong_photo_eval_report.json

  # 给 verify 脚本产物 JSON（自动读取 scan_id / scholar / textbook）
  python scripts/math_wrong_photo_eval.py \
      --result-json /Users/hetao/CodeBuddy/scholar-skill/tmp_media/scan_result.json

  # 只跑 L0/L1 数据门禁 + 结构指标，不调混元（秒级，无 LLM 成本）
  python scripts/math_wrong_photo_eval.py --scan-id scan_xxx --no-judge
  python scripts/math_wrong_photo_eval.py --scan-id scan_xxx --dry-run   # 只打印计划

  # 补评超时/失败的评分面：仅重评指定面，其余面复用既有报告结论（分面重评）
  python scripts/math_wrong_photo_eval.py --scan-id scan_xxx \
      --families judge_classify \
      --merge-existing math_wrong_photo_eval_report.json \
      --output math_wrong_photo_eval_report.json

  # 超时/失败自动重试与单次超时控制（默认重试 2 次、超时取 .env HUNYUAN_TIMEOUT_SECONDS）
  python scripts/math_wrong_photo_eval.py --scan-id scan_xxx --judge-retries 3 --judge-timeout 120

退出码：0=通过（全 PASS/SKIP）；1=数据门禁失败或混元评分未达阈值/部分评分面跳过；2=混元凭据缺失
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    sys.exit("[ERROR] 缺少 openai，请先安装依赖：pip install openai")

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # noqa: E402

from config import (  # noqa: E402
    ENV_ID,
    HUNYUAN_BASE_URL,
    HUNYUAN_EVAL_MODEL,
    HUNYUAN_EVAL_PASS_THRESHOLD,
    HUNYUAN_SECRET_KEY,
    HUNYUAN_TIMEOUT_SECONDS,
    REGION,
    SECRET_ID,
    SECRET_KEY,
    SESSION_TOKEN,
)
from services.database import (  # noqa: E402
    CloudBaseNoSQLClient,
    CURRICULUM_NODE_COLLECTION,
    ERROR_RECORD_COLLECTION,
    MATH_SCAN_UPLOAD_COLLECTION,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("math_wrong_photo_eval")

DEFAULT_SCHOLAR = "6d758f346a6daee000859c332ed11089"
DEFAULT_TEXTBOOK = "tb_math_五年级_up_70963119"
_SCHOLAR_BOOK_COLLECTION = "scholar_book"
_ERROR_TYPES = {"concept", "method", "computation", "reading"}
_CANDIDATE_SELECT = {
    "node_id": 1, "code": 1, "grade": 1, "textbook_id": 1, "ai_summary": 1,
}

# B1.5 图谱外 EXTRA_AI 语义（对齐 error_scanner：textbook_id=EXTRA_AI / node_code=xai_*）
_EXTRA_AI_TEXTBOOK_ID = "EXTRA_AI"
_EXTRA_AI_NODE_CODE_PREFIX = "xai_"
_CLASSIFY_METHOD_AUTO = "auto_scan"
_CLASSIFY_METHOD_MANUAL = "manual_corrected"

# 题号粗略估算（结构指标用，非精确 OCR 判题）
_QN_RE = re.compile(r"(?m)^\s*(?:第\s*)?\d+\s*[.、．)）]")
_QN_RE_2 = re.compile(r"（\s*\d+\s*）|\(\s*\d+\s*\)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="数学错题拍照链路产物 · 混元质量评估（L0 环境 → L1 数据 → L2 结构指标 → L3 混元评分）"
    )
    p.add_argument("--scan-id", default="", help="被评 math_scan_upload.scan_id（与 --result-json 二选一）")
    p.add_argument("--result-json", default="", help="verify 脚本产物 JSON（自动读 scan_id/scholar/textbook）")
    p.add_argument("--scholar-id", default="", help="候选集加载用学者（默认取 scan 记录自身 scholar_id）")
    p.add_argument("--textbook-id", default="", help="仅报告用（默认基线 tb_math_五年级_up_70963119）")
    p.add_argument("--no-judge", action="store_true", help="跳过 L3 混元评分（只跑数据门禁 + 结构指标）")
    p.add_argument("--judge-limit", type=int, default=6, help="混元 Judge 最多调用数（默认 6，本次 3 个评分面）")
    p.add_argument("--concurrency", type=int, default=2, help="混元并发上限（默认 2）")
    p.add_argument("--families", default="",
                   help="只评指定评分面（逗号分隔，如 ocr,judge_classify）；默认全部评分面")
    p.add_argument("--judge-retries", type=int, default=2,
                   help="单个评分面调用失败（超时/网关/解析）的自动重试次数（默认 2）")
    p.add_argument("--judge-timeout", type=float, default=HUNYUAN_TIMEOUT_SECONDS,
                   help=f"单次混元调用超时秒数（默认取 .env HUNYUAN_TIMEOUT_SECONDS={HUNYUAN_TIMEOUT_SECONDS}）")
    p.add_argument("--merge-existing", default="",
                   help="复用既有报告的其余评分面结论（配合 --families 只补评指定面）")
    p.add_argument("--output", default="", help="JSON 报告输出路径（默认当前目录 math_wrong_photo_eval_report.json）")
    p.add_argument("--dry-run", action="store_true", help="只打印计划不触网")
    return p.parse_args()


# ---------------------------------------------------------------------------
# L0/L1：环境与数据门禁
# ---------------------------------------------------------------------------


async def _load_scan(db: CloudBaseNoSQLClient, scan_id: str) -> dict:
    res = await db.query(MATH_SCAN_UPLOAD_COLLECTION, where={"scan_id": scan_id}, limit=1)
    recs = res.get("records") or []
    if not recs:
        raise LookupError(f"math_scan_upload 无此 scan_id: {scan_id}")
    return recs[0]


async def _load_error_records(db: CloudBaseNoSQLClient, scan_id: str) -> list[dict]:
    res = await db.query(
        ERROR_RECORD_COLLECTION,
        where={"scan_upload_id": scan_id},
        order=[{"field": "created_at", "direction": "asc"}],
        limit=200,
    )
    return res.get("records") or []


async def _load_candidates(db: CloudBaseNoSQLClient, scholar_id: str) -> list[dict]:
    """复制 error_scanner._load_knowledge_point_candidates 语义：
    scholar_book → textbook_ids → curriculum_node(ai_summary.status=success).knowledge_points[]。
    """
    res = await db.query(_SCHOLAR_BOOK_COLLECTION, where={"scholar_id": scholar_id}, limit=50)
    tids = [r.get("textbook_id") for r in res.get("records") or [] if r.get("textbook_id")]

    async def _batches(where: dict) -> list[dict]:
        nodes: list[dict] = []
        for offset in range(0, 2000, 500):
            r = await db.query(
                CURRICULUM_NODE_COLLECTION, where=where, select=_CANDIDATE_SELECT,
                offset=offset, limit=500,
            )
            batch = r.get("records") or []
            nodes.extend(batch)
            if len(batch) < 500:
                break
        return nodes

    nodes = await _batches({"textbook_id": {"$in": tids}}) if tids else await _batches({})
    if not nodes and tids:
        nodes = await _batches({})

    cands: list[dict[str, str]] = []
    seen: set[str] = set()
    for node in nodes:
        ai = node.get("ai_summary")
        if not isinstance(ai, dict) or ai.get("status") != "success":
            continue
        for kp in ai.get("knowledge_points") or []:
            name = (kp.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            cands.append({
                "kp_name": name,
                "grade": node.get("grade") or "",
                "node_code": node.get("code") or "",
            })
    return cands


async def load_dataset(scan_id: str, scholar_override: str = "") -> dict:
    db = CloudBaseNoSQLClient(
        env_id=ENV_ID, region=REGION, secret_id=SECRET_ID,
        secret_key=SECRET_KEY, session_token=SESSION_TOKEN,
    )
    scan = await _load_scan(db, scan_id)
    scholar_id = scholar_override or (scan.get("scholar_id") or "").strip()
    records = await _load_error_records(db, scan_id)
    candidates = await _load_candidates(db, scholar_id) if scholar_id else []
    return {
        "db": db,
        "scan": scan,
        "scholar_id": scholar_id,
        "records": records,
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# L2 结构指标（无 LLM 成本）
# ---------------------------------------------------------------------------


def _question_markers(text: str) -> dict:
    """OCR 文本题号痕迹粗估（仅为结构参照，不构成断言）。"""
    m1 = len(_QN_RE.findall(text))
    m2 = len(_QN_RE_2.findall(text))
    return {"line_qn": m1, "paren_qn": m2, "est_total": m1 + m2}


def _anchor_type(rec: dict) -> str:
    """error_record 链锚点类型：extra_ai=课外图谱 EXTRA_AI 节点；formal=教材正式节点；unbound=无锚点。"""
    if not rec:
        return "unbound"
    tb = (rec.get("textbook_id") or "").strip()
    code = (rec.get("node_code") or "").strip()
    if tb == _EXTRA_AI_TEXTBOOK_ID or code.startswith(_EXTRA_AI_NODE_CODE_PREFIX):
        return "extra_ai"
    if code or tb:
        return "formal"
    return "unbound"


def _item_fate(item: dict, rec_by_id: dict[str, dict]) -> tuple[str, dict]:
    """按 error_record 反链判定 classify 项去向（B1.5 口径修正）。

    error_record_id 槽位在 correct 后会被回填（人工修正新建项的 id 也写回空槽），
    空槽/非空槽判「是否 auto 直落」会失真；必须以关联记录的 classify_method 为准。
    fate: auto_direct（classify 直落，含图谱外高置信 EXTRA_AI 直落）/
    manual_corrected（needs_review → correct 人工修正落库）/
    pending_review（仍未落库）/ record_missing（引用悬空，数据反常）。
    """
    rid = (item.get("error_record_id") or "").strip()
    if not rid:
        return "pending_review", {}
    rec = rec_by_id.get(rid)
    if rec is None:
        return "record_missing", {}
    method = (rec.get("classify_method") or "").strip()
    gate = "manual_corrected" if method == _CLASSIFY_METHOD_MANUAL else "auto_direct"
    return gate, rec


def _item_reason(item: dict, rec_by_id: dict[str, dict],
                 candidate_names: set[str]) -> str:
    """给出该项在 classify 门控 + correct 修正后的实际去向说明。"""
    rid = (item.get("error_record_id") or "").strip()
    kp = (item.get("knowledge_point_name") or "").strip()
    conf = float(item.get("confidence") or 0)
    if rid:
        rec = rec_by_id.get(rid)
        if rec is None:
            return f"error_record_id={rid} 悬空(记录缺失，反常)"
        method = (rec.get("classify_method") or "").strip()
        node = rec.get("node_code") or ""
        act = ("auto_scan 直落" if method == _CLASSIFY_METHOD_AUTO
               else "manual_corrected 修正落库")
        rename = ""
        orig = (rec.get("original_kp_name") or "").strip()
        if orig and orig != kp:
            # B1.5a 就近改名：原判名 → 修正为候选标准名挂正式节点
            rename = f"(就近改名 {orig}→{kp})"
        return f"{act}{rename} → {_anchor_type(rec)}({node}) conf={conf:.2f}"
    reasons = []
    if kp:
        reasons.append("命中候选" if kp in candidate_names else "图谱外")
    if conf < 0.6:
        reasons.append(f"conf={conf:.2f}<0.6")
    reasons.append("→ needs_review(未落库)")
    return f"kp={kp or '∅'} " + " & ".join(reasons)


def build_stats(scan: dict, records: list[dict], candidates: list[dict]) -> dict:
    from collections import Counter

    ocr_text = scan.get("ocr_text") or ""
    items = scan.get("classify_result") or []
    cand_names = {c["kp_name"] for c in candidates}
    confs = [float(i.get("confidence") or 0) for i in items]
    rec_by_id = {r.get("record_id"): r for r in records if r.get("record_id")}

    stats: dict[str, Any] = {
        "scan_id": scan.get("scan_id", ""),
        "ocr": {
            "status": scan.get("ocr_status", ""),
            "chars": len(ocr_text),
            "lines": len(ocr_text.splitlines()),
            "markers": _question_markers(ocr_text),
            "blocks": len(scan.get("ocr_blocks") or []),
        },
        "classify": {
            "status": scan.get("classify_status", ""),
            "items_count": len(items),
            "auto_direct": 0,
            "auto_direct_exact": 0,
            "auto_direct_renamed": 0,
            "auto_direct_extra_ai": 0,
            "manual_corrected_items": 0,
            "needs_review_items": 0,
            "record_missing_items": 0,
            "candidate_hits": 0,
            "candidate_total": len(cand_names),
            "error_type_valid": 0,
            "confidence": {
                "min": round(min(confs), 2) if confs else None,
                "max": round(max(confs), 2) if confs else None,
                "mean": round(sum(confs) / len(confs), 3) if confs else None,
                "ge_threshold": sum(1 for c in confs if c >= 0.6),
            },
            "needs_review_all": scan.get("classify_status") == "needs_review",
            "reasons": [],
        },
        "corrected_records": {
            "count": len(records),
            "kp_dist": dict(Counter((r.get("knowledge_point_name") or "").strip() for r in records)),
            "error_type_dist": dict(Counter(r.get("primary_error") or "" for r in records)),
            "method_dist": dict(Counter(r.get("classify_method") or "" for r in records)),
            "kp_in_candidates": sum(
                1 for r in records if (r.get("knowledge_point_name") or "").strip() in cand_names
            ),
        },
    }
    cls = stats["classify"]
    for item in items:
        fate, rec = _item_fate(item, rec_by_id)
        kp = (item.get("knowledge_point_name") or "").strip()
        anchor = _anchor_type(rec) if rec else ""
        if fate == "auto_direct":
            cls["auto_direct"] += 1
            if anchor == "extra_ai":
                cls["auto_direct_extra_ai"] += 1
            elif (rec or {}).get("original_kp_name"):
                # B1.5a：候选外高置信 → 就近改名直落正式候选（record.original_kp_name 留原判名）
                cls["auto_direct_renamed"] += 1
            else:
                cls["auto_direct_exact"] += 1
        elif fate == "manual_corrected":
            cls["manual_corrected_items"] += 1
        elif fate == "record_missing":
            cls["record_missing_items"] += 1
            cls["needs_review_items"] += 1
        else:
            cls["needs_review_items"] += 1
        if kp in cand_names:
            cls["candidate_hits"] += 1
        if item.get("error_type") in _ERROR_TYPES:
            cls["error_type_valid"] += 1
        cls["reasons"].append(_item_reason(item, rec_by_id, cand_names))

    # correct 落库记录统计（B1.5 链锚点：formal=教材正式 / extra_ai=课外图谱）
    auto = [r for r in records if r.get("classify_method") == _CLASSIFY_METHOD_AUTO]
    manual = [r for r in records if r.get("classify_method") == _CLASSIFY_METHOD_MANUAL]
    extra_ai = [r for r in records if _anchor_type(r) == "extra_ai"]
    # 未被任何 classify 项反链引用的记录 → 疑似 correct 未带 error_record_id 的重复新建/遗留
    linked_ids = {
        (i.get("error_record_id") or "").strip() for i in items if i.get("error_record_id")
    }
    orphans = [r for r in records if (r.get("record_id") or "") not in linked_ids]
    stats["corrected_records"].update({
        "auto_scan_count": len(auto),
        "manual_corrected_count": len(manual),
        "extra_ai_count": len(extra_ai),
        "formal_count": len(records) - len(extra_ai),
        "orphan_records": len(orphans),
    })
    return stats


# ---------------------------------------------------------------------------
# L3 混元 Judge（LLM-as-Judge，Judge≠Generator）
# ---------------------------------------------------------------------------

# 评分维度表：每个评分面一个 rubric（权重和 = 1.0）
RUBRICS: dict[str, dict] = {
    "ocr": {
        "system": (
            "你是小学数学错题拍照链路质量评审专家。OCR 文本由视觉大模型从照片转写，"
            "你只能看到文本无法看原图，请基于文本自洽性、结构完整性与下游归类结果回读评估质量。"
            "只输出合法 JSON。"
        ),
        "dimensions": [
            ("题目覆盖完整性", 0.30,
             "文本是否呈现一张完整练习卷的多道独立题（题干+问题齐全），是否疑似漏题、断行丢句或只截到半页"),
            ("数值与单位保真", 0.30,
             "小数、单位换算、价格/评分等关键数字与单位是否完整保留、无串行错位（如 0.85↔85、米↔厘米）"),
            ("结构与可切题性", 0.20,
             "题与题之间是否有明确边界（题号/换行），能否据此按题切分给下游归类"),
            ("文本自洽性", 0.20,
             "无明显乱码、重复粘贴、上下文断裂或把多页内容混在一起"),
        ],
    },
    "judge_classify": {
        "system": (
            "你是小学数学错题「Judge 自动归类」质量评审专家。Judge 依据 OCR 全文与教材知识点候选集"
            "做题目切分+知识点定位+错因四分类+置信度标注；你只做独立质量评分。"
            "门控已对齐 B1.5a：知识点在候选集外但题干语义明确、conf≥0.6 的题可高置信直落——"
            "若存在语义就近的正式候选（共享核心主题词）→ 就近改名直落正式节点"
            "（item.original_kp_name 为该题原判名，改名目标即落库名，如「小数乘法的实际应用」"
            "改为候选「小数乘除的实际应用」）；没有任何就近候选 → EXTRA_AI 课外图谱新建。"
            "两条改名护栏：操作符互斥（加减乘除不同族，如「加法…」vs「减法…」一字差）与"
            "仅共享泛化模板后缀（如「的实际应用」但主题词无关）的近似一律不得改名——"
            "被护栏拦下而保持 EXTRA_AI 不算漏改名。两者皆属正常直落，不是门控失效；"
            "低置信(conf<0.6)/无错因/无名字的题才进 needs_review，随后人工修正落库亦属正常闭环。"
            "只输出合法 JSON。"
        ),
        "dimensions": [
            ("切题完整性", 0.20,
             "Judge 切出的题目数/题面与 OCR 可切分的独立题是否对应：无漏切、无臆造、无把口算一题拆多题"),
            ("知识点定位正确性", 0.30,
             "各项 knowledge_point_name 与对应题干的语义匹配度——命中候选的应为正式节点语义；"
             "候选集外自拟名只要与题干语义吻合即为合格（下游会 EXTRA_AI 新建或就近修正命名），"
             "仅当明显错挂/张冠李戴（如把单位换算挂成小数加减）才扣分。"
             "对带 original_kp_name 的项（候选外原判名→就近改名标准名，如 小数乘法的实际应用→"
             "小数乘除的实际应用）：改名目标应与题干主题贴切、且不得跨互斥运算（把乘法题改到"
             "减法类候选即错改名）；两可时选语义更贴的一方"),
            ("错因四分类合理", 0.20,
             "error_type ∈ concept/method/computation/reading 是否贴合题意与错因语境"),
            ("置信度自洽", 0.15,
             "confidence 与匹配明确度自洽：命中候选且语义明确应 ≥0.6；候选外但题干语义明确允许 ≥0.6"
             "（B1.5 → EXTRA_AI 直落）；候选外且语义模糊/错因不明应 <0.6"),
            ("needs_review 决策正确", 0.15,
             "对照每项标注 gate（auto_direct/manual_corrected/pending_review，已按 "
             "error_record.classify_method 反链给出）：conf≥0.6 的命中候选或语义明确候选外题应直落"
             "（B1.5a 直落含三种：精确命中 / 就近改名挂正式（anchor=formal 且带 original_kp_name）/ "
             "EXTRA_AI（anchor=extra_ai），均非门控失效）；低置信题应 needs_review，其后人工修正"
             "落库不算门控失效；不要误把就近改名直落或图谱外 EXTRA_AI 直落当门控失效"),
        ],
    },
    "manual_correct": {
        "system": (
            "你是小学数学错题「人工修正归链（correct）」质量评审专家。correct 处理 Judge 判 needs_review 的项"
            "（低置信/图谱外），按 B1.5 锚点决策链落库 error_record：1) 图谱外新知识点 → EXTRA_AI 课外图谱"
            "幂等新建/复用节点（正确行为，不是漏改挂）；2) 候选内精确命中 → 锚定正式节点标准名；"
            "3) 候选内就近命名（共享核心主题词）→ 修正为候选标准名再挂正式节点；4) 确实不在图谱"
            "且无近似候选才 EXTRA_AI。错误行为是：把图谱外知识点机械改挂到语义无关的正式节点、或复制不改名。"
            "B1.5a：classify 直落已自动就近改名（record.original_kp_name 非空即此类，原判名→标准名），"
            "本面评审 full records 时须一并核对改名目标贴切性。两条改名护栏须遵守——操作符互斥"
            "（加减乘除不同族，如「加法…」vs「减法…」）或仅共享泛化模板后缀（如「的实际应用」但主题词无关）"
            "的近似不得改名，被护栏拦下走 EXTRA_AI 属正确而非漏改名。"
            "每条 error_record 已冗余 textbook_id/node_code/node_title/original_kp_name 链锚点"
            "（anchor=formal/extra_ai），直接据此判。只输出合法 JSON。"
        ),
        "dimensions": [
            ("归链语义贴切度", 0.30,
             "最终锚点（正式节点 code 或 EXTRA_AI 节点）与题面语义是否贴切：正式节点应语义吻合；"
             "EXTRA_AI 新建名应是对应题干真实的知识点（课外补充），而非臆造或照抄整句题干"),
            ("图谱外识别与 EXTRA_AI 恰当性", 0.35,
             "逐题复判锚点去向：该 EXTRA_AI 的（图谱外）确实 EXTRA_AI；该就近修正为正式标准名的"
             "（候选内近似命名，含 auto 直落带 original_kp_name 的）确实改挂正式节点且改名目标"
             "与题干主题贴切（不得跨互斥运算、不得仅因共享「的实际应用」等模板后缀而改名）；"
             "两可时选语义更贴的一方；机械乱挂/漏处理才扣分"),
            ("错因保真", 0.15,
             "修正后 error_type 与原题错因是否保持一致且合理（correct 未改错因时是否有错因本身不合理）"),
            ("重复与粒度", 0.20,
             "落库条数与题数匹配、无重复新建（同知识点多题复用同一节点属正常聚合，不扣分）；"
             "多道不同题被压到同一节点致粒度失真才扣分"),
        ],
    },
}

_JUDGE_USER_TEMPLATE = """请评估以下数学错题拍照链路产物。

评分面：{interface_label}
产物背景：{request_summary}
产物数据（JSON）：
{output_json}

评分维度（每维打分 0.0~1.0，可只输出各维度分，脚本按权重自动算总分）：
{dimensions}

输出 JSON（不要任何解释/markdown）：
{{"score": 0.0~1.0, "dimensions": [{{"name": "维度名", "score": 0.0, "comment": "一句评价"}}], "feedback": "总体评价与改进建议", "issues": ["问题1"]}}"""


def _parse_eval_response(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def _weighted_score(result: dict, dims: list[tuple]) -> float | None:
    """按维度名匹配权重加权；匹配不到时回退模型自报 score。"""
    wmap = {name: w for name, w, _desc in dims}
    dim_scores = {d.get("name"): d.get("score") for d in result.get("dimensions") or []}
    hits = [(wmap[n], float(v)) for n, v in dim_scores.items()
            if n in wmap and isinstance(v, (int, float))]
    if len(hits) == len(wmap) and all(0 <= v <= 1 for _w, v in hits):
        return round(sum(w * v for w, v in hits), 3)
    return None


async def _judge_one(artifact: dict, sem: asyncio.Semaphore,
                     timeout: float | None = None) -> dict:
    family = artifact["family"]
    rubric = RUBRICS[family]
    dim_lines = "\n".join(
        f"- {name}（权重 {w}）：{desc}" for name, w, desc in rubric["dimensions"]
    )
    prompt = _JUDGE_USER_TEMPLATE.format(
        interface_label=artifact["interface_label"],
        request_summary=artifact["request_summary"],
        output_json=json.dumps(artifact["output"], ensure_ascii=False),
        dimensions=dim_lines,
    )
    try:
        async with sem:
            client = AsyncOpenAI(
                api_key=HUNYUAN_SECRET_KEY,
                base_url=HUNYUAN_BASE_URL,
                timeout=timeout if timeout else HUNYUAN_TIMEOUT_SECONDS,
                max_retries=0,
            )
            resp = await client.chat.completions.create(
                model=HUNYUAN_EVAL_MODEL,
                messages=[
                    {"role": "system", "content": rubric["system"]},
                    {"role": "user", "content": prompt},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise ValueError(
                    f"Judge 响应 content 为空（model={HUNYUAN_EVAL_MODEL}，"
                    f"base_url={HUNYUAN_BASE_URL} 须为 OpenAI 兼容 /chat/completions 网关）"
                )
            result = _parse_eval_response(content)
            ws = _weighted_score(result, rubric["dimensions"])
            score = ws if ws is not None else max(0.0, min(1.0, float(result.get("score", -1.0))))
            return {
                "score": score,
                "feedback": str(result.get("feedback", "")),
                "issues": result.get("issues", []),
                "dimensions": result.get("dimensions", []),
            }
    except Exception as e:  # noqa: BLE001
        logger.warning(f"混元评估失败，降级跳过: {type(e).__name__}: {e}")
        return {"score": -1.0, "feedback": f"评估跳过: {e}", "issues": [], "dimensions": []}


async def _judge_all(tasks: list) -> list:
    return await asyncio.gather(*tasks)


def _parse_families(raw: str) -> list[str]:
    """解析 --families；为空返回全部评分面（按 RUBRICS 定义序）。"""
    if not raw or not raw.strip():
        return list(RUBRICS.keys())
    fams = [f.strip() for f in raw.split(",") if f.strip() in RUBRICS]
    if not fams:
        logger.warning("--families=%r 无合法评分面（可选: %s），按全部处理",
                       raw, ",".join(RUBRICS.keys()))
        return list(RUBRICS.keys())
    return fams


async def _judge_with_retry(artifact: dict, sem: asyncio.Semaphore,
                            timeout: float, retries: int,
                            delay: float = 3.0) -> dict:
    """单个评分面：失败（超时/网关/解析）自动重试 retries 次，带退避。

    全部重试仍失败时返回 score=-1（JUDGE_SKIPPED），由上层决定是否并入报告。
    """
    last: dict = {"score": -1.0, "feedback": "未执行", "issues": [], "dimensions": []}
    for attempt in range(retries + 1):
        res = await _judge_one(artifact, sem, timeout)
        if res["score"] >= 0:
            return res
        last = res
        if attempt < retries:
            wait = delay * (attempt + 1)
            logger.warning(
                "[%s] 第 %d/%d 次尝试失败（%.80s），%.0fs 后重试",
                artifact["family"], attempt + 1, retries,
                str(res.get("feedback") or type(res).__name__), wait,
            )
            await asyncio.sleep(wait)
    return last


def build_artifacts(scan: dict, records: list[dict], candidates: list[dict]) -> list[dict]:
    ocr_text = scan.get("ocr_text") or ""
    items = scan.get("classify_result") or []
    cand_names = [c["kp_name"] for c in candidates]
    cand_set = set(cand_names)
    short = ocr_text if len(ocr_text) <= 2200 else ocr_text[:2200] + "\n…[OCR 截断]"

    # 反链口径（B1.5）：item 去向以 error_record.classify_method 为准，不用 error_record_id 槽位
    # （correct 会把空槽回填为 manual 记录 id，空槽判 auto 会失真）。
    rec_by_id = {r.get("record_id"): r for r in records if r.get("record_id")}
    linked: list[tuple[dict, str, dict]] = []
    for i in items:
        gate, rec = _item_fate(i, rec_by_id)
        linked.append((i, gate, rec))
    auto_n = sum(1 for _i, g, _r in linked if g == "auto_direct")
    review_n = len(items) - auto_n
    manual_n = sum(1 for r in records if r.get("classify_method") == _CLASSIFY_METHOD_MANUAL)
    auto_recs = [r for r in records if r.get("classify_method") == _CLASSIFY_METHOD_AUTO]
    auto_rec_n = len(auto_recs)
    # B1.5a 三态：exact（精确命中）/ renamed（就近改名挂正式，record 带 original_kp_name）/
    # extra_ai（图谱外新建）。original_kp_name 仅近改名分支写入 → 可直接数出来。
    auto_renamed_n = sum(1 for r in auto_recs if (r.get("original_kp_name") or "").strip())
    auto_extra_n = sum(1 for r in auto_recs if _anchor_type(r) == "extra_ai")
    auto_exact_n = auto_rec_n - auto_renamed_n - auto_extra_n
    extra_ai_n = sum(1 for r in records if _anchor_type(r) == "extra_ai")
    formal_n = len(records) - extra_ai_n

    def _item_row(i: dict, gate: str, rec: dict) -> dict:
        row = {k: i.get(k) for k in ("knowledge_point_name", "error_type", "confidence", "error_record_id")}
        row["question_text"] = (i.get("question_text") or "").strip()[:90]
        row["gate"] = gate
        row["anchor"] = _anchor_type(rec) if rec else ""
        # B1.5a 就近改名：record.original_kp_name 为唯一可靠通道（classify_result 项
        # 里存的已是改名后标准名；老数据该字段缺失则无法识别，退化为精确命中口径）
        orig = (rec or {}).get("original_kp_name") or ""
        if orig:
            row["original_kp_name"] = orig
        return row

    def _record_row(r: dict) -> dict:
        kp = (r.get("knowledge_point_name") or "").strip()
        return {
            "record_id": r.get("record_id", ""),
            "knowledge_point_name": kp,
            "original_kp_name": (r.get("original_kp_name") or "").strip(),
            "error_type": r.get("primary_error", ""),
            "classify_method": r.get("classify_method", ""),
            "textbook_id": r.get("textbook_id", ""),
            "node_code": r.get("node_code", ""),
            "node_title": r.get("node_title", ""),
            "anchor": _anchor_type(r),
            "kp_in_candidates": kp in cand_set,
            "question_text": (r.get("question_text") or "").strip()[:90],
        }

    return [
        {
            "family": "ocr",
            "interface_label": "OCR 文本质量（OCR 文本 → math_scan_upload.ocr_text）",
            "request_summary": (
                f"来源: 五年级上册练习二照片一张（ocr_provider={scan.get('ocr_provider') or 'vision-dev'}）；"
                f"ocr_status={scan.get('ocr_status')}；下游 Judge 据此切出 {len(items)} 道题；下方为 OCR 全文。"
            ),
            "output": {"ocr_text": short},
        },
        {
            "family": "judge_classify",
            "interface_label": "Judge 自动归类（题目切分+知识点定位+错因+置信度）",
            "request_summary": (
                f"输入候选知识点 {len(cand_names)} 个（学者 {scan.get('scholar_id') or ''} 教材 F1）；"
                f"Judge 输出 {len(items)} 项：{auto_n} 项 conf≥0.6 已直落 error_record"
                f"（B1.5a 三态：精确命中={auto_exact_n} / 就近改名挂正式={auto_renamed_n}"
                f"（items.original_kp_name 为原判名，如 小数乘法的实际应用→小数乘除的实际应用）/ "
                f"图谱外高置信→EXTRA_AI 自动新建={auto_extra_n}，见 items.anchor），"
                f"{review_n} 项低置信/未定位 → needs_review（其中 {manual_n} 条已由 correct 人工修正落库）。"
                f"items 已按 error_record.classify_method 反链标注 gate"
                f"（auto_direct/manual_corrected/pending_review），请以此核对门控而非 error_record_id 槽位。"
                f"classify_status={scan.get('classify_status')}（correct 后置 success）；"
                f"item 已带 question_text（B1 题干），可结合题干逐题语义比对。"
            ),
            "output": {
                "candidates": cand_names,
                "items": [_item_row(i, g, r) for i, g, r in linked],
                "ocr_text": short,
            },
        },
        {
            "family": "manual_correct",
            "interface_label": "correct 人工修正归链（落库 error_record，B1.5 锚点决策链）",
            "request_summary": (
                f"correct 处理 {review_n} 个 needs_review 项（{auto_n} 项已 auto 直落未参与；"
                f"其中直落三态 exact={auto_exact_n} / 就近改名挂正式={auto_renamed_n}"
                f"（final_records.original_kp_name 为原判名，如 小数乘法的实际应用→小数乘除的实际应用）/ "
                f"EXTRA_AI={auto_extra_n}）；"
                f"DB 现有关联该 scan 的 error_record 共 {len(records)} 条"
                f"（auto_scan={auto_rec_n} + manual_corrected={manual_n}；"
                f"锚点 formal={formal_n} / extra_ai={extra_ai_n}）。"
                f"final_records 已带 textbook_id/node_code/node_title/anchor/original_kp_name"
                f"（formal=教材正式节点，extra_ai=课外图谱 EXTRA_AI 自动新建/复用；"
                f"original_kp_name 非空 = classify 直落就近改名，须核对改名目标与题干语义贴切性），"
                f"请按 B1.5 归链语义评估（图谱外 → EXTRA_AI 属正确，机械改挂正式节点才扣分）。"
            ),
            "output": {
                "candidates": cand_names,
                "original_judge_items": [_item_row(i, g, r) for i, g, r in linked],
                "final_records": [_record_row(r) for r in records],
                "ocr_text": short,
            },
        },
    ]


def print_report(report: dict) -> None:
    print("\n\n===== 评估汇总 =====")
    print("[L1 数据门禁]")
    for a in report["availability"]:
        mark = "✓" if a["available"] else "✗"
        print(f"  {mark} {a['label']}: {a['detail']}")
    print("[L2 结构指标]")
    stats = report["stats"]
    print(f"  OCR: status={stats['ocr']['status']} chars={stats['ocr']['chars']} "
          f"题号痕迹={stats['ocr']['markers']['est_total']}")
    cls = stats["classify"]
    print(f"  classify: status={cls['status']} items={cls['items_count']} "
          f"auto直落={cls['auto_direct']}(精确={cls['auto_direct_exact']} "
          f"就近改名={cls['auto_direct_renamed']} 图谱外EXTRA_AI={cls['auto_direct_extra_ai']}) "
          f"manual修正={cls['manual_corrected_items']} 待处理={cls['needs_review_items']} "
          f"候选命中={cls['candidate_hits']}/{cls['candidate_total']} "
          f"conf均值={cls['confidence']['mean']}")
    cr = stats["corrected_records"]
    print(f"  correct 落库: {cr['count']} 条 (auto_scan={cr['auto_scan_count']} "
          f"manual={cr['manual_corrected_count']} 锚点 formal={cr['formal_count']}/"
          f"EXTRA_AI={cr['extra_ai_count']} 悬空={cr['orphan_records']}) "
          f"kp分布={cr['kp_dist']}")
    print("[L3 混元质量]")
    for j in report["judgments"]:
        s = j["score"]
        if s < 0:
            mark, label = "⚠", "JUDGE_SKIPPED"
        else:
            mark = "✓" if s >= HUNYUAN_EVAL_PASS_THRESHOLD else "✗"
            label = f"score={s:.2f} (阈值 {HUNYUAN_EVAL_PASS_THRESHOLD})"
        print(f"  {mark} [{j['family']}] {j['interface_label']}: {label}")
        for d in j.get("dimensions") or []:
            print(f"      · {d.get('name')}: {d.get('score')} — {d.get('comment')}")
        if j.get("issues"):
            print(f"      issues: {'; '.join(str(x) for x in j['issues'][:6])}")
    if report["judgments"]:
        scores = [j["score"] for j in report["judgments"] if j["score"] >= 0]
        if scores:
            print(f"  → 被评 {len(scores)} 个产物平均分 = {sum(scores) / len(scores):.2f}")
    print(f"\n[结论] {'PASS' if report['pass'] else 'FAIL'}")
    for reason in report["reasons"]:
        print(f"  - {reason}")


def main() -> int:
    args = parse_args()
    scan_id = args.scan_id
    scholar_id = args.scholar_id
    textbook_id = args.textbook_id or DEFAULT_TEXTBOOK
    result_meta: dict = {"scholar_id": scholar_id, "textbook_id": textbook_id}

    if not scan_id and args.result_json:
        rp = Path(args.result_json)
        if not rp.is_file():
            logger.error("--result-json 文件不存在: %s", rp)
            return 1
        data = json.loads(rp.read_text(encoding="utf-8"))
        scan_id = (data.get("seed") or {}).get("scan_id") or ""
        scholar_id = scholar_id or data.get("scholar_id") or ""
        textbook_id = textbook_id or data.get("textbook_id") or textbook_id
        result_meta = {"result_json": str(rp), "scholar_id": scholar_id,
                       "textbook_id": textbook_id}
    if not scan_id:
        logger.error("缺少被评对象：请传 --scan-id 或 --result-json")
        return 1

    logger.info("计划: scan_id=%s scholar=%s textbook=%s judge_model=%s no_judge=%s",
                scan_id, scholar_id or "(取 scan)", textbook_id,
                HUNYUAN_EVAL_MODEL, args.no_judge)
    if args.dry_run:
        logger.info("--dry-run：仅打印计划，退出 0")
        return 0

    report: dict = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "judge_model": HUNYUAN_EVAL_MODEL,
            "pass_threshold": HUNYUAN_EVAL_PASS_THRESHOLD,
            "scan_id": scan_id,
            **result_meta,
            "source_docs": [
                "scholar-skill/docs_v1/待确认/数学错题拍照设计脚本-测试脚本设计与实现.md",
                "scholar-skill/docs_v1/待确认/数学错题拍照设计脚本-教材链条归集与同题型出题.md",
            ],
        },
        "availability": [],
        "stats": {},
        "judgments": [],
        "summary": {},
        "pass": False,
        "reasons": [],
    }
    reasons: list[str] = []
    hunyuan_ok = bool(HUNYUAN_SECRET_KEY and HUNYUAN_EVAL_MODEL)

    print("===== [L0 环境门禁] =====")
    print(f"  混元 Judge: HUNYUAN_EVAL_MODEL={HUNYUAN_EVAL_MODEL or '(未配置)'} | "
          f"SECRET_KEY={'已配置' if HUNYUAN_SECRET_KEY else '未配置'} | "
          f"PASS_THRESHOLD={HUNYUAN_EVAL_PASS_THRESHOLD}")
    print(f"  DB: ENV_ID={ENV_ID} SECRET_ID={'已配置' if SECRET_ID else '未配置'}")
    if not hunyuan_ok and not args.no_judge:
        print("[WARN] 混元凭据未配置（HUNYUAN_SECRET_KEY/HUNYUAN_EVAL_MODEL），L3 将全部 JUDGE_SKIPPED")
    if HUNYUAN_BASE_URL.rstrip("/").endswith("hunyuan.tencentcloudapi.com"):
        print("[WARN] HUNYUAN_BASE_URL 仍为腾讯云 OpenAPI 默认值，不兼容 Bearer /chat/completions 网关，"
              "L3 将全部跳过。请在 .env 设 HUNYUAN_BASE_URL=https://tokenhub.tencentmaas.com/v1")
    if not SECRET_ID:
        print("[WARN] TENCENTCLOUD_SECRETID 未配置，无法读 DB，后续门禁将 FAIL")

    print("\n===== [L1 数据门禁] =====")
    dataset: dict = {}
    try:
        dataset = asyncio.run(load_dataset(scan_id, scholar_override=scholar_id))
    except LookupError as e:
        reasons.append(str(e))
    except Exception as e:  # noqa: BLE001
        reasons.append(f"DB 读取失败: {type(e).__name__}: {e}")

    if dataset:
        scan = dataset["scan"]
        records = dataset["records"]
        candidates = dataset["candidates"]
        scholar_id = dataset["scholar_id"] or scholar_id
        report["meta"]["scholar_id"] = scholar_id
        gates = [
            ("scan 记录读取", True, f"scan_id={scan.get('scan_id')} ocr_status={scan.get('ocr_status')} "
             f"classify_status={scan.get('classify_status')} ocr_chars={len(scan.get('ocr_text') or '')}"),
            ("OCR 就绪", scan.get("ocr_status") == "success", "ocr_status=success"),
            ("classify 有产物", bool(scan.get("classify_result")), f"items={len(scan.get('classify_result') or [])}"),
            ("落库 error_record 读取", True, f"count={len(records)}"),
            ("候选知识点加载", bool(candidates), f"candidates={len(candidates)}"),
        ]
        for label, ok, detail in gates:
            report["availability"].append({"label": label, "available": ok, "detail": detail})
            mark = "✓" if ok else "✗"
            print(f"  {mark} {label}: {detail}")
            if not ok:
                reasons.append(f"数据门禁失败: {label} — {detail}")
        report["stats"] = build_stats(scan, records, candidates)
    else:
        report["availability"].append({"label": "数据集加载", "available": False, "detail": "DB 读取失败"})

    print("\n===== [L2 结构指标] =====")
    if report["stats"]:
        s = report["stats"]
        print(f"  OCR status={s['ocr']['status']} chars={s['ocr']['chars']} lines={s['ocr']['lines']} "
              f"题号痕迹≈{s['ocr']['markers']['est_total']} blocks={s['ocr']['blocks']}")
        c = s["classify"]
        print(f"  classify status={c['status']} items={c['items_count']} "
              f"auto直落={c['auto_direct']}(精确={c['auto_direct_exact']} "
              f"就近改名={c['auto_direct_renamed']} 图谱外={c['auto_direct_extra_ai']}) "
              f"manual修正={c['manual_corrected_items']} 待处理={c['needs_review_items']} "
              f"候选命中={c['candidate_hits']} conf=[{c['confidence']['min']},"
              f"{c['confidence']['max']}] mean={c['confidence']['mean']}")
        for i, r_ in enumerate(s["classify"]["reasons"]):
            print(f"    item{i}: {r_}")
        cr = s["corrected_records"]
        print(f"  correct 落库 {cr['count']} 条 "
              f"(auto_scan={cr['auto_scan_count']} manual_corrected={cr['manual_corrected_count']} "
              f"锚点 formal={cr['formal_count']}/EXTRA_AI={cr['extra_ai_count']} "
              f"悬空={cr['orphan_records']})")

    print("\n===== [L3 混元质量] =====")
    if not args.no_judge and dataset and report["stats"]:
        report["meta"].update({
            "judge_retries": args.judge_retries,
            "judge_timeout": args.judge_timeout,
        })
        artifacts = build_artifacts(scan, records, candidates)[: args.judge_limit]
        selected = _parse_families(args.families)
        logger.info("L3 评分面: 本次重评=%s", selected or "（空）")

        # 分面重评：--merge-existing 的既有结论中，未被本次 --families 覆盖的面原样保留
        merged: dict[str, dict] = {}
        if args.merge_existing:
            mp = Path(args.merge_existing)
            if mp.is_file():
                old = json.loads(mp.read_text(encoding="utf-8"))
                for j in old.get("judgments") or []:
                    if j.get("family") not in selected:
                        merged[j.get("family")] = j
                report["meta"]["merged_from"] = str(mp)
                logger.info("L3 复用既有报告 %d 个评分面（未在本次 --families 中）: %s",
                            len(merged), sorted(merged.keys()))
            else:
                logger.warning("--merge-existing 文件不存在，忽略: %s", mp)

        to_judge = [a for a in artifacts if a["family"] in selected]
        if to_judge:
            sem = asyncio.Semaphore(args.concurrency)

            async def _run_judges() -> list[dict]:
                tasks = [_judge_with_retry(a, sem, args.judge_timeout,
                                           args.judge_retries) for a in to_judge]
                return await _judge_all(tasks)

            judged_new = asyncio.run(_run_judges())
        else:
            judged_new = []

        # 按 RUBRICS 定义序组装最终 judgments（本次重评 > 合并保留）
        fresh = {a["family"]: j for a, j in zip(to_judge, judged_new)}
        final_judgments: list[dict] = []
        for a in artifacts:
            fam = a["family"]
            if fam in fresh:
                final_judgments.append(
                    {"family": fam, "interface_label": a["interface_label"], **fresh[fam]})
            elif fam in merged:
                final_judgments.append(merged[fam])
        report["judgments"] = final_judgments

        skipped = [j for j in report["judgments"] if j["score"] < 0]
        judged = [j for j in report["judgments"] if j["score"] >= 0]
        low = [j for j in judged if j["score"] < HUNYUAN_EVAL_PASS_THRESHOLD]
        for j in low:
            reasons.append(
                f"混元评分未达标: [{j['family']}] score={j['score']:.2f} "
                f"< {HUNYUAN_EVAL_PASS_THRESHOLD} | {str(j['feedback'])[:160]}")
        for j in skipped:
            reasons.append(
                f"混元评分跳过: [{j['family']}] "
                f"{str(j.get('feedback') or '')[:160] or '未知原因（多次重试仍失败）'}")
        if skipped and not judged:
            reasons.append(
                "混元 Judge 全部跳过（凭据/网关/超时），请人工复核或检查 HUNYUAN_* 配置")
    elif args.no_judge:
        print("  --no-judge：跳过混元评分")

    judged = [j for j in report["judgments"] if j["score"] >= 0]
    report["summary"] = {
        "data_gate_ok": all(a["available"] for a in report["availability"]),
        "judge_count": len(report["judgments"]),
        "judge_pass_count": sum(1 for j in judged if j["score"] >= HUNYUAN_EVAL_PASS_THRESHOLD),
    }
    report["pass"] = not reasons
    report["reasons"] = reasons

    out_path = Path(args.output) if args.output else HERE / "math_wrong_photo_eval_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[报告] 已写入 {out_path}")

    print_report(report)

    if report["pass"]:
        return 0
    if not hunyuan_ok and not args.no_judge:
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
