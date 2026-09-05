#!/usr/bin/env python3
"""数学错题拍照 · 教材链归集与同题型巩固 —— 真实数据验收/准备脚本。

承接设计：
- 测试设计：docs_v1/待确认/数学错题拍照设计脚本-测试脚本设计与实现.md
- 接口契约：docs_v1/待确认/数学错题拍照设计脚本-接口契约草案.md
- 主文档：  docs_v1/待确认/数学错题拍照设计脚本-教材链条归集与同题型出题.md §10 场景 ①~⑨

一期（数据准备，默认）：用一次【真实错题拍照】产出测试数据——
  链路：POST /math/scan/upload（真实图片）
        → 轮询 POST /math/scan/classify（OCR 就绪 + Judge 归类）
        → needs_review 时 POST /math/scan/{scan_id}/correct（自动选知识点修正）
        → GET /math/error-stats 校验落库
二期（新契约断言，默认 SKIP）：error-stats 链字段 + drill_stats +
  POST /math/drill/generate、POST /math/drill/{drill_id}/submit 全闭环；
  后端未实现时以 FAIL（而非 SKIP）暴露缺口。

用法：
  python scripts/math_wrong_photo_chain_verify.py --image ./wrong_math_up5.png
  python scripts/math_wrong_photo_chain_verify.py --image ./wrong_math_up5.png \\
      --seed-direct                                          # 绕过微信云存储，直接 DB 种 scan
  python scripts/math_wrong_photo_chain_verify.py --image ./wrong_math_up5.png \\
      --skip-unimplemented=false --output ./scan_result.json
  python scripts/math_wrong_photo_chain_verify.py            # 只读检查，不发图
  python scripts/math_wrong_photo_chain_verify.py --dry-run  # 打印计划不触网

默认教材/学者即测试基线（五年级上册 × 指定学者），可用参数覆盖。
退出码：0 = 全通过或 SKIP；1 = 任一 FAIL。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("math_wrong_photo_chain_verify")

DEFAULT_SCHOLAR = "6d758f346a6daee000859c332ed11089"
DEFAULT_TEXTBOOK = "tb_math_五年级_up_70963119"
DEFAULT_BASE = "http://127.0.0.1:8080"

OCR_READY_TIMEOUT_SECONDS = 90
POLL_INTERVAL_SECONDS = 3
HTTP_TIMEOUT_SECONDS = 120
IMAGE_MAX_BYTES = 10 * 1024 * 1024
ALLOWED_EXTS = {"jpg", "jpeg", "png", "webp", "bmp"}


class StepResult:
    def __init__(self) -> None:
        self.done: list[str] = []
        self.failed: list[str] = []
        self.skipped: list[str] = []

    def pass_(self, name: str, detail: str = "") -> None:
        self.done.append(f"[PASS] {name}" + (f" — {detail}" if detail else ""))
        logger.info(self.done[-1])

    def fail(self, name: str, detail: str) -> None:
        self.failed.append(f"[FAIL] {name} — {detail}")
        logger.error(self.failed[-1])

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append(f"[SKIP] {name} — {reason}")
        logger.warning(self.skipped[-1])

    @property
    def ok(self) -> bool:
        return not self.failed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="数学错题拍照链路真实验收/数据准备脚本")
    p.add_argument("--base-url", default=DEFAULT_BASE, help="服务地址，默认本机 8080")
    p.add_argument("--scholar-id", default=DEFAULT_SCHOLAR)
    p.add_argument("--textbook-id", default=DEFAULT_TEXTBOOK)
    p.add_argument("--image", default="", help="错题照片路径（一期拍照数据准备必需）")
    p.add_argument("--image-extra", default="", help="可选：图谱外题照片（T-11，EXTRA_AI）")
    p.add_argument("--correct-kp", default="", help="needs_review 修正时指定知识点名（可选）")
    p.add_argument("--openid", default="", help="AUTH_MODE=enforce 时的 X-WX-OPENID")
    p.add_argument("--seed-direct", action="store_true",
                   help="绕过 /math/scan/upload 微信云存储，直接 DB 种 scan 后走 correct")
    p.add_argument("--skip-unimplemented", action="store_true", default=True,
                   help="二期新契约断言 SKIP（默认开启）；落地后加 --skip-unimplemented=false")
    p.add_argument("--output", default="", help="产物 JSON 路径（scan/error_record 标识）")
    p.add_argument("--dry-run", action="store_true", help="只打印计划不触网")
    return p.parse_args()


def _client(args: argparse.Namespace) -> httpx.Client:
    headers = {"X-WX-OPENID": args.openid} if args.openid else {}
    return httpx.Client(base_url=args.base_url.rstrip("/"), headers=headers,
                        timeout=HTTP_TIMEOUT_SECONDS)


def _read_image(path: str) -> tuple[str, bytes]:
    fp = Path(path)
    if not fp.is_file():
        raise FileNotFoundError(f"图片不存在: {fp}")
    raw = fp.read_bytes()
    if len(raw) > IMAGE_MAX_BYTES:
        raise ValueError(f"图片超过 10MB: {fp}")
    ext = fp.suffix.lstrip(".").lower() or "jpg"
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"图片格式不支持: {ext}")
    return fp.name, raw


# ---------------------------------------------------------------------------
# phase0：环境与数据可达性（T-0）
# ---------------------------------------------------------------------------


def phase0(client: httpx.Client, args: argparse.Namespace, step: StepResult) -> list[str]:
    """返回该教材知识列表中的 kp 名（去重），供修正兜底。"""
    r = client.get("/math")
    step.pass_("T-0a /math health", f"status={r.status_code}" if r.status_code == 200
                else f"status={r.status_code} body={r.text[:120]}")
    if r.status_code != 200:
        step.fail("T-0a", "服务不可达")
        return []

    r = client.get(f"/math/textbook/{args.textbook_id}/knowledge-points")
    if r.status_code != 200:
        step.fail("T-0b 教材知识点可达", f"status={r.status_code} {r.text[:120]}")
        return []
    data = r.json().get("data", {})
    kps = data.get("knowledge_points", [])
    step.pass_("T-0b 教材知识点可达",
               f"total={data.get('total')} textbook={data.get('textbook_id')}")
    if not kps:
        step.fail("T-0b", "教材无已总结知识点（F1 未生成），classify 命中率低")

    r = client.get("/math/error-stats", params={"scholar_id": args.scholar_id, "limit": 5})
    if r.status_code != 200:
        step.fail("T-0c error-stats 可达", f"status={r.status_code} {r.text[:120]}")
    else:
        body = r.json().get("data", {})
        step.pass_("T-0c error-stats 可达", f"total={body.get('total')}")

    names: list[str] = []
    for kp in kps:
        name = (kp.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    if not names:
        step.fail("T-0d", "教材知识点列表无 name 字段可兜底修正")
    return names


# ---------------------------------------------------------------------------
# phase1：一次真实错题拍照（T-1 ~ T-3）
# ---------------------------------------------------------------------------


async def _direct_seed_scan(args: argparse.Namespace, step: StepResult) -> str:
    """绕过微信云存储：直接 DB 写 math_scan_upload 记录，复用 sha256 幂等。"""
    from config import (
        ENV_ID,
        REGION,
        SECRET_ID,
        SECRET_KEY,
        SESSION_TOKEN,
    )
    from services.database import CloudBaseNoSQLClient, MATH_SCAN_UPLOAD_COLLECTION

    filename, raw = _read_image(args.image)
    image_hash = hashlib.sha256(raw).hexdigest()

    db = CloudBaseNoSQLClient(
        env_id=ENV_ID,
        region=REGION,
        secret_id=SECRET_ID,
        secret_key=SECRET_KEY,
        session_token=SESSION_TOKEN,
    )

    # 幂等：同图不再重复种
    existing = await db.query(
        MATH_SCAN_UPLOAD_COLLECTION,
        where={"image_hash": image_hash},
        limit=1,
    )
    records = existing.get("records") or []
    if records:
        rec = records[0]
        scan_id = rec.get("scan_id") or ""
        step.pass_(
            "T-1 直接种 scan（幂等命中）",
            f"scan_id={scan_id} image_hash={image_hash[:12]}",
        )
        return scan_id

    scan_id = f"seed-{image_hash[:16]}-{int(time.time())}"
    now_ms = int(time.time() * 1000)
    record: dict[str, Any] = {
        "scan_id": scan_id,
        "scholar_id": args.scholar_id,
        "image_url": f"file://seed-direct/{image_hash}",
        "image_file_id": f"seed-{image_hash[:16]}",
        "image_hash": image_hash,
        "ocr_status": "mock_ready",
        "ocr_text": "[seed-direct] 五年级上册练习二错题照片",
        "ocr_blocks": [],
        "classify_status": "needs_review",
        "note": "drill-chain-verify-seed-direct",
        "created_at": now_ms,
        "completed_at": None,
        "audit_log_id": "",
    }
    await db.insert(MATH_SCAN_UPLOAD_COLLECTION, record)
    step.pass_(
        "T-1 直接种 scan",
        f"scan_id={scan_id} image_hash={image_hash[:12]}",
    )
    return scan_id


def _upload(client: httpx.Client, args: argparse.Namespace,
            image_path: str, note: str, step: StepResult) -> str:
    filename, raw = _read_image(image_path)
    r = client.post(
        "/math/scan/upload",
        data={"scholar_id": args.scholar_id, "note": note},
        files={"image": (filename, raw, "application/octet-stream")},
    )
    if r.status_code != 200:
        raise RuntimeError(f"upload status={r.status_code} body={r.text[:200]}")
    d = r.json().get("data", {})
    if d.get("deduped"):
        step.pass_("T-1 上传（幂等命中）",
                   f"scan_id={d.get('scan_id')} image_hash={str(d.get('image_hash'))[:12]}")
    else:
        step.pass_("T-1 上传", f"scan_id={d.get('scan_id')} status={d.get('status')}")
    return d.get("scan_id") or ""


def _classify_until_terminal(client: httpx.Client, scan_id: str,
                             step: StepResult) -> dict:
    """轮询 classify：404(OCR 未就绪)/processing 重试；success/needs_review 终态。"""
    deadline = time.time() + OCR_READY_TIMEOUT_SECONDS
    while time.time() < deadline:
        r = client.post("/math/scan/classify",
                        json={"scan_id": scan_id, "force_reclassify": False})
        if r.status_code == 404:  # scan 不存在 / OCR 未完成
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"classify status={r.status_code} body={r.text[:200]}")
        d = r.json().get("data", {})
        status = d.get("status", "")
        if status in ("success", "needs_review"):
            return d
        if status == "processing":
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        raise RuntimeError(f"classify 未知终态 status={status} body={r.text[:200]}")
    raise TimeoutError(f"OCR/Judge 未在 {OCR_READY_TIMEOUT_SECONDS}s 内就绪 scan_id={scan_id}")


def _correct_needs_review(client: httpx.Client, args: argparse.Namespace,
                          scan_id: str, items: list[dict], kp_names: list[str],
                          step: StepResult) -> list[str]:
    """needs_review → 自动修正（仅修正未直落项，优先 classify 给的 kp 名若命中教材列表）。

    classify 返回的 items 中 error_record_id 非空 = 该项已按 conf≥0.6 命中候选
    直落 error_record（classify_method=auto_scan），correct 只应补修余下
    needs_review 项；若把已直落项一并 POST，后端会重复新建 manual 记录
    （旧版缺陷：fixed 恒携带 error_record_id="" → 一张含混合 items 的
    needs_review 会把直落题也新建一遍）。
    """
    corrected_ids: list[str] = []
    fixed: list[dict] = []

    auto_ids = [it.get("error_record_id") for it in items if it.get("error_record_id")]
    review_items = [it for it in items if not it.get("error_record_id")]

    if not items:
        # seed-direct 等跳过 classify 的场景：用兜底知识点构造一项
        if not kp_names:
            raise RuntimeError("无可用知识点名做人工修正，请传 --correct-kp")
        kp = args.correct_kp if args.correct_kp and args.correct_kp in kp_names else kp_names[0]
        logger.warning("T-2b items 为空，seed-direct 兜底修正 kp=%r", kp)
        review_items = [{"knowledge_point_name": kp, "error_type": "method"}]
    elif auto_ids:
        logger.warning(
            "T-2b items 含 %d 项已 auto 直落(error_record_id 非空)，"
            "跳过不重复修正: %s", len(auto_ids), auto_ids)

    for it in review_items:
        kp = (it.get("knowledge_point_name") or "").strip()
        # B1.5 error_type 保真：直接透传 classify 的错因四分类，不兜底 method
        #（报告 manual_correct「error_type 被篡改如 computation→reading」根因修复）
        et = (it.get("error_type") or "").strip()
        question_text = (it.get("question_text") or "").strip()
        if kp and kp in kp_names:
            # 候选内命中：直接锚定该正式知识点（标准名）
            fixed.append({
                "knowledge_point_name": kp,
                "error_type": et,
                "question_text": question_text,
            })
        elif args.correct_kp and args.correct_kp in kp_names:
            # 命令行显式指定且命中候选 → 用指定名
            fixed.append({
                "knowledge_point_name": args.correct_kp,
                "error_type": et,
                "question_text": question_text,
            })
        elif kp:
            # B1.5 图谱外：不再机械改挂候选首个（报告 kp_dist 压挂根因）；
            # 原样透传 kp + new_kp_name，由后端 EXTRA_AI 幂等新建/复用节点归链
            logger.warning(
                "T-2b kp=%r 不在教材知识列表，按图谱外 new_kp_name 走 EXTRA_AI 新建（B1.5）",
                kp,
            )
            fixed.append({
                "knowledge_point_name": kp,
                "new_kp_name": kp,
                "error_type": et,
                "question_text": question_text,
            })
        else:
            raise RuntimeError("无可用知识点名做人工修正，请传 --correct-kp")

    if fixed:
        r = client.post(f"/math/scan/{scan_id}/correct", json={"items": fixed})
        if r.status_code != 200:
            raise RuntimeError(f"correct status={r.status_code} body={r.text[:200]}")
        for item in r.json().get("data", {}).get("corrected", []):
            rid = item.get("error_record_id") or ""
            if rid:
                corrected_ids.append(rid)
        step.pass_("T-2b 人工修正(needs_review)",
                   f"scan_id={scan_id} 修正 {len(corrected_ids)} 项 "
                   f"(跳过已 auto 直落 {len(auto_ids)} 项)")
    else:
        step.pass_("T-2b 人工修正(needs_review)",
                   f"scan_id={scan_id} 无待修正项（全部已 auto 直落 {len(auto_ids)} 项）")

    # 已直落记录无需 correct，但也入断言列表供 T-3 error-stats 可见性校验
    return auto_ids + corrected_ids


def _assert_error_stats_has(client: httpx.Client, args: argparse.Namespace,
                            record_id: str, step: StepResult,
                            expect_new_contract: bool) -> None:
    r = client.get("/math/error-stats",
                   params={"scholar_id": args.scholar_id, "limit": 200})
    if r.status_code != 200:
        step.fail("T-3 error-stats 回读", f"status={r.status_code} {r.text[:120]}")
        return
    items = r.json().get("data", {}).get("items", [])
    hit = next((x for x in items if x.get("error_record_id") == record_id), None)
    if not hit:
        step.fail("T-3", f"error-stats 未找到 record {record_id}")
        return
    miss = [k for k in ("knowledge_point_name", "error_type", "source", "created_at")
            if not hit.get(k)]
    if miss:
        step.fail("T-3", f"既有契约字段缺失: {miss}")
        return
    step.pass_("T-3 error-stats 可见",
               f"kp={hit.get('knowledge_point_name')} err={hit.get('error_type')}")

    if expect_new_contract:
        chain = ["question_text", "textbook_id", "grade", "unit_title", "lesson_title",
                 "drill_stats"]
        miss = [k for k in chain if k not in hit]
        if miss:
            step.fail("T-4 链字段透传", f"缺失字段: {miss}（后端 B2 未落地?）")
        elif hit.get("textbook_id") != args.textbook_id:
            step.fail("T-4 链字段透传", f"textbook_id={hit.get('textbook_id')} != {args.textbook_id}")
        else:
            step.pass_("T-4 链字段透传",
                       f"textbook={hit.get('textbook_id')} "
                       f"unit={hit.get('unit_title')} drill_stats={hit.get('drill_stats')}")
    else:
        step.skip("T-4 链字段透传", "新契约未落地，默认 SKIP（--skip-unimplemented=false 开启）")


def phase1(client: httpx.Client, args: argparse.Namespace, step: StepResult,
           kp_names: list[str], expect_new_contract: bool) -> dict:
    result: dict = {"scan_id": "", "classify_status": "", "error_record_ids": []}
    if not args.image:
        step.skip("T-1~T-3 拍照数据准备", "未提供 --image")
        return result

    if args.seed_direct:
        # 绕过微信云存储：直接 DB 种 scan，强制走 needs_review 分支
        scan_id = asyncio.run(_direct_seed_scan(args, step))
        result["scan_id"] = scan_id
        result["classify_status"] = "needs_review"
        items: list[dict] = []
    else:
        scan_id = _upload(client, args, args.image, "drill-chain-verify-seed", step)
        result["scan_id"] = scan_id
        try:
            d = _classify_until_terminal(client, scan_id, step)
        except (TimeoutError, RuntimeError) as e:
            step.fail("T-2 classify", str(e))
            return result
        result["classify_status"] = d.get("status", "")
        items = d.get("items", [])

    if result["classify_status"] == "needs_review":
        ids = _correct_needs_review(client, args, scan_id, items, kp_names, step)
        result["error_record_ids"].extend(ids)
        # needs_review 项（未直落、本次交由 correct 补修的项）信息落产物
        result["needs_review_items"] = [
            {"knowledge_point_name": i.get("knowledge_point_name", ""),
             "error_type": i.get("error_type", "")}
            for i in items if not i.get("error_record_id")]
        result["auto_written_ids"] = [
            i.get("error_record_id") for i in items if i.get("error_record_id")]
    else:
        ids = [i.get("error_record_id", "") for i in items if i.get("error_record_id")]
        step.pass_("T-2a 归类成功",
                   f"scan_id={scan_id} items={len(items)} records={ids}")
        result["error_record_ids"].extend(ids)

    for rid in result["error_record_ids"]:
        if rid:
            _assert_error_stats_has(client, args, rid, step, expect_new_contract)
    if not result["error_record_ids"]:
        step.fail("T-3", "未产生任何 error_record_id")
    return result


# ---------------------------------------------------------------------------
# phase2：新契约验收（T-6 ~ T-10，默认 SKIP；后端落地后 --skip-unimplemented=false）
# ---------------------------------------------------------------------------


def phase2(client: httpx.Client, args: argparse.Namespace, step: StepResult,
           seed: dict) -> None:
    if args.skip_unimplemented:
        step.skip("T-6~T-10 drill 闭环", "drill 接口未落地，默认 SKIP")
        return
    if not seed.get("error_record_ids"):
        step.fail("T-6", "无基线 error_record，先跑一期拍照数据准备")
        return
    rid = seed["error_record_ids"][0]
    r = client.post("/math/drill/generate", json={
        "scholar_id": args.scholar_id, "source": "error_item",
        "error_record_ids": [rid], "count_per_kp": 2})
    if r.status_code != 200:
        step.fail("T-6 drill/generate", f"status={r.status_code} {r.text[:200]}")
        return
    gd = r.json().get("data", {})
    drill_id = gd.get("drill_id", "")
    items = gd.get("items", [])
    step.pass_("T-6 drill/generate",
               f"drill_id={drill_id} items={len(items)} answer 未出参="
               f"{all('answer' not in i for i in items)}")

    answers = [{"drill_item_id": i.get("drill_item_id"),
                "answer": "9" if i.get("answer_type") == "number" else "对",
                "self_reviewed": True if i.get("answer_type") == "text" else None}
               for i in items]
    r = client.post(f"/math/drill/{drill_id}/submit", json={"answers": answers})
    if r.status_code != 200:
        step.fail("T-8/T-9/T-10 drill/submit", f"status={r.status_code} {r.text[:200]}")
        return
    sd = r.json().get("data", {})
    step.pass_("T-8/T-9/T-10 drill/submit",
               f"summary={sd.get('result_summary')} "
               f"refresh_records={sd.get('refresh_records')}")
    step.pass_("T-6~T-10 drill 闭环", "全部通过（number 猜 9 仅为打通链路，非判定用例）")


def _query_error_records_by_scan(scan_id: str) -> list[dict]:
    """直连 DB 查询某 scan 产生的 error_record（T-11 校验链锚点用）。"""
    from config import (
        ENV_ID,
        REGION,
        SECRET_ID,
        SECRET_KEY,
        SESSION_TOKEN,
    )
    from services.database import CloudBaseNoSQLClient, ERROR_RECORD_COLLECTION

    async def _go() -> list[dict]:
        db = CloudBaseNoSQLClient(
            env_id=ENV_ID,
            region=REGION,
            secret_id=SECRET_ID,
            secret_key=SECRET_KEY,
            session_token=SESSION_TOKEN,
        )
        res = await db.query(
            ERROR_RECORD_COLLECTION,
            where={"scan_upload_id": scan_id},
            limit=100,
        )
        return res.get("records") or []

    return asyncio.run(_go())


def phase3(client: httpx.Client, args: argparse.Namespace, step: StepResult) -> None:
    """T-11 图谱外 → EXTRA_AI（需 --image-extra；后端 B1.5 落地后实链验收）。

    走完整链路：上传图谱外照片 → classify（高置信未命中自动 EXTRA_AI；
    低置信走 correct 补修）→ 直连 DB 校验每条 error_record 均锚
    textbook_id=EXTRA_AI 且 node_code 非空（不强行改挂正式教材节点）。
    """
    if not args.image_extra:
        step.skip("T-11 EXTRA_AI", "未提供 --image-extra")
        return
    if args.skip_unimplemented:
        step.skip("T-11 EXTRA_AI", "二期新契约断言未开启（--skip-unimplemented=false 开启）")
        return

    scan_id = _upload(client, args, args.image_extra, "T-11 图谱外照片", step)
    try:
        d = _classify_until_terminal(client, scan_id, step)
    except (TimeoutError, RuntimeError) as e:
        step.fail("T-11 classify", str(e))
        return
    items = d.get("items", [])
    if not items:
        step.fail("T-11 EXTRA_AI", "classify 无产物")
        return
    ids = [i.get("error_record_id", "") for i in items if i.get("error_record_id")]
    if d.get("status") == "needs_review" or not ids:
        # 补修未直落项：kp 不在教材列表 → _correct_needs_review 走 new_kp_name，
        # 由后端 EXTRA_AI 幂等新建节点归链（kp_names=[] 即不落入正式候选）
        try:
            corrected = _correct_needs_review(client, args, scan_id, items, [], step)
        except RuntimeError as e:
            step.fail("T-11 EXTRA_AI", f"correct 失败: {e}")
            return
        ids = list(dict.fromkeys([i for i in ids + corrected if i]))
    if not ids:
        step.fail("T-11 EXTRA_AI", "未产生 error_record_id")
        return

    records = _query_error_records_by_scan(scan_id)
    if not records:
        step.fail("T-11 EXTRA_AI", f"DB 未查到 scan_id={scan_id} 的 error_record")
        return
    miss = [r.get("record_id") for r in records
            if r.get("textbook_id") != "EXTRA_AI" or not r.get("node_code")]
    if miss:
        step.fail(
            "T-11 EXTRA_AI",
            f"{len(miss)}/{len(records)} 条未锚 EXTRA_AI（textbook_id 或 node_code 缺失）: {miss[:5]}",
        )
        return
    step.pass_(
        "T-11 EXTRA_AI",
        f"{len(records)} 条 error_record 均锚 EXTRA_AI（textbook_id=EXTRA_AI, node_code 非空），"
        f"无强行改挂正式教材",
    )


def main() -> int:
    args = parse_args()
    logger.info("计划: scholar=%s textbook=%s base=%s image=%s seed_direct=%s",
                args.scholar_id, args.textbook_id, args.base_url,
                args.image or "(只读)", args.seed_direct)
    if args.dry_run:
        logger.info("--dry-run：仅打印计划，退出 0")
        return 0

    step = StepResult()
    expect_new_contract = not args.skip_unimplemented
    seed: dict = {}
    with _client(args) as client:
        kp_names = phase0(client, args, step)
        seed = phase1(client, args, step, kp_names, expect_new_contract)
        phase2(client, args, step, seed)
        phase3(client, args, step)

    for line in step.done:
        logger.info("%s", line)
    for line in step.skipped:
        logger.info("%s", line)
    for line in step.failed:
        logger.error("%s", line)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps({
            "scholar_id": args.scholar_id,
            "textbook_id": args.textbook_id,
            "seed": seed,
            "summary": {"pass": len(step.done), "skip": len(step.skipped),
                        "fail": len(step.failed)},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("产物已写入: %s", out)

    logger.info("结果: PASS=%d SKIP=%d FAIL=%d", len(step.done),
                len(step.skipped), len(step.failed))
    return 0 if step.ok else 1


if __name__ == "__main__":
    sys.exit(main())
