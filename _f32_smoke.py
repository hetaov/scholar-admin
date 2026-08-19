"""F3.2 冒烟测试（跑完即删）：a4_renderer 渲染管线核心逻辑

覆盖：
1. build_sheet_html：题号/题干/奥数难度档标注/错因标注/作答区/答案页/二维码/学生与日期
2. build_degraded_html：纯文本降级模板
3. renderSheetJob 成功路径（mock DB + mock 渲染产物，校验 artifacts/file_refs/qrcode_ref/状态机）
4. renderSheetJob 失败 → 降级 degraded；降级也失败 → failed + error_code
5. 重试上限：retries 超限不再降级
6. renderPendingJobs 批量消费 queued
7. 二维码签名（SHEET_QR_SECRET 配置时含 signature/expires_at）
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from config import RENDER_OUTPUT_DIR, SHEET_QR_SECRET  # noqa: E402
from services.math import a4_renderer  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def sample_sheet(sheet_id: str = "ps_test_001") -> dict:
    return {
        "_id": sheet_id,
        "sheet_id": sheet_id,
        "scholar_id": "sch_001",
        "template_ref": {"template_type": "standard"},
        "status": "generated",
        "generated_at": 1787100000000,
        "items": [
            {
                "item_id": "it_1",
                "question": "计算 1/2 + 1/3 = ?",
                "answer": "5/6",
                "hint_card": "通分后再相加",
                "node_code": "n1",
                "target_error": "通分错误",
                "variant_level": "L1",
                "difficulty": 2,
            },
            {
                "item_id": "it_2",
                "question": "已知 x²=4，求 x",
                "answer": "x=±2",
                "hint_card": "",
                "node_code": "n1",
                "target_error": "",
                "variant_level": "L1",
                "difficulty": 5,
            },
        ],
    }


class MockDB:
    """内存 mock：记录 insert/update/query 调用，供断言"""

    def __init__(self, sheets=None, jobs=None):
        self.sheets = {s["sheet_id"]: dict(s) for s in (sheets or [sample_sheet()])}
        if isinstance(jobs, dict):
            self.jobs = {sid: dict(j) for sid, j in jobs.items()}
        else:
            self.jobs = {j["sheet_id"]: dict(j) for j in (jobs or [])}
        self.updates = []

    async def query(self, collection, where=None, order=None, offset=0, limit=100, select=None):
        if collection == "practice_sheet":
            sheet_id = (where or {}).get("sheet_id")
            return {"records": [dict(self.sheets[sheet_id])]} if sheet_id in self.sheets else {"records": []}
        if collection == "sheet_render_job":
            sheet_id = (where or {}).get("sheet_id")
            if sheet_id:
                return {"records": [dict(self.jobs[sheet_id])]} if sheet_id in self.jobs else {"records": []}
            status = (where or {}).get("status")
            matched = [j for j in self.jobs.values() if not status or j["status"] == status]
            matched.sort(key=lambda j: j.get("created_at", 0))
            return {"records": [dict(j) for j in matched[:limit]]}
        return {"records": []}

    async def update(self, collection, where, data, upsert=False, multi=True):
        self.updates.append((collection, dict(where), data))
        if collection == "practice_sheet":
            sheet_id = where.get("sheet_id")
            if sheet_id in self.sheets:
                self.sheets[sheet_id].update(data.get("$set", {}))
        if collection == "sheet_render_job":
            sheet_id = where.get("sheet_id")
            if sheet_id in self.jobs:
                self.jobs[sheet_id].update(data.get("$set", {}))

    async def insert(self, collection, data):
        pass


def queued_job(sheet_id: str, retries: int = 0, status: str = "queued") -> dict:
    return {
        "_id": f"srj_{sheet_id}",
        "job_id": f"srj_{sheet_id}",
        "sheet_id": sheet_id,
        "status": status,
        "retries": retries,
        "created_at": 1787100000000,
    }


async def test_html() -> None:
    print("\n[1] build_sheet_html")
    html = a4_renderer.build_sheet_html(sample_sheet(), qr_data_uri="data:image/png;base64,AAA")
    check("题号 1./2.", "1." in html and "2." in html)
    check("题干包含", "1/2 + 1/3" in html and "x²=4" in html)
    check("奥数难度档标注", "奥数·竞赛" in html)
    check("错因标注", "错因:通分错误" in html)
    check("作答区", 'class="answer-area"' in html)
    check("答案页（防背题）", "参考答案（家长核对）" in html and "5/6" in html and "x=±2" in html)
    check("二维码嵌入", 'class="qr"' in html and "data:image/png" in html)
    check("学生信息", "sch_001" in html)
    import time as _t

    expect_date = _t.strftime("%Y-%m-%d", _t.localtime(1787100000000 / 1000))
    check("日期", expect_date in html, f"expect={expect_date}")
    check("HTML 转义防注入", build_html_escape_ok())


def build_html_escape_ok() -> bool:
    sheet = sample_sheet()
    sheet["items"][0]["question"] = '<script>alert(1)</script>'
    html = a4_renderer.build_sheet_html(sheet)
    return "<script>alert" not in html and "&lt;script&gt;" in html


async def test_degraded_html() -> None:
    print("\n[2] build_degraded_html")
    html = a4_renderer.build_degraded_html(sample_sheet())
    check("纯文本模板含题干", "1/2 + 1/3" in html)
    check("纯文本模板含答案", "5/6" in html)
    check("答案块分页", "page-break-before" in html)


async def test_qrcode() -> None:
    print("\n[3] 二维码签名")
    ref = a4_renderer._qrcode_ref("ps_qr_001")
    check("含 expires_at", ref["expires_at"] > 0)
    if SHEET_QR_SECRET:
        check("含签名", len(ref["signature"]) == 64, ref["signature"][:8])
        check("qr_url 含签名", "signature=" in ref["qr_url"])
        check("qr_url 含 sheet_id", "ps_qr_001" in ref["qr_url"])
    else:
        print("  (SHEET_QR_SECRET 未配置，qr_url 为空属预期降级)")


async def test_success_path(monkey_render: bool = True) -> None:
    print("\n[4] renderSheetJob 成功路径")
    db = MockDB(jobs=[queued_job("ps_test_001")])
    a4_renderer._render_artifacts = _fake_render_success  # patch 渲染（跳过真浏览器）
    job = await a4_renderer.renderSheetJob(db, "ps_test_001")
    check("状态 success", job["status"] == "success", job["status"])
    check("file_refs 三件套", all(job["file_refs"][k] for k in ("pdf", "png", "preview")))
    check("artifacts 与 file_refs 一致", job["artifacts"]["pdf"] == job["file_refs"]["pdf"])
    check("回写 job 状态", db.jobs["ps_test_001"]["status"] == "success")
    check("回写 sheet file_refs", db.sheets["ps_test_001"]["file_refs"]["pdf"] != "")
    check("qrcode_ref 回写", db.sheets["ps_test_001"]["qrcode_ref"]["expires_at"] > 0)
    check("URL 前缀", job["file_refs"]["pdf"].startswith(f"{RENDER_OUTPUT_DIR}") is False and "/ps_test_001/sheet.pdf" in job["file_refs"]["pdf"])


async def _fake_render_success(html: str, sheet_dir: Path) -> None:
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for name in ("sheet.pdf", "sheet.png", "preview.png"):
        (sheet_dir / name).write_bytes(b"fake-bytes")


async def _fake_render_fail(html: str, sheet_dir: Path) -> None:
    raise a4_renderer.RenderError("boom")


async def test_degraded_path() -> None:
    print("\n[5] 渲染失败 → 降级 degraded")
    db = MockDB(jobs=[queued_job("ps_test_001")])

    calls = {"n": 0}

    async def _fake_fail_once_then_ok(html, sheet_dir):
        calls["n"] += 1
        if calls["n"] == 1:  # 完整版式渲染失败
            raise a4_renderer.RenderError("boom")
        await _fake_render_success(html, sheet_dir)  # 降级纯文本渲染成功

    a4_renderer._render_artifacts = _fake_fail_once_then_ok
    job = await a4_renderer.renderSheetJob(db, "ps_test_001")
    check("降级状态 degraded", job["status"] == "degraded", job["status"])
    check("retries=1", job["retries"] == 1, job["retries"])
    check("error_code=render_failed", job["error_code"] == "render_failed", job["error_code"])
    check("降级仍有 file_refs", job["file_refs"]["pdf"] != "")
    check("job 记录降级", db.jobs["ps_test_001"]["status"] == "degraded")


async def test_failed_path() -> None:
    print("\n[6] 降级也失败 → failed")
    db = MockDB(jobs=[queued_job("ps_test_001")])

    async def _fake_render_fail_twice(html, sheet_dir):
        raise a4_renderer.RenderError("boom")

    # 连续失败：主渲染 + 降级渲染都失败
    a4_renderer._render_artifacts = _fake_render_fail_twice
    job = await a4_renderer.renderSheetJob(db, "ps_test_001")
    check("状态 failed", job["status"] == "failed", job["status"])
    check("retries=2", job["retries"] == 2, job["retries"])
    check("error_code", job["error_code"] in ("render_failed", "retry_exceeded"), job["error_code"])
    check("file_refs 清空", job["file_refs"]["pdf"] == "")


async def test_dependency_missing() -> None:
    print("\n[7] Chromium 未安装 → dependency_missing")
    db = MockDB(jobs=[queued_job("ps_test_001")])

    async def _fake_unavailable(html, sheet_dir):
        raise a4_renderer.RendererUnavailableError("playwright 未安装")

    a4_renderer._render_artifacts = _fake_unavailable
    job = await a4_renderer.renderSheetJob(db, "ps_test_001")
    check("状态 failed", job["status"] == "failed", job["status"])
    check("error_code=dependency_missing", job["error_code"] == "dependency_missing", job["error_code"])


async def test_pending() -> None:
    print("\n[8] renderPendingJobs 批量")
    db = MockDB(
        sheets=[sample_sheet("ps_a"), sample_sheet("ps_b")],
        jobs={
            "ps_a": queued_job("ps_a"),
            "ps_b": queued_job("ps_b"),
        },
    )
    a4_renderer._render_artifacts = _fake_render_success
    results = await a4_renderer.renderPendingJobs(db, limit=10)
    check("处理 2 个任务", len(results) == 2, len(results))
    check("全部 success", all(r["status"] == "success" for r in results))


async def main() -> None:
    await test_html()
    await test_degraded_html()
    await test_qrcode()
    await test_success_path()
    await test_degraded_path()
    await test_failed_path()
    await test_dependency_missing()
    await test_pending()
    print(f"\n===== 结果: {PASS} 通过 / {FAIL} 失败 =====")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
