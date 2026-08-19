"""F3.2 练习纸渲染任务兜底脚本：批量消费 queued 的 sheet_render_job。

用途：
- 生成接口在进程内 create_task 触发渲染，但 CloudRun 多实例/重启可能丢失后台任务；
  本脚本可挂定时（cron / CloudRun 定时任务）轮询补偿，保证 queued 任务最终被消费。
- 也用于手动重试失败任务：将 failed 任务改回 queued 后运行本脚本即可重渲染。

用法示例（服务进程 .venv 中运行）：
    .venv/bin/python scripts/render_sheet_jobs.py                  # 处理 20 个 queued 任务
    .venv/bin/python scripts/render_sheet_jobs.py --limit 50       # 单次最多 50 个
    .venv/bin/python scripts/render_sheet_jobs.py --sheet ps_xxx   # 指定练习纸重渲染

依赖：playwright + qrcode（requirements.txt）；未安装时任务置 failed（dependency_missing），
不影响主链路。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# 允许直接以脚本方式运行（scripts/ 下无包结构）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.database import SHEET_RENDER_JOB_COLLECTION  # noqa: E402
from services.dependencies import get_db  # noqa: E402
from services.math.a4_renderer import (  # noqa: E402
    RENDER_JOB_FAILED,
    RENDER_JOB_QUEUED,
    renderPendingJobs,
    renderSheetJob,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("render_sheet_jobs")


async def _retry_sheet(db, sheet_id: str) -> dict | None:
    """将指定练习纸的渲染任务重置为 queued 并立即重渲染"""
    # 已存在任务：置回 queued 重新消费
    await db.update(
        SHEET_RENDER_JOB_COLLECTION,
        {"sheet_id": sheet_id},
        {"$set": {"status": RENDER_JOB_QUEUED, "error_code": ""}},
    )
    return await renderSheetJob(db, sheet_id)


async def run(args: argparse.Namespace) -> int:
    db = get_db()
    if args.sheet:
        logger.info("重渲染指定练习纸 sheet_id=%s", args.sheet)
        result = await _retry_sheet(db, args.sheet)
        if result:
            print(
                f"sheet_id={result['sheet_id']} status={result['status']} "
                f"file_refs={result.get('file_refs')}"
            )
        return 0

    logger.info("消费 queued 渲染任务（limit=%d）", args.limit)
    results = await renderPendingJobs(db, limit=args.limit)
    print(
        f"\n处理完成：{len(results)} 个任务\n"
        + "\n".join(
            f"  - {r['sheet_id']}  status={r['status']}  error_code={r.get('error_code') or '-'}"
            for r in results
        )
    )
    failed = [r for r in results if r.get("status") == RENDER_JOB_FAILED]
    if failed:
        logger.warning("%d 个任务渲染失败（error_code 见上表）", len(failed))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="练习纸渲染任务兜底脚本（F3.2）")
    parser.add_argument("--limit", type=int, default=20, help="单次最多处理任务数（默认 20）")
    parser.add_argument(
        "--sheet", type=str, default="", help="指定 sheet_id 重渲染（覆盖 --limit）"
    )
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("已中断")
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
