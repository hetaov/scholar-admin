"""一次性脚本：创建 `immersive_progress` 集合（幂等）

用法（在 scholar-admin 项目根目录执行）：
    python -m scripts.init_immersive_progress_collection
    # 或
    python scripts/init_immersive_progress_collection.py

已存在集合时直接跳过（幂等，可重复执行）。
需要 CloudBase 环境变量（.env / cloudbaserc.json 注入），本地若无凭据会提示。

对应契约：docs_v2/03-change/proposals/2026-08-29-沉浸式五步进度持久化后端接口.md
（data-model-contract §4.17 `immersive_progress`，复合唯一索引
 `{scholar_id, textbook_id, group_id}` 建议控制台/初始化时创建）
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.dependencies import get_db  # noqa: E402
from services.learning.immersive_progress import COLLECTION  # noqa: E402


async def main() -> None:
    db = get_db()
    if await db.check_collection(COLLECTION):
        print(f"集合 `{COLLECTION}` 已存在，跳过创建（幂等）")
        return
    await db.create_collection(COLLECTION)
    print(f"集合 `{COLLECTION}` 创建成功")


if __name__ == "__main__":
    asyncio.run(main())
