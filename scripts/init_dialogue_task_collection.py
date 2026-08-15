"""一次性脚本：创建 `dialogue_task` 集合（幂等）

用法（在 scholar-admin 项目根目录执行）：
    python -m scripts.init_dialogue_task_collection
    # 或
    python scripts/init_dialogue_task_collection.py

已存在集合时直接跳过（幂等，可重复执行）。
需要 CloudBase 环境变量（.env / cloudbaserc.json 注入），本地若无凭据会提示。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.dependencies import get_db  # noqa: E402
from services.dialogue_task import COLLECTION  # noqa: E402


async def main() -> None:
    db = get_db()
    if await db.check_collection(COLLECTION):
        print(f"集合 `{COLLECTION}` 已存在，跳过创建（幂等）")
        return
    await db.create_collection(COLLECTION)
    print(f"集合 `{COLLECTION}` 创建成功")


if __name__ == "__main__":
    asyncio.run(main())
