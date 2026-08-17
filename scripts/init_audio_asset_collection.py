"""一次性脚本：创建 `audio_asset` 集合（幂等，F3/3.3）

用法（在 scholar-admin 项目根目录执行）：
    python -m scripts.init_audio_asset_collection
    # 或
    python scripts/init_audio_asset_collection.py

已存在集合时直接跳过（幂等，可重复执行）。
需要 CloudBase 环境变量（.env / cloudbaserc.json 注入），本地若无凭据会提示。

注意：data-model-contract §4.10 要求 `text_hash` 建**唯一索引**（命中即缓存命中）。
CloudBase 索引需在控制台创建：数据库 → `audio_asset` → 索引管理 → 新增
`text_hash`（升序，唯一=是）。本脚本只保证集合存在。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.dependencies import get_db  # noqa: E402
from services.tts import AUDIO_ASSET_COLLECTION  # noqa: E402


async def main() -> None:
    db = get_db()
    if await db.check_collection(AUDIO_ASSET_COLLECTION):
        print(f"集合 `{AUDIO_ASSET_COLLECTION}` 已存在，跳过创建（幂等）")
        return
    await db.create_collection(AUDIO_ASSET_COLLECTION)
    print(f"集合 `{AUDIO_ASSET_COLLECTION}` 创建成功")
    print("提示：请在 CloudBase 控制台为 text_hash 建立唯一索引（data-model-contract §4.10）")


if __name__ == "__main__":
    asyncio.run(main())
