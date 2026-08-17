"""一次性脚本：创建 `speech_evaluation` 集合（幂等，F2/2.3）

用法（在 scholar-admin 项目根目录执行）：
    python -m scripts.init_speech_evaluation_collection
    # 或
    python scripts/init_speech_evaluation_collection.py

已存在集合时直接跳过（幂等，可重复执行）。
需要 CloudBase 环境变量（.env / cloudbaserc.json 注入），本地若无凭据会提示。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.dependencies import get_db  # noqa: E402
from services.speech_eval import SPEECH_EVALUATION_COLLECTION  # noqa: E402


async def main() -> None:
    db = get_db()
    if await db.check_collection(SPEECH_EVALUATION_COLLECTION):
        print(f"集合 `{SPEECH_EVALUATION_COLLECTION}` 已存在，跳过创建（幂等）")
        return
    await db.create_collection(SPEECH_EVALUATION_COLLECTION)
    print(f"集合 `{SPEECH_EVALUATION_COLLECTION}` 创建成功")


if __name__ == "__main__":
    asyncio.run(main())
