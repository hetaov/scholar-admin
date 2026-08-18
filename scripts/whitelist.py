"""付费能力白名单管理脚本（app_whitelist 集合）

本脚本用于维护"仅限授权用户使用付费 AI 能力"的白名单：
- scholar-admin 侧：/eval /evaluation /audio/tts /vision /match/dialogue
  /conversation /planner /training /build 等付费路由
- 云函数侧：updateTrackingStatus（ASR + 混元评分）

用法（在 scholar-admin 项目根目录执行，需已配置 CloudBase 凭据）：
    python scripts/whitelist.py list
    python scripts/whitelist.py add  <openid>
    python scripts/whitelist.py remove <openid>

说明：
- 集合/文档不存在时自动创建（幂等）。
- 如何获取 openid：在小程序开发者工具中调用 getOpenId 云函数，
  或在小程序内打印 wx.cloud 返回的 openid。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import WHITELIST_COLLECTION  # noqa: E402
from services.database import CloudBaseNoSQLClient  # noqa: E402
from services.dependencies import get_db  # noqa: E402

DOC_ID = "paid"


async def _ensure_doc(db: CloudBaseNoSQLClient) -> None:
    """确保集合与白名单文档存在（幂等）。"""
    if not await db.check_collection(WHITELIST_COLLECTION):
        await db.create_collection(WHITELIST_COLLECTION)
        print(f"集合 `{WHITELIST_COLLECTION}` 创建成功")

    res = await db.query(
        WHITELIST_COLLECTION,
        where={"_id": DOC_ID},
        limit=1,
        select={"openids": 1},
    )
    if not (res.get("records") or []):
        await db.insert(WHITELIST_COLLECTION, {"_id": DOC_ID, "openids": []})
        print(f"白名单文档 {DOC_ID} 创建成功")


async def _read_openids(db: CloudBaseNoSQLClient) -> list[str]:
    res = await db.query(
        WHITELIST_COLLECTION,
        where={"_id": DOC_ID},
        limit=1,
        select={"openids": 1},
    )
    records = res.get("records") or []
    if not records:
        return []
    openids = records[0].get("openids") or []
    return [o for o in openids if isinstance(o, str)]


async def _write_openids(db: CloudBaseNoSQLClient, openids: list[str]) -> None:
    await db.update(
        WHITELIST_COLLECTION,
        where={"_id": DOC_ID},
        data={"$set": {"openids": openids}},
    )


async def cmd_list(db: CloudBaseNoSQLClient) -> None:
    await _ensure_doc(db)
    openids = await _read_openids(db)
    if not openids:
        print("白名单为空（当前无授权用户，所有付费能力均被拒绝）")
        return
    print(f"当前授权用户（{len(openids)} 个）：")
    for i, o in enumerate(openids, 1):
        print(f"  {i}. {o}")


async def cmd_add(db: CloudBaseNoSQLClient, openid: str) -> None:
    await _ensure_doc(db)
    openids = await _read_openids(db)
    if openid in openids:
        print(f"openid 已在白名单中：{openid}")
        return
    openids.append(openid)
    await _write_openids(db, openids)
    print(f"已添加授权用户：{openid}（当前共 {len(openids)} 个）")


async def cmd_remove(db: CloudBaseNoSQLClient, openid: str) -> None:
    await _ensure_doc(db)
    openids = await _read_openids(db)
    if openid not in openids:
        print(f"openid 不在白名单中：{openid}")
        return
    openids.remove(openid)
    await _write_openids(db, openids)
    print(f"已移除授权用户：{openid}（剩余 {len(openids)} 个）")


async def main() -> None:
    parser = argparse.ArgumentParser(description="付费能力白名单管理")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出所有授权 openid")
    p_add = sub.add_parser("add", help="添加授权 openid")
    p_add.add_argument("openid")
    p_rm = sub.add_parser("remove", help="移除授权 openid")
    p_rm.add_argument("openid")

    args = parser.parse_args()
    db = get_db()

    if args.command == "list":
        await cmd_list(db)
    elif args.command == "add":
        await cmd_add(db, args.openid)
    elif args.command == "remove":
        await cmd_remove(db, args.openid)


if __name__ == "__main__":
    asyncio.run(main())
