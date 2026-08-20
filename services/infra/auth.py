"""付费能力白名单鉴权

所有调用付费 AI 能力（ASR / LLM / TTS / 视觉 / 评测 / 内容生成）的接口
都必须经过 require_paid_user 依赖校验：调用者 openid 必须存在于
app_whitelist 集合（文档 _id 固定为 "paid"，字段 openids: string[]）。

openid 来源：
- 生产（AUTH_MODE=enforce）：请求必须携带 X-WX-OPENID（微信云托管
  callContainer 网关自动注入），缺失即拒绝，防止绕过小程序直接 HTTP 调用。
- 开发（AUTH_MODE=dev，默认）：请求无 X-WX-OPENID 时放行，保证本地调试与
  pytest 测试不受影响；若显式携带 header 仍会校验白名单。
"""
from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Request

from config import AUTH_MODE, WHITELIST_COLLECTION
from services.database import CloudBaseNoSQLClient
from services.dependencies import get_db

logger = logging.getLogger("scholar-admin.auth")

# 微信云托管 callContainer 网关注入的用户 openid 请求头
OPENID_HEADER = "X-WX-OPENID"

# 白名单文档固定 _id
WHITELIST_DOC_ID = "paid"


def get_request_openid(request: Request) -> str:
    """从请求头解析调用者 openid。

    enforce 模式缺少 X-WX-OPENID 视为不可信请求（返回空串）；
    dev 模式允许无身份请求（本地调试 / pytest）。
    """
    openid = (request.headers.get(OPENID_HEADER) or "").strip()
    if not openid:
        if AUTH_MODE == "enforce":
            logger.warning("生产模式收到无 openid 请求（可能来自非小程序直连）")
        else:
            logger.debug("开发模式：请求无 X-WX-OPENID，按本地调试放行")
    return openid


async def is_whitelisted(openid: str, db: CloudBaseNoSQLClient) -> bool:
    """openid 是否在付费白名单中。集合/文档不存在一律视为未授权（安全默认）。"""
    if not openid:
        return False
    try:
        res = await db.query(
            WHITELIST_COLLECTION,
            where={"_id": WHITELIST_DOC_ID},
            limit=1,
            select={"openids": 1},
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"白名单查询失败: {type(e).__name__}: {e}")
        return False
    records = res.get("records") or []
    if not records:
        logger.warning("白名单文档不存在，拒绝访问")
        return False
    openids = records[0].get("openids") or []
    return openid in openids


async def require_paid_user(
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> str:
    """FastAPI 依赖：校验调用者是否有权使用付费 AI 能力。"""
    openid = get_request_openid(request)
    if AUTH_MODE == "dev" and not openid:
        return ""  # 本地开发/测试放行
    if not await is_whitelisted(openid, db):
        raise HTTPException(
            status_code=403,
            detail="未授权使用：本小程序仅供授权用户使用",
        )
    return openid
