"""课文断点持久化接口 — GET/POST/DELETE /lesson/resume_point（详情页重构 v1 · D5 + §3）

契约 §3 接口规格：
- GET    /lesson/resume_point?lesson_id=...  → {tab, sentence_id?, microflow_node?, updated_at} 或 {}
- POST   /lesson/resume_point                  → {ok: true}（body: {lesson_id, tab, sentence_id?, microflow_node?}）
- DELETE /lesson/resume_point?lesson_id=...    → {ok: true, deleted: bool}（L-T7-2：全课掌握/脏数据清理）

鉴权：免费路由（main.py 挂载进 _FREE_ROUTERS），按 X-WX-OPENID 头注入 openid，入参不带 openid。
错误映射：openid 缺失 → 401；lesson_id 缺失 → 400；tab/microflow_node 非法 → 400
（upsert_resume_point 内部抛 ValueError，路由层捕获转 400）。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from services.auth import get_request_openid
from services.dependencies import get_db
from services.models_lesson_resume import get_resume_point, upsert_resume_point, clean_resume_point

logger = logging.getLogger("scholar-admin.routes.lesson")
router = APIRouter(prefix="/lesson", tags=["lesson"])


# ===========================================================================
# 请求体模型
# ===========================================================================


class ResumePointUpsertRequest(BaseModel):
    """POST /lesson/resume_point 请求体。

    字段全部给默认值，由路由层做 400 校验（而非 Pydantic 422），
    与 routes/state.py 的 400 风格对齐。
    """

    lesson_id: str = Field("", description="课文 ID（必填）")
    tab: str = Field("", description="Tab 枚举：content/mastery/training/conversation（必填）")
    sentence_id: str | None = Field(None, description="句子 ID（掌握 Tab 无；可空）")
    microflow_node: str | None = Field(
        None,
        description="微流程节点：listen/shadowing/recall/compare/ai_followup（可空）",
    )


# ===========================================================================
# GET /lesson/resume_point — 读取断点
# ===========================================================================


@router.get("/resume_point")
async def get_lesson_resume_point(
    request: Request,
    lesson_id: str = Query("", description="课文 ID（必填，query）"),
):
    """读取某用户某课的断点位置。

    返回 `{tab, sentence_id, microflow_node, updated_at}`；无断点返回空对象 `{}`。
    """
    openid = get_request_openid(request)
    if not openid:
        raise HTTPException(status_code=401, detail="缺少 openid（未携带 X-WX-OPENID）")

    lesson_id = (lesson_id or "").strip()
    if not lesson_id:
        raise HTTPException(status_code=400, detail="缺少参数 lesson_id")

    db = get_db()
    doc = await get_resume_point(db, openid=openid, lesson_id=lesson_id)
    if not doc:
        return {}
    return {
        "tab": doc.get("tab"),
        "sentence_id": doc.get("sentence_id"),
        "microflow_node": doc.get("microflow_node"),
        "updated_at": doc.get("updated_at"),
    }


# ===========================================================================
# POST /lesson/resume_point — upsert 断点
# ===========================================================================


@router.post("/resume_point")
async def save_lesson_resume_point(body: ResumePointUpsertRequest, request: Request):
    """按 (openid, lesson_id) 唯一键 upsert 断点；返回 `{ok: true}`。

    - tab 必填且 ∈ VALID_RESUME_TABS（由 upsert_resume_point 校验，非法抛 ValueError → 400）
    - microflow_node 非空时须 ∈ VALID_MICROFLOW_NODES（同上）
    - sentence_id / microflow_node 可空
    """
    openid = get_request_openid(request)
    if not openid:
        raise HTTPException(status_code=401, detail="缺少 openid（未携带 X-WX-OPENID）")

    lesson_id = (body.lesson_id or "").strip()
    if not lesson_id:
        raise HTTPException(status_code=400, detail="缺少参数 lesson_id")

    tab = (body.tab or "").strip()
    sentence_id = body.sentence_id
    microflow_node = body.microflow_node

    db = get_db()
    try:
        await upsert_resume_point(
            db,
            openid=openid,
            lesson_id=lesson_id,
            tab=tab,
            sentence_id=sentence_id,
            microflow_node=microflow_node,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"[resume_point] upsert openid={openid} lesson_id={lesson_id} "
        f"tab={tab} sentence_id={sentence_id} microflow_node={microflow_node}"
    )
    return {"ok": True}


# ===========================================================================
# DELETE /lesson/resume_point — 清理断点（L-T7-2：全课掌握 / 脏数据清理）
# ===========================================================================


@router.delete("/resume_point")
async def delete_lesson_resume_point(
    request: Request,
    lesson_id: str = Query("", description="课文 ID（必填，query）"),
):
    """删除某用户某课的断点；返回 `{ok: true, deleted: bool}`。

    无记录可删时 `deleted=false`（不抛 404，幂等）。
    """
    openid = get_request_openid(request)
    if not openid:
        raise HTTPException(status_code=401, detail="缺少 openid（未携带 X-WX-OPENID）")

    lesson_id = (lesson_id or "").strip()
    if not lesson_id:
        raise HTTPException(status_code=400, detail="缺少参数 lesson_id")

    db = get_db()
    deleted = await clean_resume_point(db, openid=openid, lesson_id=lesson_id)
    logger.info(
        f"[resume_point] delete openid={openid} lesson_id={lesson_id} deleted={deleted}"
    )
    return {"ok": True, "deleted": deleted}
