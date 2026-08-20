"""通用 CRUD 接口：通用集合查询/增/改/删/计数"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from services.dependencies import (
    DeleteRequest,
    InsertRequest,
    QueryRequest,
    UpdateRequest,
    get_db,
)

logger = logging.getLogger("scholar-admin.routes.crud")
router = APIRouter(tags=["CRUD"])


@router.post("/collections/{collection}/query")
async def query_documents(collection: str, body: QueryRequest):
    """通用查询 — 支持 where / order / offset / limit / select"""
    try:
        db = get_db()
        req = body.model_dump(exclude_none=True)
        return await db.query(collection=collection, **req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post("/collections/{collection}/insert")
async def insert_document(collection: str, body: InsertRequest):
    """通用插入 — data 可为 dict 或 list[dict]"""
    try:
        db = get_db()
        return await db.insert(collection=collection, data=body.data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"插入失败: {str(e)}")


@router.post("/collections/{collection}/update")
async def update_documents(collection: str, body: UpdateRequest):
    """通用更新 — 必须提供 where 条件"""
    try:
        db = get_db()
        return await db.update(
            collection=collection,
            where=body.where,
            data=body.data,
            upsert=body.upsert,
            multi=body.multi,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@router.post("/collections/{collection}/delete")
async def delete_documents(collection: str, body: DeleteRequest):
    """通用删除 — 必须提供 where 条件"""
    try:
        db = get_db()
        return await db.delete(
            collection=collection,
            where=body.where,
            multi=body.multi,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.post("/collections/{collection}/count")
async def count_documents(collection: str, body: QueryRequest):
    """通用计数 — 使用 where 条件过滤统计"""
    try:
        db = get_db()
        return await db.count(collection=collection, where=body.where)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计数失败: {str(e)}")
