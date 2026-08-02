"""CloudBase NoSQL 数据库 API - FastAPI 服务

提供对 CloudBase 文档型数据库的 RESTful API 接口。
支持集合管理和文档 CRUD 操作。
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional

from config import ENV_ID, REGION, PORT
from services.database import CloudBaseNoSQLClient

app = FastAPI(
    title="CloudBase NoSQL API",
    description=f"CloudBase 文档型数据库 API - 环境: {ENV_ID}",
    version="1.0.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库客户端（单例）
_db_client: Optional[CloudBaseNoSQLClient] = None


def get_db() -> CloudBaseNoSQLClient:
    global _db_client
    if _db_client is None:
        _db_client = CloudBaseNoSQLClient()
    return _db_client


# ==================== 请求/响应模型 ====================


class QueryRequest(BaseModel):
    where: dict = {}
    order: Optional[list] = None
    offset: int = 0
    limit: int = 100
    select: Optional[dict] = None


class InsertRequest(BaseModel):
    data: dict | list[dict]


class UpdateRequest(BaseModel):
    where: dict
    data: dict
    upsert: bool = False
    multi: bool = True


class DeleteRequest(BaseModel):
    where: dict
    multi: bool = True


# ==================== 健康检查 ====================


@app.get("/")
async def root():
    return {
        "service": "CloudBase NoSQL API",
        "env_id": ENV_ID,
        "region": REGION,
        "status": "running",
    }


@app.get("/health")
async def health():
    try:
        db = get_db()
        collections = await db.list_collections()
        return {
            "status": "healthy",
            "env_id": ENV_ID,
            "collections": [c.get("TableName") for c in collections],
            "collection_count": len(collections),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库连接失败: {str(e)}")


# ==================== 集合管理 ====================


@app.get("/collections")
async def list_collections():
    """列出所有集合"""
    db = get_db()
    collections = await db.list_collections()
    return {"collections": collections}


@app.get("/collections/{collection_name}")
async def check_collection(collection_name: str):
    """检查集合是否存在"""
    db = get_db()
    exists = await db.check_collection(collection_name)
    return {"collection": collection_name, "exists": exists}


# ==================== 文档 CRUD ====================


@app.post("/collections/{collection}/query")
async def query_documents(collection: str, req: QueryRequest):
    """查询文档

    示例 where 条件：
    - 精确匹配: {"name": "张三"}
    - 大于: {"age": {"$gt": 18}}
    - 逻辑与: {"$and": [{"age": {"$gt": 18}}, {"city": "北京"}]}
    """
    try:
        db = get_db()
        result = await db.query(
            collection=collection,
            where=req.where,
            order=req.order,
            offset=req.offset,
            limit=req.limit,
            select=req.select,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.post("/collections/{collection}/insert")
async def insert_documents(collection: str, req: InsertRequest):
    """插入文档"""
    try:
        db = get_db()
        result = await db.insert(collection=collection, data=req.data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"插入失败: {str(e)}")


@app.put("/collections/{collection}/update")
async def update_documents(collection: str, req: UpdateRequest):
    """更新文档

    data 使用 MongoDB 更新操作符：
    - 设置字段: {"$set": {"name": "新名称"}}
    - 自增: {"$inc": {"count": 1}}
    """
    try:
        db = get_db()
        result = await db.update(
            collection=collection,
            where=req.where,
            data=req.data,
            upsert=req.upsert,
            multi=req.multi,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@app.delete("/collections/{collection}/delete")
async def delete_documents(collection: str, req: DeleteRequest):
    """删除文档"""
    try:
        db = get_db()
        result = await db.delete(
            collection=collection,
            where=req.where,
            multi=req.multi,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@app.get("/collections/{collection}/count")
async def count_documents(
    collection: str,
    where: Optional[str] = Query(None, description="JSON 查询条件"),
):
    """统计文档数量"""
    import json

    try:
        where_dict = json.loads(where) if where else {}
        db = get_db()
        count = await db.count(collection=collection, where=where_dict)
        return {"collection": collection, "count": count}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="where 参数必须是有效的 JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"统计失败: {str(e)}")


# ==================== 启动配置 ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
