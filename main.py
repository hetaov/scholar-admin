"""CloudBase NoSQL 数据库 API + 火山引擎 AI 图片识别 — FastAPI 服务

提供对 CloudBase 文档型数据库的 RESTful API 接口，支持集合管理和文档 CRUD 操作。
提供火山引擎豆包视觉模型英文教材图片识别与语句提取接口。
"""
from __future__ import annotations

import base64 as b64_lib
import json as json_lib
import logging
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from config import ENV_ID, REGION, PORT
from services.database import CloudBaseNoSQLClient
from services.volcano import VolcanoVisionService

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scholar-admin")

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

# 火山视觉服务（单例）
_vision_service: Optional[VolcanoVisionService] = None


def get_db() -> CloudBaseNoSQLClient:
    global _db_client
    if _db_client is None:
        _db_client = CloudBaseNoSQLClient()
    return _db_client


def get_vision() -> VolcanoVisionService:
    global _vision_service
    if _vision_service is None:
        _vision_service = VolcanoVisionService()
    return _vision_service


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
    try:
        where_dict = json_lib.loads(where) if where else {}
        db = get_db()
        count = await db.count(collection=collection, where=where_dict)
        return {"collection": collection, "count": count}
    except json_lib.JSONDecodeError:
        raise HTTPException(status_code=400, detail="where 参数必须是有效的 JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"统计失败: {str(e)}")


# ==================== 学习掌握度追踪 ====================


@app.get("/tracking/{scholar_id}")
async def get_tracking_by_scholar_id(scholar_id: str):
    """根据 scholar_id 查询学习掌握度追踪记录

    查询 learning_mastery_tracking 集合中指定学者的所有掌握记录。

    文档结构：
    {
      "scholar_id": "6d758f346a6daee000859c332ed11089",
      "sentence_id": "dd2fe3f1-e797-477e-80d0-79b5fe6adfec",
      "status": 1,
      "times": 1,
      "unit_id": "f836a964-a4c3-4753-b9c3-14d159140f78",
      "update_time": [1785815342890]
    }
    """
    try:
        db = get_db()
        result = await db.query(
            collection="learning_mastery_tracking",
            where={"scholar_id": scholar_id},
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/textbook/{scholar_id}")
async def get_textbook_by_scholar_id(scholar_id: str):
    """根据 scholar_id 查询所有教材列表

    直接查询 textbook 集合，获取指定学者的全部教材。

    文档结构：
    {
      "scholar_id": "6d758f346a6daee000859c332ed11089",
      "title": "新概念2"
    }
    """
    try:
        db = get_db()
        result = await db.query(
            collection="textbook",
            where={"scholar_id": scholar_id},
        )
        logger.info(f"[查询] 查询 textbook 集合，scholar_id={scholar_id}，结果={result}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


# ==================== 火山引擎 AI 图片识别 ====================


async def _store_recognition_result(
    db: CloudBaseNoSQLClient,
    result: dict,
    image_source: str,
) -> dict:
    """将识别结果存入 unit / paragraph / sentence 三张表

    Args:
        db: 数据库客户端
        result: 识别返回的完整 JSON 结果
        image_source: 图片来源标识（upload / url / base64）

    Returns:
        {"unit_id": str, "paragraph_id": str, "sentence_count": int}
    """
    logger.info(f"[存储] 开始处理识别结果，image_source={image_source}")
    logger.info(f"[存储] result 顶层字段: {list(result.keys())}")
    logger.info(f"[存储] result.language = {result.get('language')!r}")
    logger.info(f"[存储] result.material_type = {result.get('material_type')!r}")
    logger.info(f"[存储] result.total_sentences = {result.get('total_sentences')!r}")
    logger.info(f"[存储] result.sentences 类型 = {type(result.get('sentences')).__name__}, 长度 = {len(result.get('sentences') or [])}")

    # 非英文材料不存储
    if not result.get("language"):
        logger.warning("[存储] 跳过：language 为空或 null，非英文材料")
        return {"stored": False, "reason": "not English material"}

    now = datetime.now(timezone.utc).isoformat()
    unit_id = str(uuid.uuid4())
    paragraph_id = str(uuid.uuid4())
    sentences = result.get("sentences", [])

    if not sentences:
        logger.warning("[存储] 跳过：sentences 为空列表，没有可存储的语句")
        return {"stored": False, "reason": "no sentences"}

    logger.info(f"[存储] 生成 unit_id={unit_id}, paragraph_id={paragraph_id}")

    # 1. 写入 unit
    unit_doc = {
        "unit_id": unit_id,
        "title": result.get("title", ""),
        "material_type": result.get("material_type", "other"),
        "language": result.get("language", "en"),
        "summary": result.get("summary", ""),
        "image_source": image_source,
        "total_sentences": len(sentences),
        "paragraph_count": 1,
        "created_at": now,
        "updated_at": now,
    }
    try:
        logger.info(f"[存储] 写入 unit 集合，文档 keys: {list(unit_doc.keys())}")
        unit_res = await db.insert(collection="unit", data=unit_doc)
        logger.info(f"[存储] unit 写入成功: {unit_res}")
    except Exception as e:
        logger.error(f"[存储] unit 写入失败: {type(e).__name__}: {e}")
        raise

    # 2. 写入 paragraph
    sentence_ids = [str(uuid.uuid4()) for _ in sentences]
    paragraph_doc = {
        "paragraph_id": paragraph_id,
        "unit_id": unit_id,
        "index": 1,
        "total_sentences": len(sentences),
        "sentence_ids": sentence_ids,
        "created_at": now,
    }
    try:
        logger.info(f"[存储] 写入 paragraph 集合")
        para_res = await db.insert(collection="paragraph", data=paragraph_doc)
        logger.info(f"[存储] paragraph 写入成功: {para_res}")
    except Exception as e:
        logger.error(f"[存储] paragraph 写入失败: {type(e).__name__}: {e}")
        raise

    # 3. 逐句写入 sentence
    try:
        logger.info(f"[存储] 开始写入 {len(sentences)} 条 sentence")
        for i, s in enumerate(sentences):
            sentence_doc = {
                "sentence_id": sentence_ids[i],
                "unit_id": unit_id,
                "paragraph_id": paragraph_id,
                "index": s.get("index", i + 1),
                "text": s.get("text", ""),
                "translation": s.get("translation", ""),
                "level": s.get("level", ""),
                "keywords": s.get("keywords", []),
                "created_at": now,
            }
            sen_res = await db.insert(collection="sentence", data=sentence_doc)
            logger.info(f"[存储] sentence[{i}] (index={s.get('index')}) 写入成功")
        logger.info(f"[存储] 全部 {len(sentences)} 条 sentence 写入完成")
    except Exception as e:
        logger.error(f"[存储] sentence 写入失败: {type(e).__name__}: {e}")
        raise

    logger.info(f"[存储] 全部存储完成！unit_id={unit_id}, sentences={len(sentences)}")
    return {
        "stored": True,
        "unit_id": unit_id,
        "paragraph_id": paragraph_id,
        "sentence_count": len(sentences),
    }


@app.post("/vision/recognize")
async def recognize_image(file: UploadFile = File(...)):
    """识别英文教材图片，提取语句并返回结构化 JSON

    上传一张英文教材/试题/教辅的图片，豆包视觉模型将：
    1. 识别图片是否为英文学习材料
    2. 提取所有英文语句
    3. 给出每句的 CEFR 等级、中文翻译、关键词
    4. 返回 JSON 结构化结果

    返回格式：
    {
      "language": "en",
      "material_type": "textbook|workbook|...",
      "title": "材料标题",
      "sentences": [
        {
          "index": 1,
          "text": "原文",
          "translation": "中文翻译",
          "level": "A1~C2",
          "keywords": ["word1", ...]
        }
      ],
      "total_sentences": N,
      "summary": "图片内容摘要"
    }
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="上传的图片为空")

        logger.info(f"[识别] 收到上传图片，大小={len(image_bytes)} bytes")

        vision = get_vision()
        result = vision.recognize(image_bytes=image_bytes)
        logger.info(f"[识别] 模型返回成功，准备写入数据库")

        # 自动存储到数据库
        db = get_db()
        store_info = await _store_recognition_result(db, result, "upload")
        result["_storage"] = store_info

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片识别失败: {str(e)}")


class ImageUrlRequest(BaseModel):
    url: str


@app.post("/vision/recognize-url")
async def recognize_image_url(req: ImageUrlRequest):
    """通过图片 URL 识别英文教材内容

    传入在线图片地址，效果与上传文件接口相同。
    """
    try:
        if not req.url:
            raise HTTPException(status_code=400, detail="url 不能为空")

        vision = get_vision()
        result = vision.recognize(image_url=req.url)

        # 自动存储到数据库
        db = get_db()
        store_info = await _store_recognition_result(db, result, "url")
        result["_storage"] = store_info

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片识别失败: {str(e)}")


class RecognizeBase64Request(BaseModel):
    base64: str
    mime_type: str = "image/jpeg"


@app.post("/vision/recognize-base64")
async def recognize_image_base64(req: RecognizeBase64Request):
    """通过 Base64 编码识别英文教材图片

    适用于前端 Canvas 截图、拍照后直接传 Base64 的场景。

    请求体：
    {
      "base64": "iVBORw0KGgo...（纯 Base64，不含 data:image 前缀）",
      "mime_type": "image/jpeg"  // 可选，默认 image/jpeg
    }
    """
    try:
        raw_b64 = req.base64
        # 如果用户传了 data:image/xxx;base64, 前缀，自动去除
        if "," in raw_b64 and raw_b64.startswith("data:"):
            raw_b64 = raw_b64.split(",", 1)[1]

        image_bytes = b64_lib.b64decode(raw_b64, validate=True)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Base64 解码后图片为空")

        vision = get_vision()
        result = vision.recognize(image_bytes=image_bytes)

        # 自动存储到数据库
        db = get_db()
        store_info = await _store_recognition_result(db, result, "base64")
        result["_storage"] = store_info

        return result

    except b64_lib.binascii.Error:
        raise HTTPException(status_code=400, detail="Base64 编码无效")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片识别失败: {str(e)}")


# ==================== 启动配置 ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
