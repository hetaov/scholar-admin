"""共享依赖：数据库/视觉服务单例 + 请求模型"""

from __future__ import annotations

from pydantic import BaseModel

from services.database import CloudBaseNoSQLClient
from services.volcano import VolcanoVisionService

# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

_db_client: CloudBaseNoSQLClient | None = None
_vision_service: VolcanoVisionService | None = None


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


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    where: dict = {}
    order: list | None = None
    offset: int = 0
    limit: int = 100
    select: dict | None = None


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


class ImageUrlRequest(BaseModel):
    url: str


class RecognizeBase64Request(BaseModel):
    base64: str
    mime_type: str = "image/jpeg"
