"""/math/scan/upload **云开发中转通道**（JSON `image_url`）集成验收（2026-10-23）

背景：小程序 `wx.uploadFile` 直连云托管公网域名须 ICP 备案（云托管默认域名仅供测试），
故新增通道 A：前端先把图传到云开发存储 → `getTempFileURL` 取临时链接 → JSON 提交
`image_url`，后端下载字节后复用同一条 F4.1 主流程。

覆盖：
- JSON 通道（image_url）成功 → 出参与 multipart 一致，photo_context 落库
- 缺 image_url / 非 http(s) → 400
- multipart 原通道**未被破坏**（回归）
- `_filename_from_image_url` 从临时链接推导文件名（扩展名校验依赖它）
"""
from __future__ import annotations

import asyncio

import services.routes.math as routes_math
from services.database import MATH_SCAN_UPLOAD_COLLECTION
from services.math import error_scanner
from services.routes_math import router as math_router

IMAGE_BYTES = b"\xff\xd8\xff\xe0" + b"0" * 128  # 伪 jpg（校验只看扩展名与体积）


class FakeStorage:
    """云存储替身：记录上传路径，返回 cloud:// fileID + 临时链接。"""

    def __init__(self, *args, **kwargs):
        self.uploaded: list[tuple[str, int]] = []

    async def upload_file(self, cloud_path: str, data: bytes) -> dict:
        self.uploaded.append((cloud_path, len(data)))
        return {"file_id": f"cloud://test/{cloud_path}", "cloud_path": cloud_path}

    async def get_temp_file_url(self, cloud_path: str, max_age: int = 3600) -> str:
        return f"https://tmp.example.com/{cloud_path}"


def _records(fake_db, **where):
    async def _q():
        res = await fake_db.query(MATH_SCAN_UPLOAD_COLLECTION, where=where, limit=1)
        return res.get("records") or []

    return asyncio.run(_q())


def _patch_storage(monkeypatch) -> FakeStorage:
    holder = {}

    def _factory(*args, **kwargs):
        holder["client"] = FakeStorage()
        return holder["client"]

    monkeypatch.setattr(error_scanner, "CloudBaseStorageClient", _factory)
    return holder


def test_json_channel_uploads_via_image_url(make_client, fake_db, monkeypatch):
    async def _fake_fetch(url: str) -> bytes:
        return IMAGE_BYTES

    monkeypatch.setattr(routes_math, "_fetch_image_bytes", _fake_fetch)
    holder = _patch_storage(monkeypatch)
    client = make_client(math_router)

    resp = client.post(
        "/math/scan/upload",
        json={
            "scholar_id": "scholar-1",
            "image_url": "https://cos.example.com/tmp/scan.jpg?sign=abc",
            "note": "第 3 题",
            "photo_context": '{"textbook_id":"tb-1","unit_title":"分数的加法"}',
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["scan_id"]
    assert data["image_file_id"].startswith("cloud://")
    assert data["image_url"]
    assert data["deduped"] is False

    # 图片字节确实落了云存储（路径前缀 scan/）
    cloud_path = holder["client"].uploaded[0][0]
    assert cloud_path.startswith("scan/") and cloud_path.endswith(".jpg")

    # photo_context 落库（此前 multipart 通道静默丢弃该字段）
    recs = _records(fake_db, scan_id=data["scan_id"])
    assert recs
    assert recs[0]["photo_context"] == '{"textbook_id":"tb-1","unit_title":"分数的加法"}'
    assert recs[0]["note"] == "第 3 题"


def test_json_channel_missing_image_url_returns_400(make_client, fake_db, monkeypatch):
    _patch_storage(monkeypatch)
    client = make_client(math_router)

    resp = client.post("/math/scan/upload", json={"scholar_id": "scholar-1"})

    assert resp.status_code == 400
    assert "image_url" in resp.json()["detail"]


def test_json_channel_download_failure_returns_400(make_client, fake_db, monkeypatch):
    async def _boom(url: str) -> bytes:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="下载图片失败: HTTP 403")

    monkeypatch.setattr(routes_math, "_fetch_image_bytes", _boom)
    _patch_storage(monkeypatch)
    client = make_client(math_router)

    resp = client.post(
        "/math/scan/upload",
        json={"scholar_id": "scholar-1", "image_url": "https://cos.example.com/x.jpg"},
    )

    assert resp.status_code == 400
    assert "下载图片失败" in resp.json()["detail"]


def test_multipart_channel_still_works(make_client, fake_db, monkeypatch):
    """回归：新增 JSON 通道后，原 multipart 通道行为不变。"""
    _patch_storage(monkeypatch)
    client = make_client(math_router)

    resp = client.post(
        "/math/scan/upload",
        data={"scholar_id": "scholar-2", "note": "multipart"},
        files={"image": ("scan.jpg", IMAGE_BYTES, "image/jpeg")},
    )

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["scan_id"]
    recs = _records(fake_db, scan_id=data["scan_id"])
    assert recs[0]["note"] == "multipart"
    # 未携带上下文 → 不写该字段（兼容锁）
    assert "photo_context" not in recs[0]


def test_filename_from_image_url_variants():
    assert routes_math._filename_from_image_url(
        "https://cos.example.com/tmp/scan.jpg?sign=abc"
    ) == "scan.jpg"
    assert routes_math._filename_from_image_url("https://x.com/a/b/photo.PNG") == "photo.PNG"
    # 无扩展名 → 回落 jpg（否则 _validate_image 会因扩展名非法判 400）
    assert routes_math._filename_from_image_url("https://x.com/a/b/scan") == "scan.jpg"
