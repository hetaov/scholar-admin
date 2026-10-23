"""单元测试：渲染产物上云（a4_renderer 2026-10-23）

背景：小程序 `wx.downloadFile` 下载公网域名须 ICP 备案，故渲染产物（pdf/png/preview）
同步上传**云开发存储**，`file_refs` 增补 `file_ids`，前端改走 `wx.cloud.downloadFile`
（云开发 API 不校验「合法域名」）。

覆盖：
- 上云成功 → `file_refs.file_ids` 三种产物齐全，原静态 URL 口径不变
- 上云失败 → **不阻断渲染**：`file_refs` 无 file_ids，pdf 仍可用（降级回落静态下载）
- 开关 `RENDER_ARTIFACTS_TO_CLOUD=0` → 完全不上云
- 失败态（failed）不上传
"""
from __future__ import annotations

import asyncio

import pytest

import services.math.a4_renderer as ar
from services.math.a4_renderer import RENDER_JOB_FAILED, RENDER_JOB_SUCCESS

ENTITY_ID = "ps_cloud_1"


class FakeStorage:
    """云存储替身：记录上传路径；raise_error=True 模拟上云失败。"""

    instances: list = []

    def __init__(self, *args, **kwargs):
        self.uploaded: list[str] = []
        FakeStorage.instances.append(self)

    async def upload_file(self, cloud_path: str, data: bytes) -> dict:
        if getattr(self, "fail", False):
            raise RuntimeError("云存储不可用（模拟）")
        self.uploaded.append(cloud_path)
        return {"file_id": f"cloud://test/{cloud_path}", "cloud_path": cloud_path}


@pytest.fixture(autouse=True)
def _reset_storage():
    FakeStorage.instances = []
    yield
    FakeStorage.instances = []


@pytest.fixture
def artifacts_dir(tmp_path, monkeypatch):
    """产物目录：pdf/png/preview 三件套落盘 + 上云替身注入。"""
    monkeypatch.setattr(ar, "RENDER_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(ar, "RENDER_ARTIFACTS_TO_CLOUD", True)
    entity_dir = tmp_path / ENTITY_ID
    entity_dir.mkdir()
    for name in (ar.PDF_FILENAME, ar.PNG_FILENAME, ar.PREVIEW_FILENAME):
        (entity_dir / name).write_bytes(b"fake")
    monkeypatch.setattr("services.infra.tcb_storage.CloudBaseStorageClient", FakeStorage)
    return tmp_path


def test_upload_artifacts_to_cloud_returns_file_ids(artifacts_dir):
    out = asyncio.run(ar._upload_artifacts_to_cloud(ENTITY_ID))

    assert out == {
        "pdf": f"cloud://test/sheets/{ENTITY_ID}/{ar.PDF_FILENAME}",
        "png": f"cloud://test/sheets/{ENTITY_ID}/{ar.PNG_FILENAME}",
        "preview": f"cloud://test/sheets/{ENTITY_ID}/{ar.PREVIEW_FILENAME}",
    }
    assert FakeStorage.instances[0].uploaded == [
        f"sheets/{ENTITY_ID}/{ar.PDF_FILENAME}",
        f"sheets/{ENTITY_ID}/{ar.PNG_FILENAME}",
        f"sheets/{ENTITY_ID}/{ar.PREVIEW_FILENAME}",
    ]


def test_compose_artifacts_includes_file_ids_on_success(artifacts_dir):
    artifacts, file_refs = asyncio.run(
        ar._compose_artifacts_with_cloud(ENTITY_ID, RENDER_JOB_SUCCESS)
    )

    # 原静态 URL 口径完全不变（向后兼容）
    assert file_refs["pdf"].endswith(f"/{ENTITY_ID}/{ar.PDF_FILENAME}")
    assert file_refs["png"].endswith(f"/{ENTITY_ID}/{ar.PNG_FILENAME}")
    assert file_refs["preview"].endswith(f"/{ENTITY_ID}/{ar.PREVIEW_FILENAME}")
    # 新增 file_ids（cloud://）
    assert set(file_refs["file_ids"]) == {"pdf", "png", "preview"}
    assert file_refs["file_ids"]["pdf"].startswith("cloud://")
    assert artifacts["file_ids"] == file_refs["file_ids"]


def test_upload_failure_does_not_break_artifacts(artifacts_dir):
    """上云失败 → 降级：无 file_ids，静态 URL 仍可用（前端回落旧下载）。"""
    def _failing(*args, **kwargs):
        client = FakeStorage()
        client.fail = True
        return client

    import services.infra.tcb_storage as tcb_storage

    original = tcb_storage.CloudBaseStorageClient
    tcb_storage.CloudBaseStorageClient = _failing
    try:
        artifacts, file_refs = asyncio.run(
            ar._compose_artifacts_with_cloud(ENTITY_ID, RENDER_JOB_SUCCESS)
        )
    finally:
        tcb_storage.CloudBaseStorageClient = original

    assert "file_ids" not in file_refs
    assert "file_ids" not in artifacts
    assert file_refs["pdf"]  # 产物引用仍可用


def test_switch_off_skips_cloud_upload(artifacts_dir, monkeypatch):
    monkeypatch.setattr(ar, "RENDER_ARTIFACTS_TO_CLOUD", False)

    artifacts, file_refs = asyncio.run(
        ar._compose_artifacts_with_cloud(ENTITY_ID, RENDER_JOB_SUCCESS)
    )

    assert "file_ids" not in file_refs
    assert FakeStorage.instances == []  # 完全不实例化云存储客户端


def test_failed_status_skips_cloud_upload(artifacts_dir):
    artifacts, file_refs = asyncio.run(
        ar._compose_artifacts_with_cloud(ENTITY_ID, RENDER_JOB_FAILED)
    )

    assert file_refs == {"pdf": "", "png": "", "preview": ""}
    assert FakeStorage.instances == []
