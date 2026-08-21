"""CloudBase 云存储客户端（F4 错题扫描图片落对象存储）

复用 database.CloudBaseNoSQLClient 的 TC3 签名与请求管道（service=tcb，
X-TC-Version=2018-06-08），凭据同样来自 config（SECRET_ID/SECRET_KEY/
SESSION_TOKEN/REGION/ENV_ID），无需新增任何依赖：

- UploadFile：上传文件到云存储，返回 fileID（cloud://env/xxx）
- GetTempFileURL：fileID → 临时 HTTPS 访问 URL（供小程序即时预览）

注意：UploadFile 接口要求 multipart/form-data（Content-Type: application/json
会返回 UnsupportedProtocol「this action does not support Content-Type=...」），
因此不能像数据库 action 那样发 JSON body，须按 _build_multipart_upload 构造
multipart 字节，并把含 boundary 的 Content-Type 一并参与 TC3 签名。

契约：api-contract.md §3.10（F4 上传接口，image_url 出参）。
"""
from __future__ import annotations

import logging
import os
import uuid

from services.database import CloudBaseNoSQLClient

logger = logging.getLogger("scholar-admin.tcb_storage")


def _build_multipart_upload(
    fields: dict,
    file_field: str,
    file_bytes: bytes,
    filename: str,
) -> tuple[bytes, str]:
    """构造 multipart/form-data 请求体

    Args:
        fields: 普通表单字段（如 EnvId / FilePath / CloudPath）
        file_field: 文件字段名（TCB UploadFile 为 FileContent）
        file_bytes: 文件原始二进制
        filename: 上传文件名（进入 Content-Disposition）

    Returns:
        (body, content_type)；content_type 含 boundary，必须随 body
        一同交给 _request 参与 TC3 签名（_build_tc3_headers 按实际发送
        Content-Type 计算 canonical headers）。
    """
    boundary = "----TCBFormBoundary" + uuid.uuid4().hex
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class CloudBaseStorageClient(CloudBaseNoSQLClient):
    """CloudBase 云存储客户端（继承 DB 客户端，复用 TC3 签名与 _request）"""

    async def upload_file(self, cloud_path: str, data: bytes) -> dict:
        """上传文件到云存储

        Args:
            cloud_path: 云存储路径（相对根目录、不含前导 /，如 "scan/scan_xxx.jpg"）
            data: 文件二进制内容

        Returns:
            {"file_id": "cloud://env.xxx/scan/scan_xxx.jpg", "cloud_path": "..."}

        Raises:
            Exception: TCB API 错误（由 _request 抛出）
        """
        # UploadFile 要求 multipart/form-data；body 与含 boundary 的 Content-Type
        # 同时参与 TC3 签名，避免服务端重算 sha256 失配（AuthFailure.SignatureFailure）。
        body, content_type = _build_multipart_upload(
            {
                "EnvId": self.env_id,
                "FilePath": cloud_path,
                "CloudPath": cloud_path,
            },
            file_field="FileContent",
            file_bytes=data,
            filename=os.path.basename(cloud_path) or "upload.jpg",
        )
        resp = await self._request(
            "UploadFile", None, body=body, content_type=content_type
        )
        file_id = resp.get("FileId") or f"cloud://{self.env_id}/{cloud_path}"
        logger.info(f"[storage] UploadFile ok cloud_path={cloud_path} file_id={file_id}")
        return {"file_id": file_id, "cloud_path": cloud_path}

    async def get_temp_file_url(self, cloud_path: str, max_age: int = 3600) -> str:
        """获取云存储文件的临时 HTTPS 访问 URL

        Args:
            cloud_path: 云存储路径（与上传时一致）
            max_age: 临时链接有效期（秒，默认 1 小时）

        Returns:
            临时 HTTPS URL；无返回时为空串（image_file_id 仍可永久引用）
        """
        payload = {
            "EnvId": self.env_id,
            "FileList": [{"CloudPath": cloud_path, "MaxAge": max_age}],
        }
        resp = await self._request("GetTempFileURL", payload)
        urls = resp.get("FileURLs") or []
        if urls:
            return urls[0].get("TempFileURL", "")
        logger.warning(f"[storage] GetTempFileURL 无返回 URL cloud_path={cloud_path}")
        return ""
