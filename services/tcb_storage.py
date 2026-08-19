"""CloudBase 云存储客户端（F4 错题扫描图片落对象存储）

复用 database.CloudBaseNoSQLClient 的 TC3 签名与请求管道（service=tcb，
X-TC-Version=2018-06-08），凭据同样来自 config（SECRET_ID/SECRET_KEY/
SESSION_TOKEN/REGION/ENV_ID），无需新增任何依赖：

- UploadFile：上传文件到云存储，返回 fileID（cloud://env/xxx）
- GetTempFileURL：fileID → 临时 HTTPS 访问 URL（供小程序即时预览）

契约：api-contract.md §3.10（F4 上传接口，image_url 出参）。
"""
from __future__ import annotations

import base64
import logging

from services.database import CloudBaseNoSQLClient

logger = logging.getLogger("scholar-admin.tcb_storage")


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
        payload = {
            "EnvId": self.env_id,
            "FilePath": cloud_path,
            "CloudPath": cloud_path,
            "FileContent": base64.b64encode(data).decode("ascii"),
        }
        resp = await self._request("UploadFile", payload)
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
