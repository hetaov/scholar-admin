"""CloudBase 云存储客户端（F4 错题扫描图片落对象存储）— 微信云开发 HTTP API 实现

背景（2026-08-21 真机故障链，逐环暴露）：
1. 403「未授权使用」：wx.uploadFile 直传云托管公网域名缺 X-WX-OPENID
   （已由前端 ensureOpenId 修复）。
2. UnsupportedProtocol：tcb.tencentcloudapi.com 的 UploadFile 拒绝
   Content-Type: application/json（该 action 只支持特定协议格式）。
3. AuthFailure.SignatureFailure：改为 multipart/form-data + TC3 签名后
   服务端仍验签失败。

结论：tcb.tencentcloudapi.com 的 UploadFile / GetTempFileURL 是文档缺失的
非标准 action（TCB OpenAPI 接口目录仅有 RunCommands 等数据库/云函数 action），
不再调用；改走微信云开发 HTTP API（api.weixin.qq.com/tcb/...，官方文档完整）：

- upload_file：POST /tcb/uploadfile 换取 COS 上传链接（url/token/authorization/
  cos_file_id）→ POST {url} 上传 multipart/form-data（key=path、Signature=
  authorization、x-cos-security-token=token、x-cos-meta-fileid=cos_file_id、
  file=文件二进制）。COS 签名在表单字段中携带，不校验请求体哈希，
  无 TC3 multipart 签名问题。
- get_temp_file_url：POST /tcb/batchdownloadfile 换取临时 HTTPS 下载链接。

鉴权：小程序 access_token（GET /cgi-bin/token，appid+secret 换取，2 小时过期，
模块级缓存、提前 300s 刷新）。配置：config.WX_APPID / config.WX_SECRET
（生产经云托管环境变量注入）。

契约：api-contract.md §3.10（F4 上传接口，image_url 出参）。
"""
from __future__ import annotations

import logging
import os
import time

import httpx

import config

logger = logging.getLogger("scholar-admin.tcb_storage")

# 微信云开发 HTTP API 基础地址
WX_API_HOST = "https://api.weixin.qq.com"

# access_token 模块级缓存（2h 过期，提前 300s 刷新；asyncio 单线程安全）
_TOKEN_CACHE: dict = {"token": "", "expires_at": 0}


class StorageAPIError(Exception):
    """云存储 API 错误（上传/取链接失败；由调用方包装为 StorageError 出参）"""


class CloudBaseStorageClient:
    """CloudBase 云存储客户端（微信云开发 HTTP API 实现）

    保持与旧实现（tcb OpenAPI）一致的接口：upload_file / get_temp_file_url，
    调用方（error_scanner.create_scan_upload）无感。
    """

    def __init__(
        self,
        env_id: str | None = None,
        appid: str | None = None,
        secret: str | None = None,
    ):
        self.env_id = env_id or config.ENV_ID
        self.appid = appid or config.WX_APPID
        self.secret = secret or config.WX_SECRET

    # ------------------------------------------------------------------
    # 鉴权：小程序 access_token
    # ------------------------------------------------------------------
    async def _get_access_token(self) -> str:
        """获取小程序 access_token（模块级缓存，提前 300s 刷新）"""
        now = time.time()
        if _TOKEN_CACHE["token"] and _TOKEN_CACHE["expires_at"] > now:
            return _TOKEN_CACHE["token"]
        if not self.appid or not self.secret:
            raise StorageAPIError(
                "缺少微信小程序凭据：请配置 WX_APPID / WX_SECRET 环境变量"
                "（小程序后台 → 开发管理 → 开发设置 → AppSecret）"
            )
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{WX_API_HOST}/cgi-bin/token",
                params={
                    "grant_type": "client_credential",
                    "appid": self.appid,
                    "secret": self.secret,
                },
            )
            data = resp.json()
        if data.get("errcode"):
            raise StorageAPIError(f"获取 access_token 失败: {data}")
        token = data["access_token"]
        expires_in = int(data.get("expires_in", 7200))
        _TOKEN_CACHE["token"] = token
        _TOKEN_CACHE["expires_at"] = now + expires_in - 300
        logger.info(f"[storage] access_token 获取成功，有效期 {expires_in}s")
        return token

    # ------------------------------------------------------------------
    # 上传：获取链接 → COS 表单直传
    # ------------------------------------------------------------------
    async def upload_file(self, cloud_path: str, data: bytes) -> dict:
        """上传文件到云存储

        Args:
            cloud_path: 云存储路径（相对根目录、不含前导 /，如 "scan/scan_xxx.jpg"）
            data: 文件二进制内容

        Returns:
            {"file_id": "cloud://env.xxx/scan/scan_xxx.jpg", "cloud_path": "..."}

        Raises:
            StorageAPIError: 微信 HTTP API / COS 上传失败
        """
        token = await self._get_access_token()
        # 1. 获取 COS 上传链接（微信云开发 HTTP API：/tcb/uploadfile）
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{WX_API_HOST}/tcb/uploadfile",
                params={"access_token": token},
                json={"env": self.env_id, "path": cloud_path},
            )
            upload = resp.json()
        if upload.get("errcode"):
            raise StorageAPIError(f"获取文件上传链接失败: {upload}")
        url = upload["url"]
        logger.info(f"[storage] 已获取上传链接 cloud_path={cloud_path}, url={url[:80]}...")
        # 2. COS 表单直传（multipart/form-data；签名在表单字段中，无 body 哈希校验）
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                data={
                    "key": cloud_path,
                    "Signature": upload["authorization"],
                    "x-cos-security-token": upload["token"],
                    "x-cos-meta-fileid": upload["cos_file_id"],
                },
                files={
                    "file": (os.path.basename(cloud_path) or "upload.jpg", data),
                },
            )
        if resp.status_code >= 400:
            raise StorageAPIError(
                f"COS 上传失败 HTTP {resp.status_code}: {resp.text[:300]}"
            )
        file_id = upload.get("file_id") or f"cloud://{self.env_id}/{cloud_path}"
        logger.info(f"[storage] UploadFile ok cloud_path={cloud_path} file_id={file_id}")
        return {"file_id": file_id, "cloud_path": cloud_path}

    # ------------------------------------------------------------------
    # 临时链接：/tcb/batchdownloadfile
    # ------------------------------------------------------------------
    async def get_temp_file_url(self, cloud_path: str, max_age: int = 3600) -> str:
        """获取云存储文件的临时 HTTPS 下载链接

        Args:
            cloud_path: 云存储路径（与上传时一致）
            max_age: 链接有效期（秒，默认 1 小时）

        Returns:
            临时 HTTPS URL；无返回时为空串（image_file_id 仍可永久引用）
        """
        token = await self._get_access_token()
        file_id = f"cloud://{self.env_id}/{cloud_path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{WX_API_HOST}/tcb/batchdownloadfile",
                params={"access_token": token},
                json={
                    "env": self.env_id,
                    "file_list": [{"fileid": file_id, "max_age": max_age}],
                },
            )
            data = resp.json()
        if data.get("errcode"):
            logger.warning(
                f"[storage] batchdownloadfile 失败 cloud_path={cloud_path}: {data}"
            )
            return ""
        files = data.get("file_list") or []
        if files:
            return files[0].get("download_url", "")
        logger.warning(f"[storage] batchdownloadfile 无返回 URL cloud_path={cloud_path}")
        return ""
