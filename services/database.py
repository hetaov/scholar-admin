"""CloudBase 文档型数据库 NoSQL 操作客户端

基于腾讯云 API v3 签名，使用 RunCommands 接口封装对 CloudBase NoSQL 数据库的 CRUD 操作。
支持在 CloudRun 容器或本地环境中使用。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import datetime
from typing import Any, Optional

import httpx

from config import ENV_ID, REGION, SECRET_ID, SECRET_KEY, SESSION_TOKEN, TCB_API_HOST


class CloudBaseNoSQLClient:
    """CloudBase 文档型数据库客户端"""

    def __init__(
        self,
        env_id: str = ENV_ID,
        secret_id: str = SECRET_ID,
        secret_key: str = SECRET_KEY,
        session_token: str = SESSION_TOKEN,
        region: str = REGION,
    ):
        if not secret_id or not secret_key:
            raise ValueError(
                "缺少腾讯云 API 密钥。请设置环境变量：\n"
                "  export TENCENTCLOUD_SECRETID='你的SecretId'\n"
                "  export TENCENTCLOUD_SECRETKEY='你的SecretKey'\n"
                "密钥获取地址：https://console.cloud.tencent.com/cam/capi"
            )
        self.env_id = env_id
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.session_token = session_token
        self.region = region
        self.endpoint = f"https://{TCB_API_HOST}"

    def _sign_tc3(self, action: str, payload: dict) -> dict:
        """腾讯云 API v3 签名"""
        service = "tcb"
        algorithm = "TC3-HMAC-SHA256"
        timestamp = int(time.time())
        date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")

        # 1. 构建 CanonicalRequest
        http_request_method = "POST"
        canonical_uri = "/"
        canonical_querystring = ""
        ct = "application/json; charset=utf-8"
        payload_str = json.dumps(payload)
        canonical_headers = (
            f"content-type:{ct}\nhost:{TCB_API_HOST}\nx-tc-action:{action.lower()}\n"
        )
        signed_headers = "content-type;host;x-tc-action"
        hashed_payload = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        canonical_request = "\n".join(
            [
                http_request_method,
                canonical_uri,
                canonical_querystring,
                canonical_headers,
                signed_headers,
                hashed_payload,
            ]
        )

        # 2. 构建 StringToSign
        credential_scope = f"{date}/{service}/tc3_request"
        hashed_canonical = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
        string_to_sign = "\n".join(
            [algorithm, str(timestamp), credential_scope, hashed_canonical]
        )

        # 3. 计算签名
        def sign(key, msg):
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        secret_date = sign(("TC3" + self.secret_key).encode("utf-8"), date)
        secret_service = sign(secret_date, service)
        secret_signing = sign(secret_service, "tc3_request")
        signature = hmac.new(
            secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        # 4. 构建 Authorization
        authorization = (
            f"{algorithm} "
            f"Credential={self.secret_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )

        headers = {
            "Authorization": authorization,
            "Content-Type": ct,
            "Host": TCB_API_HOST,
            "X-TC-Action": action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": "2018-06-08",
            "X-TC-Region": self.region,
        }
        if self.session_token:
            headers["X-TC-Token"] = self.session_token

        return headers

    async def _request(self, action: str, payload: dict) -> dict:
        """发起腾讯云 API 请求"""
        headers = self._sign_tc3(action, payload)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.endpoint,
                json=payload,
                headers=headers,
            )
            result = resp.json()
            if "Response" in result and "Error" in result["Response"]:
                raise Exception(
                    f"API Error [{result['Response']['Error']['Code']}]: "
                    f"{result['Response']['Error']['Message']}"
                )
            return result.get("Response", result)

    # ==================== 通用命令执行 ====================

    async def _run_command(
        self, table_name: str, command_type: str, command: dict
    ) -> str:
        """执行单条数据库命令，返回 Data[0] 的 JSON 字符串"""
        payload = {
            "EnvId": self.env_id,
            "MgoCommands": [
                {
                    "TableName": table_name,
                    "CommandType": command_type,
                    "Command": json.dumps(command),
                }
            ],
        }
        resp = await self._request("RunCommands", payload)
        data_list: list = resp.get("Data", [])
        return data_list[0] if data_list else "[]"

    # ==================== 集合管理 ====================

    async def list_collections(self) -> list[dict]:
        """列出所有集合"""
        payload = {
            "EnvId": self.env_id,
            "MgoLimit": 100,
            "MgoOffset": 0,
        }
        resp = await self._request("DescribeTables", payload)
        return resp.get("Tables", [])

    async def check_collection(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        collections = await self.list_collections()
        return any(c.get("TableName") == collection_name for c in collections)

    # ==================== 文档查询 ====================

    async def query(
        self,
        collection: str,
        where: dict | None = None,
        order: list | None = None,
        offset: int = 0,
        limit: int = 100,
        select: dict | None = None,
    ) -> dict:
        """查询文档

        Args:
            collection: 集合名称
            where: 查询条件，MongoDB 风格，如 {"age": {"$gt": 18}}
            order: 排序，如 [{"field": "age", "direction": "desc"}]
            offset: 偏移量
            limit: 返回数量上限
            select: 指定返回字段，如 {"name": 1, "age": 1}
        """
        cmd: dict[str, Any] = {
            "find": collection,
            "filter": where or {},
            "skip": offset,
            "limit": max(limit, 1),
        }

        if order:
            sort_dict: dict[str, int] = {}
            for item in order:
                field = item.get("field", "")
                direction = item.get("direction", "asc")
                sort_dict[field] = 1 if direction == "asc" else -1
            cmd["sort"] = sort_dict

        if select:
            cmd["projection"] = select

        raw = await self._run_command(collection, "QUERY", cmd)
        records = json.loads(raw)
        return {
            "records": records,
            "total": len(records),
            "offset": offset,
            "limit": limit,
        }

    # ==================== 文档插入 ====================

    async def insert(self, collection: str, data: dict | list[dict]) -> dict:
        """插入文档

        Args:
            collection: 集合名称
            data: 要插入的文档或文档列表
        """
        if isinstance(data, dict):
            data = [data]

        cmd = {"insert": collection, "documents": data}
        raw = await self._run_command(collection, "INSERT", cmd)
        result = json.loads(raw)
        return {
            "inserted_count": result.get("n", len(data)),
            "ids": [str(doc.get("_id", "")) for doc in data],
        }

    # ==================== 文档更新 ====================

    async def update(
        self,
        collection: str,
        where: dict,
        data: dict,
        upsert: bool = False,
        multi: bool = True,
    ) -> dict:
        """更新文档

        Args:
            collection: 集合名称
            where: 查询条件
            data: 更新操作，如 {"$set": {"name": "new_name"}}
            upsert: 不存在时是否插入
            multi: 是否更新多条
        """
        update_entry: dict[str, Any] = {"q": where, "u": data, "multi": multi}
        if upsert:
            update_entry["upsert"] = True

        cmd = {"update": collection, "updates": [update_entry]}
        raw = await self._run_command(collection, "UPDATE", cmd)
        result = json.loads(raw)
        return {
            "matched_count": result.get("n", 0),
            "modified_count": result.get("nModified", 0),
            "upserted_id": str(result.get("upserted", [{}])[0].get("_id", "")) if result.get("upserted") else None,
        }

    # ==================== 文档删除 ====================

    async def delete(
        self, collection: str, where: dict, multi: bool = True
    ) -> dict:
        """删除文档

        Args:
            collection: 集合名称
            where: 查询条件
            multi: 是否删除多条
        """
        delete_limit = 0 if multi else 1  # MongoDB: 0 = 全部匹配, 1 = 仅一条
        cmd = {"delete": collection, "deletes": [{"q": where, "limit": delete_limit}]}
        raw = await self._run_command(collection, "DELETE", cmd)
        result = json.loads(raw)
        return {"deleted_count": result.get("n", 0)}

    # ==================== 文档统计 ====================

    async def count(self, collection: str, where: dict | None = None) -> int:
        """统计文档数量"""
        cmd: dict[str, Any] = {
            "count": collection,
            "query": where or {},
        }
        raw = await self._run_command(collection, "QUERY", cmd)
        result = json.loads(raw)
        # count 命令返回 {"n": number, ...}
        if isinstance(result, dict) and "n" in result:
            return result["n"]
        # 如果返回的是数组，取长度
        if isinstance(result, list):
            return len(result)
        return 0
