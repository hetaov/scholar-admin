"""CloudBase 文档型数据库 NoSQL 操作客户端

基于腾讯云 API v3 签名，使用 RunCommands 接口封装对 CloudBase NoSQL 数据库的 CRUD 操作。
支持在 CloudRun 容器或本地环境中使用。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import httpx

from config import ENV_ID, REGION, SECRET_ID, SECRET_KEY, SESSION_TOKEN, TCB_API_HOST

logger = logging.getLogger("scholar-admin.db")


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

    @staticmethod
    def _normalize_types(obj):
        """递归转换 Extended JSON 类型为 Python 原生类型

        MongoDB Extended JSON:
          {"$numberDouble": "1.5"} → 1.5
          {"$numberInt": "3"}     → 3
          {"$numberLong": "123"}  → 123
        """
        if isinstance(obj, dict):
            keys = list(obj.keys())
            if len(keys) == 1:
                key = keys[0]
                val = obj[key]
                if key == "$numberDouble":
                    return float(val)
                if key == "$numberInt":
                    return int(val)
                if key == "$numberLong":
                    return int(val)
            return {k: CloudBaseNoSQLClient._normalize_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [CloudBaseNoSQLClient._normalize_types(item) for item in obj]
        return obj

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
            raw_result = resp.json()
            logger.debug(f"[DB] {action} 原始响应类型={type(raw_result).__name__}, "
                         f"内容={json.dumps(raw_result, ensure_ascii=False)[:500]}")

            # 兼容 list 类型的响应（某些 CloudBase API 返回包装数组，甚至嵌套数组）
            result = raw_result
            while isinstance(result, list):
                if not result:
                    raise Exception(f"API 返回空列表 (action={action})")
                result = result[0]
            if not isinstance(result, dict):
                raise Exception(
                    f"API 返回非预期类型 {type(result).__name__} (action={action})"
                )

            if "Response" in result and "Error" in result["Response"]:
                error_code = result["Response"]["Error"]["Code"]
                error_msg = result["Response"]["Error"]["Message"]
                logger.error(f"[DB] API 错误 [{error_code}]: {error_msg}")
                raise Exception(
                    f"API Error [{error_code}]: {error_msg}"
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
        logger.info(f"[DB] _run_command → table={table_name}, type={command_type}")
        resp = await self._request("RunCommands", payload)
        data_list: list = resp.get("Data", [])
        logger.debug(
            f"[DB] _run_command resp Data len={len(data_list)}, "
            f"first type={type(data_list[0]).__name__ if data_list else 'None'}"
        )
        if not data_list:
            logger.warning(f"[DB] _run_command empty Data for {table_name}.{command_type}")
            return "{}"
        # 兼容 data_list[0] 已经是 dict/list 的情况
        first = data_list[0]
        logger.debug(
            f"[DB] _run_command first(type={type(first).__name__}) "
            f"preview={repr(str(first)[:200])}"
        )
        if isinstance(first, str):
            return first
        return json.dumps(first)

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

        # ── 逐层解析 + 日志，方便定位非法数据 ──
        logger.debug(f"[db.query] step0 raw(type={type(raw).__name__}): {repr(raw[:300])}")

        records = json.loads(raw)
        logger.debug(f"[db.query] step1 json.loads → type={type(records).__name__} len={getattr(records, '__len__', lambda: 0)()}")
        if isinstance(records, list) and len(records) <= 3:
            logger.debug(f"[db.query] step1 sample={repr(records)}")

        # 双重 JSON 编码兜底：若解析后仍是字符串，再解一层
        if isinstance(records, str):
            records = json.loads(records)
            logger.debug(f"[db.query] step2 double-decode → type={type(records).__name__}")

        # 整个结果可能是 dict（单条文档被包成了对象）
        if isinstance(records, dict):
            # 关键日志：dict 的 key 列表，方便确认是 {records} 包裹还是裸文档
            keys = list(records.keys())
            logger.info(
                f"[db.query] step3 dict keys={keys[:10]} "
                f"sample_v={repr({k: type(v).__name__ for k, v in list(records.items())[:3]})}"
            )
            records = [records]

        # CloudBase NoSQL find 命令返回的文档是 JSON 字符串，需要二次解析
        if isinstance(records, list):
            decoded: list = []
            for i, r in enumerate(records):
                t = type(r).__name__
                logger.debug(f"[db.query] step4 elem[{i}] type={t}")
                if isinstance(r, str):
                    try:
                        r = json.loads(r)
                        logger.debug(f"[db.query] step4 elem[{i}] decoded → {type(r).__name__}")
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"[db.query] step4 elem[{i}] JSON decode fail, keeping raw str")
                decoded.append(r)
            records = decoded
        else:
            logger.warning(f"[db.query] step4 unexpected type after decode: {type(records).__name__}")

        # 转换 Extended JSON 类型为 Python 原生类型
        before_normalize_type = type(records).__name__
        records = self._normalize_types(records)
        after_normalize_type = type(records).__name__
        logger.debug(
            f"[db.query] step5 _normalize_types: "
            f"{before_normalize_type} → {after_normalize_type}"
            f" len={getattr(records, '__len__', lambda: 0)()}"
        )
        # 诊断：如果类型变了，输出转换后的前 3 个元素类型
        if isinstance(records, list) and len(records) > 0 and len(records) <= 3:
            logger.debug(
                f"[db.query] step5 sample_types={[(type(e).__name__, repr(str(e)[:60])) for e in records]}"
            )

        # 兜底：_normalize_types 不应破坏列表结构，但防御一下
        if not isinstance(records, list):
            logger.warning(
                f"[db.query] step6 _normalize_types returned {type(records).__name__} instead of list, wrapping"
            )
            records = [records] if records else []

        logger.info(f"[db.query] done → records_count={len(records)}")
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

        logger.info(f"[DB] insert → collection={collection}, doc_count={len(data)}, env={self.env_id}")
        cmd = {"insert": collection, "documents": data}
        raw = await self._run_command(collection, "INSERT", cmd)
        result = json.loads(raw) if isinstance(raw, str) else raw

        # 兼容 result 是 list 的情况
        if isinstance(result, list):
            logger.warning(f"[DB] insert 返回 list (len={len(result)}), 但视为成功")
            return {"inserted_count": len(data), "ids": [""] * len(data)}

        logger.info(f"[DB] insert 完成 → collection={collection}, 返回={json.dumps(result, ensure_ascii=False)[:300]}")
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
        logger.debug(
            f"[db.update] raw(type={type(raw).__name__}) "
            f"preview={repr(raw[:300] if isinstance(raw, str) else str(raw)[:300])}"
        )
        result = json.loads(raw)
        logger.debug(
            f"[db.update] after json.loads → type={type(result).__name__} "
            f"value={repr(result)[:300]}"
        )
        # CloudBase 某些版本将 update 结果包在单元素列表中
        if isinstance(result, list):
            logger.debug(f"[db.update] result is list(len={len(result)}), unwrap")
            result = result[0] if result else {}
        # 兜底：CloudBase 可能直接返回纯字符串（如 "ok"）
        if isinstance(result, str):
            logger.warning(
                f"[db.update] result is str '{result}', treating as success"
            )
            return {
                "matched_count": 1,
                "modified_count": 1,
                "upserted_id": None,
            }
        if not isinstance(result, dict):
            logger.error(
                f"[db.update] unexpected type={type(result).__name__}, "
                f"value={repr(result)[:300]}"
            )
            return {
                "matched_count": 0,
                "modified_count": 0,
                "upserted_id": None,
            }
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
        logger.debug(
            f"[db.delete] raw(type={type(raw).__name__}) "
            f"preview={repr(raw[:300] if isinstance(raw, str) else str(raw)[:300])}"
        )
        result = json.loads(raw)
        logger.debug(
            f"[db.delete] after json.loads → type={type(result).__name__} "
            f"value={repr(result)[:300]}"
        )
        if isinstance(result, list):
            logger.debug(f"[db.delete] result is list(len={len(result)}), unwrap")
            result = result[0] if result else {}
        if isinstance(result, str):
            logger.warning(
                f"[db.delete] result is str '{result}', treating as success"
            )
            return {"deleted_count": 1}
        if not isinstance(result, dict):
            logger.error(
                f"[db.delete] unexpected type={type(result).__name__}, "
                f"value={repr(result)[:300]}"
            )
            return {"deleted_count": 0}
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
