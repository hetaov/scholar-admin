"""通用内存 FakeDB —— 模拟 services.database.CloudBaseNoSQLClient 的接口

用途:
- 单元/集成测试中替代真实 CloudBase,不触网、不依赖密钥。
- 支持多集合,数据按集合名隔离,可注入种子数据。

实现的能力(与 CloudBaseNoSQLClient 对应):
- query  : where 等值过滤 + MongoDB 操作符($gt/$gte/$lt/$lte/$ne/$in)、
           order 排序、offset/limit 分页、select 字段投影
- insert : 插入单条/多条,自动补 _id
- update : 支持 $set,可 upsert / multi
- delete : 按 where 删除,可 multi
- count  : 统计匹配文档数

约定:
- 返回 dict 结构与真实客户端保持一致,便于接口层测试直接断言。
- 写入/读取都做深拷贝,避免测试之间相互污染。
"""

from __future__ import annotations

import copy
from typing import Any

_OPERATORS = ("$gt", "$gte", "$lt", "$lte", "$ne", "$in")


def _match(where: dict | None, doc: dict) -> bool:
    """判断 doc 是否匹配 where(等值 + 操作符)。"""
    if not where:
        return True
    for field, cond in where.items():
        if field not in doc:
            return False
        value = doc[field]
        if isinstance(cond, dict):
            for op, target in cond.items():
                if op not in _OPERATORS:
                    continue
                if op == "$gt" and not (value > target):
                    return False
                if op == "$gte" and not (value >= target):
                    return False
                if op == "$lt" and not (value < target):
                    return False
                if op == "$lte" and not (value <= target):
                    return False
                if op == "$ne" and not (value != target):
                    return False
                if op == "$in" and value not in target:
                    return False
        elif value != cond:
            return False
    return True


def _sort_key(value: Any) -> tuple:
    """排序键:None 恒排最后,其余按原类型比较。"""
    return (0, value) if value is not None else (1, 0)


def _changes(data: dict) -> dict:
    """update 载荷:支持 {$set: {...}} 或直接字段。"""
    if isinstance(data, dict) and "$set" in data:
        return data["$set"]
    return data or {}


class FakeDB:
    """内存版 CloudBaseNoSQLClient,接口签名与真实客户端对齐。"""

    def __init__(self, seed: dict[str, list[dict]] | None = None):
        self._data: dict[str, list[dict]] = {}
        for name, rows in (seed or {}).items():
            self._data[name] = [copy.deepcopy(r) for r in rows]

    # ---------------- 测试辅助 ----------------

    def add(self, collection: str, doc: dict) -> dict:
        """向集合追加一条文档(测试准备数据用),自动补 _id。"""
        new = copy.deepcopy(doc)
        new.setdefault("_id", f"{collection}_{len(self._data.get(collection, [])) + 1}")
        self._data.setdefault(collection, []).append(new)
        return new

    def all(self, collection: str) -> list[dict]:
        """读取集合全部文档(深拷贝)。"""
        return copy.deepcopy(self._data.get(collection, []))

    def clear(self, collection: str | None = None) -> None:
        if collection:
            self._data.pop(collection, None)
        else:
            self._data.clear()

    # ---------------- 集合管理 ----------------

    async def list_collections(self) -> list[dict]:
        """列出所有集合，结构对齐真实客户端的 DescribeTables 返回。"""
        return [{"TableName": name} for name in self._data]

    async def check_collection(self, collection_name: str) -> bool:
        """检查集合是否存在。"""
        return collection_name in self._data

    async def create_collection(self, collection_name: str) -> dict:
        """创建集合（表）。"""
        self._data.setdefault(collection_name, [])
        return {"created": True}

    async def delete_collection(self, collection_name: str) -> dict:
        """删除集合（表），会清除其中全部文档。"""
        self._data.pop(collection_name, None)
        return {"deleted": True}

    # ---------------- 查询 ----------------

    async def query(
        self,
        collection: str,
        where: dict | None = None,
        order: list | None = None,
        offset: int = 0,
        limit: int = 100,
        select: dict | None = None,
    ) -> dict:
        rows = [r for r in self._data.get(collection, []) if _match(where, r)]
        if order:
            # 稳定排序:从最后一个排序键开始,保证多键排序正确
            for spec in reversed(order):
                field = spec.get("field", "")
                reverse = spec.get("direction", "asc") == "desc"
                rows.sort(key=lambda r, f=field: _sort_key(r.get(f)), reverse=reverse)
        total = len(rows)
        page = [copy.deepcopy(r) for r in rows[offset : offset + limit]]
        if select:
            keys = set(select)
            page = [{k: r[k] for k in keys if k in r} for r in page]
        # 2026-08-20 SOP G0.1：与真实 database.py 同步的 GETTER 兼容层
        # 对 textbook_v2 集合的返回记录注入 subject_type=english（存量兼容），
        # 保证 integration 测试 GET /textbook 行为与生产完全一致。
        if collection == "textbook_v2":
            try:
                from services.models_content import normalize_textbook_doc
                page = [normalize_textbook_doc(r) for r in page]
            except Exception:  # pragma: no cover - 防御性
                pass
        return {"records": page, "total": total, "offset": offset, "limit": limit}

    # ---------------- 写操作 ----------------

    async def insert(self, collection: str, data: dict | list[dict]) -> dict:
        docs = [data] if isinstance(data, dict) else data
        ids: list[str] = []
        for doc in docs:
            new = self.add(collection, doc)
            ids.append(str(new["_id"]))
        return {"inserted_count": len(ids), "ids": ids}

    async def update(
        self,
        collection: str,
        where: dict,
        data: dict,
        upsert: bool = False,
        multi: bool = True,
    ) -> dict:
        changes = _changes(data)
        pool = self._data.setdefault(collection, [])
        matched = modified = 0
        for doc in pool:
            if not _match(where, doc):
                continue
            matched += 1
            if multi or matched == 1:
                doc.update(copy.deepcopy(changes))
                modified += 1
            if not multi:
                break
        upserted_id = None
        if upsert and matched == 0:
            new_doc = copy.deepcopy(where)
            new_doc.update(copy.deepcopy(changes))
            self.add(collection, new_doc)
            upserted_id = str(new_doc.get("_id"))
        return {"matched_count": matched, "modified_count": modified, "upserted_id": upserted_id}

    async def delete(self, collection: str, where: dict, multi: bool = True) -> dict:
        pool = self._data.get(collection, [])
        deleted = 0
        for doc in list(pool):
            if not _match(where, doc):
                continue
            if not multi and deleted >= 1:
                break
            pool.remove(doc)
            deleted += 1
        return {"deleted_count": deleted}

    async def count(self, collection: str, where: dict | None = None) -> int:
        return sum(1 for r in self._data.get(collection, []) if _match(where, r))
