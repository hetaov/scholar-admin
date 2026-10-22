"""单元测试示例:tests/unit/ 模板

被测对象:services.database.CloudBaseNoSQLClient._normalize_types
(Extended JSON → Python 原生类型的纯函数)

模板要点:
- 只测纯函数,不触网、不连库、不实例化客户端。
- 一个测试类对应一个被测函数;方法名 test_xxx 表达具体场景。
- 后续每新增一个纯函数/算法,按此模板在 tests/unit/ 下新建 test_*.py。
"""

from __future__ import annotations

from services.database import CloudBaseNoSQLClient

_normalize = CloudBaseNoSQLClient._normalize_types


class TestNormalizeTypes:
    """_normalize_types: Extended JSON 数字类型 → Python 原生类型"""

    def test_number_double(self):
        assert _normalize({"$numberDouble": "1.5"}) == 1.5

    def test_number_int(self):
        assert _normalize({"$numberInt": "3"}) == 3

    def test_number_long(self):
        assert _normalize({"$numberLong": "123456789"}) == 123456789

    def test_plain_values_passthrough(self):
        assert _normalize(42) == 42
        assert _normalize("abc") == "abc"
        assert _normalize(None) is None
        assert _normalize([1, "x"]) == [1, "x"]

    def test_nested_structure(self):
        raw = {"count": {"$numberInt": "2"}, "items": [{"price": {"$numberDouble": "9.9"}}]}
        assert _normalize(raw) == {"count": 2, "items": [{"price": 9.9}]}

    def test_plain_dict_unchanged(self):
        # 非 Extended JSON 的 dict 递归处理后保持原样
        assert _normalize({"a": 1, "b": {"c": 2}}) == {"a": 1, "b": {"c": 2}}

    # 2026-10-22 修复（见 wx-knowlege-graph/plan/NOTES.md N04）：
    # 漏掉 $oid 时 ObjectId 会以 {"$oid": ...} dict 返回给客户端，客户端
    # 拼 URL 时 str(dict) → "[object Object]"，而 JSON body 里是 {"$oid": ...}，
    # 两者不等 → 后端「body 主键 vs 路径参数」校验失败 → HTTP 400。
    def test_oid(self):
        assert _normalize({"$oid": "507f1f77bcf86cd799439011"}) == "507f1f77bcf86cd799439011"

    def test_oid_in_record(self):
        # 真实场景：query 返回的 records 里 _id 是 {"$oid": ...}
        raw = {"_id": {"$oid": "507f1f77bcf86cd799439011"}, "name": "小明", "count": {"$numberInt": "3"}}
        assert _normalize(raw) == {"_id": "507f1f77bcf86cd799439011", "name": "小明", "count": 3}

    def test_oid_in_record_list(self):
        raw = [{"_id": {"$oid": "aaa"}}, {"_id": {"$oid": "bbb"}}]
        assert _normalize(raw) == [{"_id": "aaa"}, {"_id": "bbb"}]
