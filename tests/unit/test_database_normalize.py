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
