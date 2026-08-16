"""集成测试示例:tests/integration/ 模板

被测链路:FastAPI TestClient + FakeDB,覆盖:
- GET  /tracking/{scholar_id}  按学者查询 skill_state 聚合状态

模板要点:
- 依赖 conftest 提供的 client / fake_db fixture,验证"接口 → DB → 响应"完整链路。
- 断言响应状态码与关键字段结构,而不是逐字相等。
- 后续每新增一个接口,按此模板在 tests/integration/ 下新建 test_*.py。
"""

from __future__ import annotations


class TestGetTrackingByScholar:
    """GET /tracking/{scholar_id}"""

    def test_found(self, client, fake_db):
        fake_db.add(
            "skill_state",
            {
                "scholar_id": "scholar_1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "learned",
            },
        )
        resp = client.get("/tracking/scholar_1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["records"][0]["scholar_id"] == "scholar_1"

    def test_not_found_returns_empty(self, client):
        resp = client.get("/tracking/nobody")
        assert resp.status_code == 200
        assert resp.json()["records"] == []
