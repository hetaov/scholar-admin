"""集成测试示例:tests/integration/ 模板

被测链路:FastAPI TestClient + FakeDB,覆盖:
- GET  /tracking/{scholar_id}  按学者查询 learning_mastery_tracking
- POST /tracking/stats         学习进度统计(成功 + 参数校验)

模板要点:
- 依赖 conftest 提供的 client / fake_db fixture,验证"接口 → DB → 响应"完整链路。
- 断言响应状态码与关键字段结构,而不是逐字相等。
- 后续每新增一个接口,按此模板在 tests/integration/ 下新建 test_*.py。
"""

from __future__ import annotations

import pytest


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


class TestPostTrackingStats:
    """POST /tracking/stats"""

    def test_success(self, client):
        resp = client.post(
            "/tracking/stats",
            json={
                "scholar_id": "scholar_1",
                "text_book_id": "tb_1",
                "record_list": [
                    {"sentence_id": "sent_1", "time_spent": 120, "status": "learned"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["summary"]["total_time_spent"] == 120.0
        assert data["summary"]["learned_sentence_count"] == 1
        assert data["summary"]["textbook_progress"] == pytest.approx(0.25)

    def test_missing_scholar_id(self, client):
        resp = client.post("/tracking/stats", json={"text_book_id": "tb_1", "record_list": []})
        assert resp.status_code == 400
        assert "scholar_id" in resp.json()["detail"]
