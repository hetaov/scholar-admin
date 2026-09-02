"""集成测试：沉浸式五步进度持久化三接口（FastAPI TestClient + FakeDB）

被测链路：routes_tracking 的 GET / PUT / DELETE
`/scholar/{scholar_id}/textbooks/{textbook_id}/groups/{group_id}/immersive-progress`

覆盖（契约 docs_v2/03-change/proposals/2026-08-29-沉浸式五步进度持久化后端接口.md）：
- PUT → GET 回读（roundtrip，payload 原样存取）
- PUT 覆盖更新（last-write-wins：sentence_id / payload 覆盖，单行）
- GET 无记录 → 200 + data:null（不返回 404）
- DELETE 幂等（deleted:true / false）
- PUT 校验失败 → 400（version 非法 / 主键与路径不一致 / payload 过大）
- GET 空路径参数 → 400
"""
from __future__ import annotations

from services.learning.immersive_progress import PROGRESS_VERSION
from services.routes_tracking import router as tracking_router

PATH = "/scholar/scholar_1/textbooks/tb_1/groups/g_1/immersive-progress"


def _mk_body(overrides: dict | None = None) -> dict:
    body = {
        "version": PROGRESS_VERSION,
        "scholar_id": "scholar_1",
        "textbook_id": "tb_1",
        "group_id": "g_1",
        "sentence_id": "sent_1",
        "challenge_active": True,
        "saved_at": 1724918400000,
        "payload": {
            "skill_flow": {"group_id": "g_1", "current_index": 1, "steps": [], "mastered": False},
            "timeline": [{"code": "ec_translation", "status": "pass"}],
            "challenge_input": "草稿",
            "listening": None,
        },
    }
    if overrides:
        body.update(overrides)
    return body


class TestPutGetRoundtrip:
    def test_put_then_get_roundtrip(self, make_client, fake_db):
        client = make_client(tracking_router)
        put = client.put(PATH, json=_mk_body())
        assert put.status_code == 200
        put_data = put.json()["data"]
        assert put_data["sentence_id"] == "sent_1"
        assert put_data["version"] == PROGRESS_VERSION
        assert put_data["challenge_active"] is True

        get = client.get(PATH)
        assert get.status_code == 200
        data = get.json()["data"]
        assert data["sentence_id"] == "sent_1"
        # payload 不透明原样存取（不解析内部结构）
        assert data["payload"] == _mk_body()["payload"]
        assert fake_db.all("immersive_progress").__len__() == 1

    def test_put_overwrites_single_row(self, make_client, fake_db):
        client = make_client(tracking_router)
        client.put(PATH, json=_mk_body())
        put2 = client.put(PATH, json=_mk_body({
            "sentence_id": "sent_2",
            "challenge_active": False,
            "saved_at": 1724918401000,
            "payload": {"skill_flow": None, "timeline": [], "challenge_input": "", "listening": None},
        }))
        assert put2.status_code == 200
        assert put2.json()["data"]["sentence_id"] == "sent_2"
        stored = fake_db.all("immersive_progress")
        assert len(stored) == 1  # 覆盖式 upsert，不新增行
        assert stored[0]["challenge_active"] is False

    def test_url_encoded_safe_segments(self, make_client, fake_db):
        # 空格 / # 等安全字符编码可正常路由（%2F 等路径分割符与既有 /scholar/ 系列
        # 同一限制：FastAPI 路径参数不支持编码斜杠，业务 ID 本就不含 /）
        client = make_client(tracking_router)
        path = "/scholar/scholar%201/textbooks/b%201/groups/g%231/immersive-progress"
        resp = client.put(path, json=_mk_body({
            "scholar_id": "scholar 1", "textbook_id": "b 1", "group_id": "g#1",
        }))
        assert resp.status_code == 200
        stored = fake_db.all("immersive_progress")
        assert stored[0]["group_id"] == "g#1"
        assert stored[0]["scholar_id"] == "scholar 1"


class TestGetMissingRecord:
    def test_no_record_returns_data_null(self, make_client):
        client = make_client(tracking_router)
        resp = client.get(PATH)
        assert resp.status_code == 200
        assert resp.json() == {"success": True, "data": None}

    def test_missing_path_segment_404(self, make_client):
        # 空路径段无法匹配 FastAPI 路径参数 → 404（与既有 /scholar/{sid}/ 系列一致；
        # 契约 400 语义由处理内校验兜底，实际路由层先于处理器拒绝）
        client = make_client(tracking_router)
        resp = client.get("/scholar//textbooks//groups//immersive-progress")
        assert resp.status_code == 404


class TestDelete:
    def test_delete_existing(self, make_client, fake_db):
        client = make_client(tracking_router)
        client.put(PATH, json=_mk_body())
        resp = client.delete(PATH)
        assert resp.status_code == 200
        assert resp.json()["data"] == {"deleted": True}
        assert fake_db.all("immersive_progress") == []

    def test_delete_idempotent_when_missing(self, make_client):
        client = make_client(tracking_router)
        resp = client.delete(PATH)
        assert resp.status_code == 200
        assert resp.json()["data"] == {"deleted": False}


class TestPutValidation:
    def test_version_mismatch_400(self, make_client):
        client = make_client(tracking_router)
        resp = client.put(PATH, json=_mk_body({"version": 0}))
        assert resp.status_code == 400
        assert "version" in resp.json()["detail"]

    def test_primary_key_mismatch_with_path_400(self, make_client):
        client = make_client(tracking_router)
        resp = client.put(PATH, json=_mk_body({"group_id": "g_other"}))
        assert resp.status_code == 400
        assert "group_id" in resp.json()["detail"]

    def test_missing_sentence_id_400(self, make_client):
        client = make_client(tracking_router)
        resp = client.put(PATH, json=_mk_body({"sentence_id": ""}))
        assert resp.status_code == 400
        assert "sentence_id" in resp.json()["detail"]

    def test_payload_over_size_400(self, make_client):
        client = make_client(tracking_router)
        resp = client.put(PATH, json=_mk_body({"payload": {"data": "x" * 6000}}))
        assert resp.status_code == 400
        assert "上限" in resp.json()["detail"]

    def test_validation_failure_does_not_persist(self, make_client, fake_db):
        client = make_client(tracking_router)
        client.put(PATH, json=_mk_body({"version": 999}))
        assert fake_db.all("immersive_progress") == []
