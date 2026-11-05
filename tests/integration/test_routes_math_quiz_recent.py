"""GET /math/quiz/recent 验收（描述出题恢复入口）

背景：描述出题为异步任务（30~120s），用户切走页面再回来拿不到 quiz_id。
本接口按 scholar 返回时间窗内最近一条 `mode="desc"` 出题，前端据此续轮询或回填。

覆盖场景：
- 命中最近一条描述出题（R3：items[].answer 不出参；带 status/file_refs）
- 只取 desc（拍照模式文档无 mode 字段 → 不返回）
- 多条取最新（created_at 倒序）
- 无记录 → data=null
- 缺 scholar_id → 400；mode 非 desc → 400
- 超出时间窗（>12h）→ data=null
- 路由顺序：/quiz/recent 不被 /quiz/{quiz_id} 吞掉（200 而非 404）
"""
from __future__ import annotations

from services.math.quiz import (
    MATH_QUIZ_COLLECTION,
    QUIZ_STATUS_FAILED,
    QUIZ_STATUS_GENERATING,
    RECENT_DESC_QUIZ_WINDOW_MS,
)
from services.routes_math import router as math_router

SCHOLAR = "scholar_recent_001"
NOW = 1_800_000_000_000


def _doc(quiz_id, *, mode="desc", created_at=NOW, **overrides):
    doc = {
        "quiz_id": quiz_id,
        "scholar_id": SCHOLAR,
        "items": [
            {
                "quiz_item_id": f"{quiz_id}-i1",
                "question": "3/7+2/7=?",
                "answer": "5/7",  # R3：不出参
            }
        ],
        "count_per_source": 2,
        "status": QUIZ_STATUS_GENERATING,
        "created_at": created_at,
        "file_refs": {"pdf": "", "png": "", "preview": ""},
        "warnings": [],
    }
    if mode is not None:
        doc["mode"] = mode
    doc.update(overrides)
    return doc


def _seed(fake_db, docs):
    for d in docs:
        fake_db.add(MATH_QUIZ_COLLECTION, d)


def test_recent_returns_latest_desc(make_client, fake_db, monkeypatch):
    monkeypatch.setattr("services.math.quiz._now_ms", lambda: NOW)
    _seed(
        fake_db,
        [
            _doc("q_old", created_at=NOW - 60_000),
            _doc(
                "q_new",
                created_at=NOW - 1_000,
                status="generated",
                items=[
                    {"quiz_item_id": "qi1", "question": "1/5+3/5=?", "answer": "4/5"}
                ],
            ),
        ],
    )
    client = make_client(math_router)
    res = client.get(f"/math/quiz/recent?scholar_id={SCHOLAR}&mode=desc")
    assert res.status_code == 200, res.text
    data = res.json()["data"]
    assert data["quiz_id"] == "q_new"
    assert data["status"] == "generated"
    assert data["items"][0]["question"] == "1/5+3/5=?"
    assert "answer" not in data["items"][0]  # R3


def test_recent_ignores_photo_mode(make_client, fake_db, monkeypatch):
    monkeypatch.setattr("services.math.quiz._now_ms", lambda: NOW)
    # 拍照模式文档无 mode 字段（存量口径）
    _seed(fake_db, [_doc("q_photo", mode=None, created_at=NOW - 500)])
    client = make_client(math_router)
    res = client.get(f"/math/quiz/recent?scholar_id={SCHOLAR}")
    assert res.status_code == 200
    assert res.json()["data"] is None


def test_recent_empty(make_client, fake_db, monkeypatch):
    monkeypatch.setattr("services.math.quiz._now_ms", lambda: NOW)
    client = make_client(math_router)
    res = client.get(f"/math/quiz/recent?scholar_id={SCHOLAR}")
    assert res.status_code == 200
    assert res.json()["data"] is None


def test_recent_stale_returns_null(make_client, fake_db, monkeypatch):
    monkeypatch.setattr("services.math.quiz._now_ms", lambda: NOW)
    _seed(
        fake_db,
        [
            _doc(
                "q_stale",
                created_at=NOW - RECENT_DESC_QUIZ_WINDOW_MS - 1,
                status="generated",
            )
        ],
    )
    client = make_client(math_router)
    res = client.get(f"/math/quiz/recent?scholar_id={SCHOLAR}")
    assert res.status_code == 200
    assert res.json()["data"] is None


def test_recent_failed_carries_error_message(make_client, fake_db, monkeypatch):
    monkeypatch.setattr("services.math.quiz._now_ms", lambda: NOW)
    _seed(
        fake_db,
        [
            _doc(
                "q_fail",
                status=QUIZ_STATUS_FAILED,
                error_message="未生成有效题目，请补充描述后重试",
                items=[],
            )
        ],
    )
    client = make_client(math_router)
    data = client.get(f"/math/quiz/recent?scholar_id={SCHOLAR}").json()["data"]
    assert data["status"] == "failed"
    assert "未生成有效题目" in data["error_message"]


def test_recent_missing_scholar_400(make_client, fake_db):
    client = make_client(math_router)
    assert client.get("/math/quiz/recent").status_code == 400


def test_recent_non_desc_mode_400(make_client, fake_db):
    client = make_client(math_router)
    res = client.get(f"/math/quiz/recent?scholar_id={SCHOLAR}&mode=photo")
    assert res.status_code == 400


def test_recent_route_not_shadowed_by_quiz_id(make_client, fake_db, monkeypatch):
    """路由顺序回归：/quiz/recent 必须先于 /quiz/{quiz_id} 注册（否则 404）。"""
    monkeypatch.setattr("services.math.quiz._now_ms", lambda: NOW)
    _seed(fake_db, [_doc("q_x", status="generated")])
    client = make_client(math_router)
    res = client.get(f"/math/quiz/recent?scholar_id={SCHOLAR}")
    assert res.status_code == 200
    assert res.json()["data"]["quiz_id"] == "q_x"
