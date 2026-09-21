"""M6 三维技能码收敛与 listening 兼容映射。"""

from __future__ import annotations

from services.routes_state import router as state_router
from services.routes_tracking import router as tracking_router
from tests.fakes.seed_factory import seed_content


def test_tracking_state_maps_legacy_listening_to_speaking(make_client, fake_db):
    client = make_client(state_router, tracking_router)

    response = client.post(
        "/tracking/state",
        json={
            "scholar_id": "s1",
            "sentence_id": "sent_1",
            "skill_code": "listening",
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["state"]["skill_code"] == "speaking"
    assert response.json()["data"]["attempt"]["skill_code"] == "speaking"


def test_tracking_state_rejects_unknown_skill_code(make_client, fake_db):
    client = make_client(state_router, tracking_router)

    response = client.post(
        "/tracking/state",
        json={
            "scholar_id": "s1",
            "sentence_id": "sent_1",
            "skill_code": "reading",
        },
    )

    assert response.status_code == 400
    assert "translation/conversation/speaking" in response.json()["detail"]


def test_books_summary_maps_legacy_listening_to_speaking(make_client, fake_db):
    seed_content(fake_db, lesson_ids=("l1",), sentence_ids=("s1",), include_text=False)
    fake_db.add(
        "skill_state",
        {
            "scholar_id": "s1",
            "sentence_id": "s1",
            "skill_code": "listening",
            "status": "learned",
            "mastery_score": 80,
            "attempt_count": 1,
        },
    )
    client = make_client(tracking_router)
    client.put(
        "/scholar/s1/books/tb_1/position",
        json={"current_lesson_id": "l1"},
    )

    response = client.get("/scholar/s1/books")

    assert response.status_code == 200
    skills = response.json()["data"]["books"][0]["summary"]["skills"]
    assert "listening" not in skills
    assert "speaking" in skills
