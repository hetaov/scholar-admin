"""集成测试: S4.3 AI Planner — GET /planner/next-action（契约 api-contract §3.9）

覆盖:
- 默认(PLANNER_ENABLED=1): cold_start / review / weakness 策略分支
- 幂等落库: learning_plan 按 scholar_id + plan_date upsert(重生成覆盖,created_at 保留)
- 返回结构: next_action{strategy, review_items, activities, difficulty, rationale}
- 回退(PLANNER_ENABLED=0): 与 S3.3 /training/recommend 同构(无 review_items/rationale)
- 参数校验: 缺 scholar_id → 400
"""

from __future__ import annotations

from datetime import datetime, timezone

import services.routes_planner as routes_planner_module
from services.routes_planner import router as planner_router
from tests.fakes.seed_factory import seed_content, seed_skill_states


def _ts(day: int, hour: int) -> int:
    """2026-08-day hour:00 UTC → 秒级时间戳(测试固定日期,避开"今天"漂移)。"""
    return int(datetime(2026, 8, day, hour, 0, 0, tzinfo=timezone.utc).timestamp())


class TestPlannerNextAction:
    """GET /planner/next-action"""

    def test_missing_scholar_id_400(self, make_client):
        client = make_client(planner_router)
        resp = client.get("/planner/next-action")
        assert resp.status_code == 400

    def test_cold_start(self, make_client, fake_db):
        """无历史 → cold_start: 标准引导序列 + 无 review_items + 落库 learning_plan。"""
        client = make_client(planner_router)
        resp = client.get(
            "/planner/next-action",
            params={"scholar_id": "scholar_1", "date": "2026-08-16"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        action = body["data"]["next_action"]
        assert action["strategy"] == "cold_start"
        assert action["activities"] == ["content", "shadowing", "translation", "listening"]
        assert action["review_items"] == []
        assert "冷启动" in action["rationale"]
        assert action["difficulty"] == 1  # COLD_START_DIFFICULTY
        # 幂等落库: learning_plan 文档写入
        docs = fake_db.all("learning_plan")
        assert len(docs) == 1
        plan = docs[0]
        assert plan["_id"] == "scholar_1_2026-08-16_plan"
        assert plan["scholar_id"] == "scholar_1"
        assert plan["plan_date"] == "2026-08-16"
        assert plan["strategy"] == "cold_start"
        assert plan["created_at"] and plan["updated_at"]

    def test_review_strategy(self, make_client, fake_db):
        """有到期复习项 → review: 复习优先,review_items 含 sentence 快照 + 到期时间。"""
        seed_content(fake_db)
        seed_skill_states(fake_db, [
            {
                "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
                "status": "learned", "mastery_score": 80, "attempt_count": 2,
                "next_review_at": _ts(16, 10), "last_outcome": "correct",
            },
            {
                "scholar_id": "scholar_1", "sentence_id": "s2", "skill_code": "translation",
                "status": "learning", "mastery_score": 40, "attempt_count": 1,
                "next_review_at": _ts(16, 12),
            },
        ])
        client = make_client(planner_router)
        resp = client.get(
            "/planner/next-action",
            params={"scholar_id": "scholar_1", "date": "2026-08-16"},
        )
        assert resp.status_code == 200
        action = resp.json()["data"]["next_action"]
        assert action["strategy"] == "review"
        assert len(action["review_items"]) == 2
        assert "1 个到期复习项" not in action["rationale"]
        assert "到期复习项" in action["rationale"]
        # 排序 + 快照字段
        assert [r["sentence_id"] for r in action["review_items"]] == ["s1", "s2"]
        assert action["review_items"][0]["content"] == "Text s1"
        assert action["review_items"][0]["next_review_at"].startswith("2026-08-16T10:00:00")
        assert action["review_items"][0]["last_result"] == "correct"

    def test_weakness_strategy(self, make_client, fake_db):
        """有历史但无到期复习且存在弱项(mastery < 0.6) → weakness。"""
        seed_content(fake_db)
        seed_skill_states(fake_db, [
            {
                "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "pronunciation",
                "status": "learning", "mastery_score": 35, "attempt_count": 2,
                "next_review_at": _ts(18, 10),  # 未来,非到期
            },
            {
                "scholar_id": "scholar_1", "sentence_id": "s2", "skill_code": "translation",
                "status": "learned", "mastery_score": 85, "attempt_count": 4,
                "next_review_at": _ts(18, 10),
            },
        ])
        client = make_client(planner_router)
        resp = client.get(
            "/planner/next-action",
            params={"scholar_id": "scholar_1", "date": "2026-08-16"},
        )
        assert resp.status_code == 200
        action = resp.json()["data"]["next_action"]
        assert action["strategy"] == "weakness"
        assert "pronunciation" in action["rationale"]
        assert action["review_items"] == []

    def test_upsert_idempotent(self, make_client, fake_db):
        """幂等: 同 scholar_id + plan_date 二次生成覆盖,不新增文档。"""
        client = make_client(planner_router)
        params = {"scholar_id": "scholar_1", "date": "2026-08-16"}
        client.get("/planner/next-action", params=params)
        client.get("/planner/next-action", params=params)
        docs = fake_db.all("learning_plan")
        assert len(docs) == 1

    def test_fallback_when_disabled(self, make_client, fake_db, monkeypatch):
        """PLANNER_ENABLED=0 → 回退 S3.3 同构响应(无 review_items/rationale)。"""
        monkeypatch.setattr(routes_planner_module, "PLANNER_ENABLED", False)
        seed_content(fake_db)
        client = make_client(planner_router)
        resp = client.get(
            "/planner/next-action",
            params={"scholar_id": "scholar_1", "date": "2026-08-16"},
        )
        assert resp.status_code == 200
        action = resp.json()["data"]["next_action"]
        assert "review_items" not in action
        assert "rationale" not in action
        assert "has_history" in action
        assert "gate_suggestion" in action
        assert "mastery" in action
        assert "skill_states" in action
        # 回退不落库 learning_plan
        assert fake_db.all("learning_plan") == []
