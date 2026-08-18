"""集成测试：P2 后续扩展功能（F6 每日目标 / F8 徽章·排行榜 / F11 错题本 / F10 对话场景）

被测链路（FastAPI TestClient + FakeDB，不触网）：
- POST /tracking/daily-goal                  每日目标（规则引擎：avg7×1.2 封顶/兜底）
- GET  /tracking/leaderboard                 排行榜（周期/指标/名次/隐私）
- GET  /tracking/{sid}/badges                徽章墙（条件聚合 + 幂等发放）
- GET  /tracking/{sid}/wrong-book            错题本（错误类型分布/降序）
- POST /match/dialogue/task                  对话任务（scenario/sessionId 透传，F10）

覆盖（契约 §3.7 口径）：
- 无数据回落空态（success:true），不抛业务错误
- 参数校验（缺 scholar_id / limit 越界 / period·metric 非法 → 400）
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from services.routes_dialogue import router as dialogue_router
from services.routes_tracking import router as tracking_router

BADGE_COLLECTION = "badge"
SCHOLAR_BADGE_COLLECTION = "scholar_badge"
SCHOLARS_COLLECTION = "scholars"


def _ts(days_ago: int, hour: int = 10) -> int:
    """今天往前 days_ago 天、hour 时(UTC)的时间戳,测试不依赖系统日期。"""
    day = datetime.now(timezone.utc).date() - timedelta(days=days_ago)
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).timestamp())


def _attempt(
    aid: str,
    sid: str,
    ts: int,
    scholar: str = "scholar_1",
    time_spent: int | None = None,
    status: str | None = None,
    error_type: str | None = None,
) -> dict:
    doc = {
        "attempt_id": aid,
        "scholar_id": scholar,
        "sentence_id": sid,
        "created_at": ts,
        "time_spent": time_spent,
        "status": status,
    }
    if error_type:
        doc["error_type"] = error_type
    return doc


def _skill_state(sid: str, scholar: str = "scholar_1", status: str = "learned") -> dict:
    return {
        "scholar_id": scholar,
        "sentence_id": sid,
        "status": status,
    }


class TestDailyGoal:
    """F6 POST /tracking/daily-goal"""

    def test_no_history_falls_back_to_floor(self, make_client, fake_db):
        """近 7 天无记录 → goal 回落 floor（3/5/10），今日无进度 → percent=0。"""
        client = make_client(tracking_router)
        resp = client.post("/tracking/daily-goal", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["goal"] == {"new_sentences": 3, "minutes": 5, "attempts": 10}
        assert data["progress"] == {"new_sentences": 0, "minutes": 0, "attempts": 0}
        assert data["percent"] == 0
        assert data["completed"] is False

    def test_with_history_grows_goal(self, make_client, fake_db):
        """近 7 天有记录 → goal = clamp(round(avg×1.2), floor, cap)。"""
        # 昨天:3 句新学、20 分钟、10 次
        fake_db.add("study_attempt", _attempt("a1", "s1", _ts(1), time_spent=600))
        fake_db.add("study_attempt", _attempt("a2", "s2", _ts(1), time_spent=600))
        fake_db.add("study_attempt", _attempt("a3", "s3", _ts(1), time_spent=0))
        fake_db.add("skill_state", _skill_state("s1"))
        fake_db.add("skill_state", _skill_state("s2"))
        fake_db.add("skill_state", _skill_state("s3"))
        client = make_client(tracking_router)
        resp = client.post("/tracking/daily-goal", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        # avg7 = {3/7, 20/7, 10/7} → ×1.2 → round，下限 floor
        assert data["goal"]["new_sentences"] >= 3
        assert data["goal"]["attempts"] >= 10
        assert data["goal"]["minutes"] >= 5
        assert data["completed"] is False

    def test_today_progress_and_completed(self, make_client, fake_db):
        """今日进度按口径统计：新学句数/分钟/次数；达标 → completed。"""
        now = int(datetime.now(timezone.utc).timestamp())
        # 10 次尝试（floor.attempts=10 达标），其中 3 句达 learned（floor.new_sentences=3 达标）
        for i in range(10):
            fake_db.add(
                "study_attempt",
                _attempt(f"t{i}", f"s{i % 3}", now, time_spent=600),
            )
        for i in range(3):
            fake_db.add("skill_state", _skill_state(f"s{i}"))
        client = make_client(tracking_router)
        resp = client.post("/tracking/daily-goal", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["progress"]["attempts"] == 10
        assert data["progress"]["minutes"] == 100  # 600s×10 / 60
        assert data["progress"]["new_sentences"] == 3
        # 目标回落 floor:3/5/10 → 三指标全达标 → percent ≥ 100
        assert data["percent"] >= 100
        assert data["completed"] is True

    def test_missing_scholar_id_400(self, make_client, fake_db):
        client = make_client(tracking_router)
        resp = client.post("/tracking/daily-goal", json={})
        assert resp.status_code == 400

    def test_bad_date_400(self, make_client, fake_db):
        client = make_client(tracking_router)
        resp = client.post(
            "/tracking/daily-goal",
            json={"scholar_id": "scholar_1", "date": "2026-13-99"},
        )
        assert resp.status_code == 400


class TestLeaderboard:
    """F8 GET /tracking/leaderboard"""

    def test_metric_minutes_ranking(self, make_client, fake_db):
        """week 窗口按 minutes 排序；昵称来自 scholars，无 openid。"""
        fake_db.add("study_attempt", _attempt("a1", "s1", _ts(0), scholar="scholar_a", time_spent=300))
        fake_db.add("study_attempt", _attempt("a2", "s2", _ts(0), scholar="scholar_a", time_spent=300))
        fake_db.add("study_attempt", _attempt("a3", "s3", _ts(0), scholar="scholar_b", time_spent=120))
        fake_db.add("scholars", {"_id": "scholar_a", "name": "小a"})
        fake_db.add("scholars", {"_id": "scholar_b", "name": "小b"})
        client = make_client(tracking_router)
        resp = client.get("/tracking/leaderboard?period=week&metric=minutes&scholar_id=scholar_a")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["period"] == "week"
        assert data["metric"] == "minutes"
        assert data["my_rank"] == 1
        items = data["items"]
        assert len(items) == 2
        assert items[0]["rank"] == 1
        assert items[0]["scholar_id"] == "scholar_a"
        assert items[0]["name"] == "小a"
        assert items[0]["value"] == 10  # 600s / 60
        assert items[0]["is_me"] is True
        assert items[1]["value"] == 2  # 120s / 60
        assert items[1]["is_me"] is False
        # 隐私：不返回 openid
        assert "openid" not in items[0]

    def test_metric_sentences_dedup(self, make_client, fake_db):
        """sentences 指标：同一学者重复句子去重计数。"""
        fake_db.add("study_attempt", _attempt("a1", "s1", _ts(0), scholar="scholar_a"))
        fake_db.add("study_attempt", _attempt("a2", "s1", _ts(0), scholar="scholar_a"))
        fake_db.add("study_attempt", _attempt("a3", "s2", _ts(0), scholar="scholar_a"))
        client = make_client(tracking_router)
        resp = client.get("/tracking/leaderboard?metric=sentences")
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert items[0]["value"] == 2  # s1 + s2 去重

    def test_period_filter_excludes_old(self, make_client, fake_db):
        """week 窗口不含 8 天前记录。"""
        fake_db.add("study_attempt", _attempt("a1", "s1", _ts(0), scholar="scholar_a", time_spent=60))
        fake_db.add("study_attempt", _attempt("a2", "s2", _ts(8), scholar="scholar_b", time_spent=600))
        client = make_client(tracking_router)
        resp = client.get("/tracking/leaderboard?period=week")
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["scholar_id"] == "scholar_a"

    def test_my_rank_null_when_not_listed(self, make_client, fake_db):
        """请求学者未上榜 → my_rank=null，is_me=false。"""
        fake_db.add("study_attempt", _attempt("a1", "s1", _ts(0), scholar="scholar_a", time_spent=60))
        client = make_client(tracking_router)
        resp = client.get("/tracking/leaderboard?scholar_id=scholar_missing")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["my_rank"] is None
        assert all(not it["is_me"] for it in data["items"])

    def test_empty_returns_empty(self, make_client, fake_db):
        client = make_client(tracking_router)
        resp = client.get("/tracking/leaderboard")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["items"] == []
        assert data["my_rank"] is None

    def test_invalid_params_400(self, make_client, fake_db):
        client = make_client(tracking_router)
        assert client.get("/tracking/leaderboard?period=day").status_code == 400
        assert client.get("/tracking/leaderboard?metric=hours").status_code == 400
        assert client.get("/tracking/leaderboard?limit=0").status_code == 400
        assert client.get("/tracking/leaderboard?limit=51").status_code == 400


class TestBadges:
    """F8 GET /tracking/{scholar_id}/badges"""

    def _seed_badges(self, fake_db):
        fake_db.add(BADGE_COLLECTION, {
            "badge_code": "first_learn",
            "name": "初学乍练",
            "icon": "🎯",
            "description": "学会第一个句子",
            "condition_type": "learned_count",
            "target_value": 1,
            "enabled": True,
        })
        fake_db.add(BADGE_COLLECTION, {
            "badge_code": "minutes_30",
            "name": "半小时达人",
            "icon": "⏰",
            "description": "累计学习 30 分钟",
            "condition_type": "study_minutes",
            "target_value": 30,
            "enabled": True,
        })
        fake_db.add(BADGE_COLLECTION, {
            "badge_code": "disabled_one",
            "name": "禁用徽章",
            "icon": "🚫",
            "description": "enabled=false 不返回",
            "condition_type": "learned_count",
            "target_value": 0,
            "enabled": False,
        })

    def test_earn_and_lock(self, make_client, fake_db):
        """达标 → 幂等发放进 earned；未达标 → locked 含 progress。"""
        self._seed_badges(fake_db)
        fake_db.add("skill_state", _skill_state("s1"))  # learned_count=1 → 达标 first_learn
        fake_db.add("study_attempt", _attempt("a1", "s1", _ts(0), time_spent=60))  # 1 分钟
        client = make_client(tracking_router)
        resp = client.get("/tracking/scholar_1/badges")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        codes = [b["badge_code"] for b in data["earned"]]
        assert "first_learn" in codes
        assert "minutes_30" not in codes  # 未达标
        assert "disabled_one" not in codes  # 禁用不返回
        locked = {b["badge_code"]: b for b in data["locked"]}
        assert "minutes_30" in locked
        assert locked["minutes_30"]["progress"] == {"current": 1, "target": 30}

    def test_idempotent_award(self, make_client, fake_db):
        """重复调用幂等：scholar_badge 只发一条，earned 不重复。"""
        self._seed_badges(fake_db)
        fake_db.add("skill_state", _skill_state("s1"))
        client = make_client(tracking_router)
        for _ in range(3):
            resp = client.get("/tracking/scholar_1/badges")
            assert resp.status_code == 200
        rows = fake_db.all(SCHOLAR_BADGE_COLLECTION)
        first = [r for r in rows if r["badge_code"] == "first_learn"]
        assert len(first) == 1
        assert first[0]["scholar_id"] == "scholar_1"
        assert first[0]["first_awarded_at"] == first[0]["awarded_at"]
        assert first[0]["badge_code"] == "first_learn"

    def test_no_badge_definitions(self, make_client, fake_db):
        """无徽章定义 → success:true, earned:[], locked:[]。"""
        client = make_client(tracking_router)
        resp = client.get("/tracking/scholar_1/badges")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data == {"earned": [], "locked": []}

    def test_missing_scholar_400(self, make_client, fake_db):
        client = make_client(tracking_router)
        resp = client.get("/tracking//badges")
        assert resp.status_code in (400, 404)  # 空路径由框架处理，此处不强约束


class TestWrongBook:
    """F11 GET /tracking/{scholar_id}/wrong-book"""

    def test_aggregation_and_ordering(self, make_client, fake_db):
        """incorrect 事件按句子聚合：error_count、error_types 分布、按 last_error_at 降序。"""
        fake_db.add("study_attempt", _attempt("w1", "s1", _ts(1), time_spent=30, status="incorrect", error_type="grammar"))
        fake_db.add("study_attempt", _attempt("w2", "s1", _ts(0), time_spent=30, status="incorrect", error_type="vocabulary"))
        fake_db.add("study_attempt", _attempt("w3", "s2", _ts(2), time_spent=30, status="incorrect", error_type="grammar"))
        fake_db.add("study_attempt", _attempt("w4", "s3", _ts(0), time_spent=30, status="correct"))  # 非错题不计
        fake_db.add("sentence_v2", {
            "sentence_id": "s1",
            "text": "I like apples.",
            "translation": "我喜欢苹果。",
            "lesson_id": "l1",
            "chapter_id": "c1",
        })
        fake_db.add("sentence_v2", {
            "sentence_id": "s2",
            "text": "He runs fast.",
            "translation": "他跑得很快。",
            "lesson_id": "l1",
            "chapter_id": "c1",
        })
        client = make_client(tracking_router)
        resp = client.get("/tracking/scholar_1/wrong-book")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["total"] == 2  # s1 + s2
        items = data["items"]
        # 按 last_error_at 降序：s1(今天) > s2(2天前)
        assert items[0]["sentence_id"] == "s1"
        assert items[1]["sentence_id"] == "s2"
        s1 = items[0]
        assert s1["content"] == "I like apples."
        assert s1["translation"] == "我喜欢苹果。"
        assert s1["error_count"] == 2
        assert s1["chapter_id"] == "c1"
        types = {t["type"]: t["count"] for t in s1["error_types"]}
        assert types == {"grammar": 1, "vocabulary": 1}

    def test_error_type_filter(self, make_client, fake_db):
        """error_type 入参：仅统计该类型错误；error_types 仍为全量分布。"""
        fake_db.add("study_attempt", _attempt("w1", "s1", _ts(0), time_spent=30, status="incorrect", error_type="grammar"))
        fake_db.add("study_attempt", _attempt("w2", "s1", _ts(0), time_spent=30, status="incorrect", error_type="vocabulary"))
        fake_db.add("sentence_v2", {"sentence_id": "s1", "text": "T", "translation": "译"})
        client = make_client(tracking_router)
        resp = client.get("/tracking/scholar_1/wrong-book?error_type=grammar")
        assert resp.status_code == 200
        item = resp.json()["data"]["items"][0]
        assert item["error_count"] == 1  # 仅 grammar
        types = {t["type"]: t["count"] for t in item["error_types"]}
        assert types == {"grammar": 1, "vocabulary": 1}  # 全量

    def test_legacy_missing_error_type_falls_back_to_other(self, make_client, fake_db):
        """存量数据无 error_type → 回落 other 口径。"""
        doc = _attempt("w1", "s1", _ts(0), time_spent=30, status="incorrect")
        fake_db.add("study_attempt", doc)
        fake_db.add("sentence_v2", {"sentence_id": "s1", "text": "T", "translation": "译"})
        client = make_client(tracking_router)
        resp = client.get("/tracking/scholar_1/wrong-book")
        assert resp.status_code == 200
        item = resp.json()["data"]["items"][0]
        assert item["error_count"] == 1
        assert item["error_types"] == [{"type": "other", "count": 1}]

    def test_empty(self, make_client, fake_db):
        """无错题 → success:true, total:0, items:[]。"""
        client = make_client(tracking_router)
        resp = client.get("/tracking/scholar_1/wrong-book")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data == {"total": 0, "items": []}

    def test_invalid_limit_400(self, make_client, fake_db):
        client = make_client(tracking_router)
        assert client.get("/tracking/scholar_1/wrong-book?limit=0").status_code == 400
        assert client.get("/tracking/scholar_1/wrong-book?limit=201").status_code == 400


class TestDialogueTaskScenario:
    """F10 POST /match/dialogue/task — scenario/sessionId 透传"""

    def test_scenario_and_session_id_persisted(self, make_client, monkeypatch, fake_db):
        called = {}

        async def fake_run(task_id, scholar_id, sentence, scenario=None, session_id=None):
            called.update({
                "task_id": task_id,
                "scholar_id": scholar_id,
                "sentence": sentence,
                "scenario": scenario,
                "session_id": session_id,
            })

        monkeypatch.setattr("services.routes_dialogue.run_dialogue_task", fake_run)
        client = make_client(dialogue_router)
        resp = client.post(
            "/match/dialogue/task",
            json={
                "scholarId": "s1",
                "sentence": "Hello",
                "scenario": "travel",
                "sessionId": "ses_abc",
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert called["scenario"] == "travel"
        assert called["session_id"] == "ses_abc"
        stored = fake_db.all("dialogue_task")[0]
        assert stored["scenario"] == "travel"
        assert stored["session_id"] == "ses_abc"

    def test_optional_fields_absent(self, make_client, monkeypatch, fake_db):
        """未传 scenario/sessionId → 任务文档不含这些字段（向后兼容）。"""
        called = {}

        async def fake_run(task_id, scholar_id, sentence, scenario=None, session_id=None):
            called.update({"scenario": scenario, "session_id": session_id})

        monkeypatch.setattr("services.routes_dialogue.run_dialogue_task", fake_run)
        client = make_client(dialogue_router)
        resp = client.post(
            "/match/dialogue/task",
            json={"scholarId": "s1", "sentence": "Hello"},
        )
        assert resp.status_code == 200
        assert called["scenario"] is None
        assert called["session_id"] is None
        stored = fake_db.all("dialogue_task")[0]
        assert "scenario" not in stored
        assert "session_id" not in stored
