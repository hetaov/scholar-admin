"""集成测试:学习状态上报接口 + 查询改走 skill_state（Phase 2）+ 事件写入（Phase 3）
+ M3 G1.2 写兼容层 L1（Lazy dedup）

被测链路:FastAPI TestClient + FakeDB,覆盖:
- POST /tracking/state        上报单句单能力状态(创建 / 累加 / 参数校验)
- POST /tracking/state        同时写一条 study_attempt 事件(append-only)
- GET  /tracking/{scholar_id} 只查 skill_state, 无记录打日志不回退旧表
- POST /tracking/state        惰性补齐 semantic_key / canonical_sentence_id（M3 G1.2）
"""

from __future__ import annotations

from services.models_content import compute_text_hash
from services.routes_state import router as state_router
from services.routes_tracking import router as tracking_router


class TestPostTrackingState:
    def test_create_new(self, make_client, fake_db):
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "已学",
                "score": 85,
                "time_spent": 120,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        state = data["state"]
        assert state["attempt_count"] == 1
        assert state["status"] == "learned"
        assert state["mastery_score"] == 85.0
        assert state["skill_code"] == "translation"
        # Phase 3: 同时写入一条 study_attempt 事件
        attempt = data["attempt"]
        assert attempt["scholar_id"] == "s1"
        assert attempt["sentence_id"] == "sent_1"
        assert attempt["skill_code"] == "translation"
        assert attempt["time_spent"] == 120
        assert attempt["attempt_id"]
        assert fake_db.all("study_attempt").__len__() == 1

    def test_repeat_accumulates_state_but_appends_events(self, make_client, fake_db):
        client = make_client(state_router, tracking_router)
        for _ in range(2):
            resp = client.post(
                "/tracking/state",
                json={"scholar_id": "s1", "sentence_id": "sent_1"},
            )
            assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["state"]["attempt_count"] == 2
        # skill_state 只一条(同复合键 upsert), study_attempt 每报一条追加一条
        assert fake_db.all("skill_state").__len__() == 1
        assert fake_db.all("study_attempt").__len__() == 2

    def test_missing_scholar_id(self, make_client, fake_db):
        client = make_client(state_router, tracking_router)
        resp = client.post("/tracking/state", json={"sentence_id": "sent_1"})
        assert resp.status_code == 400
        assert "scholar_id" in resp.json()["detail"]

    def test_missing_sentence_id(self, make_client, fake_db):
        client = make_client(state_router, tracking_router)
        resp = client.post("/tracking/state", json={"scholar_id": "s1"})
        assert resp.status_code == 400
        assert "sentence_id" in resp.json()["detail"]

    def test_default_skill_code(self, make_client, fake_db):
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state", json={"scholar_id": "s1", "sentence_id": "sent_1"}
        )
        assert resp.json()["data"]["state"]["skill_code"] == "translation"

    def test_attempt_type_inferred_from_skill_code(self, make_client, fake_db):
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={"scholar_id": "s1", "sentence_id": "sent_1", "skill_code": "listening"},
        )
        attempt = resp.json()["data"]["attempt"]
        assert attempt["attempt_type"] == "listen"

    def test_attempt_explicit_fields(self, make_client, fake_db):
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "attempt_type": "quiz",
                "attempt_status": "correct",
                "score": 100,
                "lesson_id": "unit_1",
                "session_id": "ses_abc",
            },
        )
        attempt = resp.json()["data"]["attempt"]
        assert attempt["attempt_type"] == "quiz"
        assert attempt["status"] == "correct"
        assert attempt["lesson_id"] == "unit_1"
        assert attempt["session_id"] == "ses_abc"

    def test_invalid_attempt_status_falls_back(self, make_client, fake_db):
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={"scholar_id": "s1", "sentence_id": "sent_1", "attempt_status": "???"},
        )
        assert resp.json()["data"]["attempt"]["status"] == "completed"

    def test_attempt_source_passthrough(self, make_client, fake_db):
        # Q5 审计：消灭来源 eval/self 写入 study_attempt.source
        client = make_client(state_router, tracking_router)
        for src in ("eval", "self"):
            resp = client.post(
                "/tracking/state",
                json={
                    "scholar_id": "s1",
                    "sentence_id": f"sent_{src}",
                    "skill_code": "translation",
                    "status": "mastered",
                    "source": src,
                },
            )
            assert resp.status_code == 200
            assert resp.json()["data"]["attempt"]["source"] == src

    def test_attempt_source_absent_omits_field(self, make_client, fake_db):
        # 缺省不带 source → 文档不写该字段（向后兼容，存量口径零差）
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={"scholar_id": "s1", "sentence_id": "sent_1", "status": "mastered"},
        )
        assert resp.status_code == 200
        assert "source" not in resp.json()["data"]["attempt"]


class TestM3WriteCompatLayer:
    """M3 G1.2：POST /tracking/state 写兼容层 L1 —— 惰性补齐语义键（不切 skill_state 写入键）。"""

    def test_lazy_backfills_semantic_key(self, make_client, fake_db):
        fake_db.add("sentence_v2", {
            "sentence_id": "sent_1", "text": "Hello!", "created_at": 1000,
        })
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state", json={"scholar_id": "s1", "sentence_id": "sent_1"}
        )
        assert resp.status_code == 200
        stored = fake_db.all("sentence_v2")[0]
        assert stored["semantic_key"] == compute_text_hash("Hello!")
        assert stored["canonical_sentence_id"] == "sent_1"  # 自指 = canonical
        # skill_state 写入键仍为原 sentence_id（M3 不切键）
        states = fake_db.all("skill_state")
        assert len(states) == 1
        assert states[0]["sentence_id"] == "sent_1"

    def test_duplicate_points_to_canonical(self, make_client, fake_db):
        fake_db.add("sentence_v2", {
            "sentence_id": "sent_a", "text": "Good morning!", "created_at": 1000,
        })
        fake_db.add("sentence_v2", {
            "sentence_id": "sent_b", "text": "good morning", "created_at": 2000,
        })
        client = make_client(state_router, tracking_router)
        # 上报晚句 → canonical 指向最早者 sent_a
        resp = client.post(
            "/tracking/state", json={"scholar_id": "s1", "sentence_id": "sent_b"}
        )
        assert resp.status_code == 200
        by_id = {r["sentence_id"]: r for r in fake_db.all("sentence_v2")}
        assert by_id["sent_b"]["canonical_sentence_id"] == "sent_a"
        assert by_id["sent_b"]["semantic_key"] == compute_text_hash("good morning")
        # sent_a 未被上报 → 未被触碰（惰性）
        assert by_id["sent_a"].get("semantic_key") is None

    def test_sentence_not_in_content_skips_backfill(self, make_client, fake_db):
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state", json={"scholar_id": "s1", "sentence_id": "no_such_sent"}
        )
        # 契约 §3.2：句子不在内容库不阻断上报（错误仅 400/500），跳过补齐继续写状态
        assert resp.status_code == 200
        states = fake_db.all("skill_state")
        assert states and states[0]["sentence_id"] == "no_such_sent"
        assert fake_db.all("sentence_v2") == []  # 不产生任何 sentence_v2 写入

    def test_repeat_report_idempotent(self, make_client, fake_db):
        fake_db.add("sentence_v2", {
            "sentence_id": "sent_1", "text": "Hello", "created_at": 1,
        })
        client = make_client(state_router, tracking_router)
        for _ in range(2):
            resp = client.post(
                "/tracking/state", json={"scholar_id": "s1", "sentence_id": "sent_1"}
            )
            assert resp.status_code == 200
        stored = fake_db.all("sentence_v2")[0]
        assert stored["semantic_key"] == compute_text_hash("Hello")  # 只补一次，幂等


class TestGetTrackingByScholarV2:
    def test_prefers_skill_state(self, make_client, fake_db):
        client = make_client(state_router, tracking_router)
        client.post(
            "/tracking/state",
            json={"scholar_id": "s1", "sentence_id": "sent_1", "status": "learned"},
        )
        resp = client.get("/tracking/s1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["records"][0]["sentence_id"] == "sent_1"
        assert body["records"][0]["attempt_count"] == 1

    def test_no_skill_state_returns_empty(self, make_client, fake_db):
        """skill_state 无记录时返回空(旧表已下线, 无回退)。"""
        client = make_client(state_router, tracking_router)
        resp = client.get("/tracking/legacy_user")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["records"] == []


class TestMasteredScoreFloor:
    """短板消灭战 D2：status=mastered 且无 score → mastery_score 抬升至 MASTERED_SCORE_FLOOR(80)。

    覆盖 I1（消灭后 score≥60）/ I2（mastered ⇒ score≥80）/ I3（非消灭路径行为不变）。
    """

    @staticmethod
    def _seed_state(
        fake_db,
        *,
        scholar_id="s1",
        sentence_id="sent_1",
        skill_code="translation",
        score=40.0,
        status="learning",
        attempt_count=1,
    ):
        from services.models_learning import skill_state_id

        key = skill_state_id(scholar_id, sentence_id, skill_code)
        fake_db.add(
            "skill_state",
            {
                "_id": key,
                "state_id": key,
                "scholar_id": scholar_id,
                "sentence_id": sentence_id,
                "skill_code": skill_code,
                "status": status,
                "mastery_score": score,
                "progress": (score or 0) / 100.0,
                "attempt_count": attempt_count,
                "last_studied_at": 1000,
                "next_review_at": 2000,
                "created_at": 1000,
                "updated_at": 1000,
            },
        )

    def test_mastered_without_score_raises_to_floor(self, make_client, fake_db):
        """旧分 40 + status=mastered 无分 → 80（消灭真实抬升）。"""
        self._seed_state(fake_db, score=40.0, status="learning")
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "mastered",
            },
        )
        assert resp.status_code == 200
        state = resp.json()["data"]["state"]
        assert state["status"] == "mastered"
        assert state["mastery_score"] == 80.0
        assert state["progress"] == 0.8

    def test_mastered_with_real_score_keeps_truth(self, make_client, fake_db):
        """携带真实评估分 100 → 以真值为准，不被 floor 覆盖。"""
        self._seed_state(fake_db, score=40.0)
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "mastered",
                "score": 100,
            },
        )
        state = resp.json()["data"]["state"]
        assert state["mastery_score"] == 100.0
        assert state["status"] == "mastered"

    def test_mastered_does_not_lower_existing_high_score(self, make_client, fake_db):
        """旧分 90 + mastered 无分 → max(90, 80) = 90（不回退）。"""
        self._seed_state(fake_db, score=90.0, status="mastered")
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "mastered",
            },
        )
        state = resp.json()["data"]["state"]
        assert state["mastery_score"] == 90.0

    def test_non_mastered_without_score_keeps_old(self, make_client, fake_db):
        """I3 回归：status=learning 无分 → mastery_score 保持旧值 40（不抬升）。"""
        self._seed_state(fake_db, score=40.0, status="learning")
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "learning",
            },
        )
        state = resp.json()["data"]["state"]
        assert state["mastery_score"] == 40.0

    def test_mastered_first_write_without_score_uses_floor(self, make_client, fake_db):
        """P3 L1 冷启动：首次写入即 mastered 无分 → floor=60（仅退出 weakness 线）。"""
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "mastered",
            },
        )
        state = resp.json()["data"]["state"]
        assert state["status"] == "mastered"
        assert state["mastery_score"] == 60.0
        assert state["progress"] == 0.6

    def test_mastered_with_existing_attempt_uses_l2_floor(self, make_client, fake_db):
        """P3 L2：已有 attempt_count>=1 且无达标评估 → floor=80。"""
        self._seed_state(fake_db, score=40.0, status="learning", attempt_count=2)
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "mastered",
            },
        )
        state = resp.json()["data"]["state"]
        assert state["mastery_score"] == 80.0

    def test_mastered_with_passing_eval_uses_l3_floor(self, make_client, fake_db):
        """P3 L3：历史有达标评估分 90 → floor=max(90, 80)=90。"""
        self._seed_state(fake_db, score=40.0, status="learning", attempt_count=2)
        fake_db.add(
            "study_attempt",
            {
                "_id": "att_1",
                "attempt_id": "att_1",
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "source": "eval",
                "score": 90.0,
                "status": "correct",
                "created_at": 1000,
            },
        )
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "mastered",
            },
        )
        state = resp.json()["data"]["state"]
        assert state["mastery_score"] == 90.0

    def test_mastered_with_passing_eval_below_80_floor_80(self, make_client, fake_db):
        """P3 L3：历史评估分 70（<80）→ floor=max(70, 80)=80，不回退到评估分。"""
        self._seed_state(fake_db, score=40.0, status="learning", attempt_count=2)
        fake_db.add(
            "study_attempt",
            {
                "_id": "att_1",
                "attempt_id": "att_1",
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "source": "eval",
                "score": 70.0,
                "status": "correct",
                "created_at": 1000,
            },
        )
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "mastered",
            },
        )
        state = resp.json()["data"]["state"]
        assert state["mastery_score"] == 80.0


class TestSelfEvalCooldown:
    """P3.2 自评冷却：source='self' 距上次学习 < 3s 时拒绝写侧。"""

    @staticmethod
    def _seed_state(fake_db, *, last_studied_at):
        from services.models_learning import skill_state_id

        key = skill_state_id("s1", "sent_1", "translation")
        fake_db.add(
            "skill_state",
            {
                "_id": key,
                "state_id": key,
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "learning",
                "mastery_score": 40.0,
                "progress": 0.4,
                "attempt_count": 2,
                "last_studied_at": last_studied_at,
                "next_review_at": 2000,
                "created_at": 1000,
                "updated_at": 1000,
            },
        )

    def test_self_eval_within_cooldown_rejected(self, make_client, fake_db, monkeypatch):
        """source='self' 且 last_studied_at 距今 1s → 冷却拒绝，skill_state 不变。"""
        fixed_now = 1_700_000_000_000  # ms
        self._seed_state(fake_db, last_studied_at=fixed_now - 1000)  # 1s ago
        monkeypatch.setattr("services.routes.state.time.time", lambda: fixed_now / 1000.0)
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "mastered",
                "source": "self",
            },
        )
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "SELF_EVAL_COOLDOWN"
        # skill_state 未被修改
        state = fake_db.all("skill_state")[0]
        assert state["status"] == "learning"
        assert state["mastery_score"] == 40.0

    def test_self_eval_outside_cooldown_passes(self, make_client, fake_db, monkeypatch):
        """source='self' 且 last_studied_at 距今 10s → 冷却已过，正常写侧。"""
        fixed_now = 1_700_000_000_000
        self._seed_state(fake_db, last_studied_at=fixed_now - 10_000)  # 10s ago
        monkeypatch.setattr("services.routes.state.time.time", lambda: fixed_now / 1000.0)
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "mastered",
                "source": "self",
            },
        )
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["state"]["status"] == "mastered"

    def test_eval_path_not_affected_by_cooldown(self, make_client, fake_db, monkeypatch):
        """source='eval' 不受冷却限制，即使在冷却窗口内也正常写侧。"""
        fixed_now = 1_700_000_000_000
        self._seed_state(fake_db, last_studied_at=fixed_now - 1000)  # 1s ago
        monkeypatch.setattr("services.routes.state.time.time", lambda: fixed_now / 1000.0)
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "mastered",
                "score": 85,
                "source": "eval",
            },
        )
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["state"]["mastery_score"] == 85.0
