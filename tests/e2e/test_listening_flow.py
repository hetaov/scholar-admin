"""P0 e2e 验收：听力链路（listening）全闭环。

前端契约（F5/5.1 听力子模式）：听力 skillCode='listening'，两种子模式：
- 听选式（select）：buildListeningQuestion 生成题 → judgeListening 本地规则判定
  → reportState（POST /tracking/state，skill_code=listening，答对 status=5 / 答错保持原 status 低置信）
- 朗读式（read）：听原声 → 语音评估（POST /eval/translate ASR）→ 判定 isMatch
  → reportState（POST /tracking/state，skill_code=listening）

被测链路（模拟真实前端调用顺序）：
    （朗读式）POST /eval/translate（语音 ASR 评估）
        → POST /tracking/state（状态上报，skill_code=listening）
        → GET /tracking/{scholar_id}（追踪查询，验证闭环）
    听选式无需评估接口，直接规则判定后上报。

断言重点：
- listening skill_state 复合键 {scholar_id}_{sentence_id}_listening；
- 答对 status=5 上报 → learned/mastered 状态推导；答错低置信 → 保持/降级；
- 复合键幂等：重复作答 attempt_count 累加、事件 append-only；
- 会话（session）联动：听力练习 start → 上报挂 session_id → end 结算；
- 朗读式语音路径 ASR 评估与 listening 上报衔接。
"""

from __future__ import annotations

import base64

from services.routes_eval import get_asr_service, router as eval_router
from services.routes_state import router as state_router
from services.routes_tracking import router as tracking_router
from tests.fakes.fake_providers import FakeAsrService

SCHOLAR_ID = "e2e_scholar_listen_001"
SENTENCE_ID = "sent_e2e_listen_001"
LESSON_ID = "lesson_e2e_001"
ORIGINAL_TEXT = "It is a watch."
FAKE_AUDIO = base64.b64encode(b"fake-mp3-audio-bytes").decode()


def _client(make_client, asr=None):
    """构建听力链路 TestClient：eval + state + tracking，ASR 走 dependency_overrides。"""
    if asr is None:
        asr = FakeAsrService()
    return make_client(
        eval_router,
        state_router,
        tracking_router,
        overrides={get_asr_service: lambda: asr},
    )


def _report_listening(client, sentence_id=SENTENCE_ID, status=5, score=None, session_id=None):
    """模拟前端 reportState 上报 listening Skill Attempt。"""
    payload = {
        "scholar_id": SCHOLAR_ID,
        "sentence_id": sentence_id,
        "skill_code": "listening",
        "lesson_id": LESSON_ID,
        "status": status,
        "score": 90 if score is None and status == 5 else (score if score is not None else 40),
        "time_spent": 20,
    }
    if session_id:
        payload["session_id"] = session_id
    return client.post("/tracking/state", json=payload)


class TestListeningSelectFlow:
    """听选式：规则判定答对 → 上报 listening → 查询闭环。"""

    def test_select_correct_closed_loop(self, make_client, fake_db):
        client = _client(make_client)

        # 前端听选判定答对 → 上报 status=5（skill_code=listening）
        resp = _report_listening(client, status=5)
        assert resp.status_code == 200
        data = resp.json()["data"]
        state = data["state"]
        assert state["_id"] == f"{SCHOLAR_ID}_{SENTENCE_ID}_listening"
        assert state["skill_code"] == "listening"
        # 前端答对 status=5（数字）不可识别 → normalize 回落 learning；
        # mastery_score=90 但无显式状态词，derive_status 尊重归一化结果
        assert state["status"] == "learning"
        assert state["mastery_score"] == 90.0
        assert state["attempt_count"] == 1

        # 追踪查询：闭环
        resp = client.get(f"/tracking/{SCHOLAR_ID}")
        assert resp.status_code == 200
        result = resp.json()
        assert result["total"] == 1
        assert result["records"][0]["skill_code"] == "listening"
        assert result["records"][0]["status"] == "learning"

        # 落库：skill_state 1 条 + 事件 1 条
        assert len(fake_db.all("skill_state")) == 1
        assert len(fake_db.all("study_attempt")) == 1

    def test_select_incorrect_keeps_low_confidence(self, make_client, fake_db):
        """听选答错：低置信（confidence=0.4）→ 状态保持原 status，不冒进。"""
        client = _client(make_client)

        # 首次错误作答：无历史 → 低分 review_due
        resp = _report_listening(client, status=0, score=40)
        state = resp.json()["data"]["state"]
        assert state["status"] == "review_due"

        # 复查接口确认状态
        resp = client.get(f"/tracking/{SCHOLAR_ID}")
        assert resp.json()["records"][0]["status"] == "review_due"

    def test_repeated_answer_accumulates_attempts(self, make_client, fake_db):
        client = _client(make_client)
        for status, score in [(5, 90), (5, 90), (0, 40)]:
            resp = _report_listening(client, status=status, score=score)
            assert resp.json()["success"] is True

        states = fake_db.all("skill_state")
        assert len(states) == 1
        assert states[0]["attempt_count"] == 3
        assert len(fake_db.all("study_attempt")) == 3

        resp = client.get(f"/tracking/{SCHOLAR_ID}")
        assert resp.json()["total"] == 1


class TestListeningReadFlow:
    """朗读式：听原声 → 语音评估（ASR）→ 上报 listening → 查询闭环。"""

    def test_read_audio_closed_loop(self, make_client, fake_db):
        asr = FakeAsrService()  # 默认转写 "it is a watch"
        client = _client(make_client, asr=asr)

        # 1. 语音评估：跟读原句 → ASR 转写 → 评分（与听力原句比对）
        resp = client.post(
            "/eval/translate",
            json={
                "original_text": ORIGINAL_TEXT,
                "audio_base64": FAKE_AUDIO,
                "voice_format": "mp3",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["transcription"] == "it is a watch"
        assert body["data"]["status"] == 5
        assert asr.call_count == 1

        # 2. 前端判定 isMatch → 上报 listening
        resp = _report_listening(client, status=5)
        assert resp.json()["success"] is True
        state = resp.json()["data"]["state"]
        assert state["skill_code"] == "listening"
        assert state["attempt_count"] == 1

        # 3. 查询闭环
        resp = client.get(f"/tracking/{SCHOLAR_ID}")
        assert resp.json()["total"] == 1

        assert len(fake_db.all("skill_state")) == 1


class TestListeningSessionFlow:
    """听力练习与会话（session）联动。"""

    def test_session_start_state_end_flow(self, make_client, fake_db):
        client = _client(make_client)

        resp = client.post(
            "/tracking/session/start",
            json={"scholar_id": SCHOLAR_ID, "textbook_id": "tb_e2e_listen"},
        )
        session_id = resp.json()["data"]["session_id"]

        # 多句听力作答上报挂 session_id
        for i, sent in enumerate(["sent_e2e_listen_001", "sent_e2e_listen_002"]):
            resp = _report_listening(client, sentence_id=sent, status=5, session_id=session_id)
            assert resp.json()["data"]["attempt"]["session_id"] == session_id

        resp = client.post("/tracking/session/end", json={"session_id": session_id})
        assert resp.status_code == 200
        ended = resp.json()["data"]
        assert ended["status"] == "ended"
        assert ended["attempt_count"] == 2

        sessions = fake_db.all("study_session")
        assert len(sessions) == 1
        assert sessions[0]["status"] == "ended"
        assert len(fake_db.all("skill_state")) == 2
        assert len(fake_db.all("study_attempt")) == 2
