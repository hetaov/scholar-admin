"""e2e 验收：翻译评估 v2 异步全链路闭环（ADR-0022 / docs_v1 §11 步骤 9）

被测链路（模拟真实前端调用顺序）：
    POST /eval/translate/v2（提交，毫秒级 task_id）
        → 后台执行器 run_translation_task（ASR → LLM 评分 → evaluation 双写）
        → GET /eval/translate/v2/task/{task_id}（轮询查询）

断言重点：
- 提交返回 pending，不做任何 LLM/ASR 调用（毫秒级）；
- 执行后查询 → success + result{ transcription, status(0-5), feedback, confidence }；
- 终态双写：evaluation 集合落一条 type=translation 记录；
- 失败链路：LLM 不可用 → failed + 失败全量留痕（succeeded=false + error_code）。

说明：后台任务由提交接口 asyncio.create_task 调度（TestClient 内事件循环），
为避免异步时序抖动，测试将提交接口的调度替换为记录型 stub，随后同步驱动
真实执行器 run_translation_task，最后经 GET 查询闭环（同 integration 策略）。
"""
from __future__ import annotations

import asyncio

from services.routes_eval import get_asr_service, router as eval_router
from services.translation_task import run_translation_task
from tests.fakes.fake_providers import FakeAsrService

MODEL_OK = '{"status": 4, "feedback": "用词准确，注意时态", "confidence": 0.85}'


def _run(coro):
    return asyncio.run(coro)


def _client(make_client, monkeypatch, model_output=None):
    """构建 v2 TestClient：提交接口替换为记录型 stub + 模型输出可注入。"""
    submitted = {}

    async def fake_run(task_id, **kwargs):
        submitted["task_id"] = task_id
        submitted.update(kwargs)

    monkeypatch.setattr("services.routes_eval.run_translation_task", fake_run)
    if model_output is not None:
        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm",
            lambda *a, **k: model_output,
        )
    client = make_client(
        eval_router, overrides={get_asr_service: lambda: FakeAsrService()}
    )
    return client, submitted


class TestTranslationV2TextFullFlow:
    """文字路径：提交 → 后台执行 → 轮询查询 → 成功终态 + 证据落库。"""

    def test_text_translation_v2_closed_loop(self, make_client, monkeypatch, fake_db):
        client, submitted = _client(make_client, monkeypatch, model_output=MODEL_OK)

        # 1. 提交（毫秒级，不做 LLM/ASR 调用）
        resp = client.post(
            "/eval/translate/v2",
            json={
                "original_text": "It is a watch.",
                "user_input": "它是一块手表。",
                "scholar_id": "e2e_v2_scholar",
                "sentence_id": "e2e_v2_sent",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        task_id = body["data"]["task_id"]
        assert task_id.startswith("tr_")
        assert body["data"]["status"] == "pending"

        # 2. 后台执行器（同步驱动真实 worker，参数来自提交透传）
        _run(
            run_translation_task(
                submitted["task_id"],
                original_text=submitted["original_text"],
                mode=submitted["mode"],
                input_mode=submitted["input_mode"],
                user_input=submitted["user_input"],
                scholar_id=submitted["scholar_id"],
                sentence_id=submitted["sentence_id"],
            )
        )

        # 3. 轮询查询 → success + result
        resp = client.get(f"/eval/translate/v2/task/{task_id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "success"
        assert data["error"] is None
        result = data["result"]
        assert result["transcription"] == "它是一块手表。"
        assert result["status"] == 4
        assert result["feedback"] == "用词准确，注意时态"
        assert result["confidence"] == 0.85

        # 4. 终态双写：evaluation 一条 succeeded=true
        evals = fake_db.all("evaluation")
        assert len(evals) == 1
        assert evals[0]["type"] == "translation"
        assert evals[0]["succeeded"] is True
        assert evals[0]["status"] == 4
        assert evals[0]["task_id"] == task_id

    def test_llm_failure_leaves_failed_trace(self, make_client, monkeypatch, fake_db):
        """失败链路（LLM 不可用）：failed + 失败全量留痕（succeeded=false）。"""
        # 不注入模型输出：no_external_calls 屏蔽 LLM → EVAL_UNAVAILABLE
        client, submitted = _client(make_client, monkeypatch)

        resp = client.post(
            "/eval/translate/v2",
            json={
                "original_text": "It is a watch.",
                "user_input": "它是一块手表。",
                "scholar_id": "e2e_v2_scholar",
            },
        )
        assert resp.status_code == 200
        task_id = resp.json()["data"]["task_id"]

        _run(
            run_translation_task(
                submitted["task_id"],
                original_text=submitted["original_text"],
                mode=submitted["mode"],
                input_mode=submitted["input_mode"],
                user_input=submitted["user_input"],
                scholar_id=submitted["scholar_id"],
            )
        )

        resp = client.get(f"/eval/translate/v2/task/{task_id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "failed"
        assert data["result"] is None
        assert data["error"]  # 可读失败原因（error_detail 字符串）

        # 失败留痕：evaluation succeeded=false + error_code + llm_timeout_seconds
        evals = fake_db.all("evaluation")
        assert len(evals) == 1
        assert evals[0]["succeeded"] is False
        assert evals[0]["error_code"] == "EVAL_UNAVAILABLE"
        assert evals[0]["failure_stage"] == "llm"
        assert evals[0]["llm_timeout_seconds"] == 300


class TestTranslationV2VoiceFlow:
    """语音路径：ASR 转写 → 评分 → 成功闭环。"""

    def test_voice_translation_v2_closed_loop(
        self, make_client, monkeypatch, fake_db
    ):
        import base64

        monkeypatch.setattr(
            "services.translation_task.get_asr_service",
            lambda: FakeAsrService(),
            raising=False,
        )
        client, submitted = _client(make_client, monkeypatch, model_output=MODEL_OK)
        fake_audio = base64.b64encode(b"fake-mp3-audio-bytes").decode()

        resp = client.post(
            "/eval/translate/v2",
            json={
                "original_text": "It is a watch.",
                "audio_base64": fake_audio,
                "voice_format": "mp3",
                "scholar_id": "e2e_v2_voice",
            },
        )
        assert resp.status_code == 200
        task_id = resp.json()["data"]["task_id"]
        assert resp.json()["data"]["status"] == "pending"

        _run(
            run_translation_task(
                submitted["task_id"],
                original_text=submitted["original_text"],
                mode=submitted["mode"],
                input_mode=submitted["input_mode"],
                audio_base64=submitted["audio_base64"],
                voice_format=submitted["voice_format"],
                scholar_id=submitted["scholar_id"],
            )
        )

        resp = client.get(f"/eval/translate/v2/task/{task_id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "success"
        result = data["result"]
        assert result["transcription"] == "it is a watch"  # FakeAsrService 默认转写
        assert result["status"] == 4

        evals = fake_db.all("evaluation")
        assert evals[0]["succeeded"] is True
        assert evals[0]["input_mode"] == "voice"
