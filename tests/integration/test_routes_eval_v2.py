"""集成测试：翻译评估 v2 异步接口（ADR-0022 / api-contract §3.4 / docs_v1 §11 步骤 3-5）

被测链路（FastAPI TestClient + FakeDB，不触真实火山）：
- POST /eval/translate/v2           提交任务 → 毫秒级返回 { task_id, status: pending }
- GET  /eval/translate/v2/task/{id} 查询任务状态（pending/success/failed + TTL/卡死自愈）
- run_translation_task              后台执行器（语音 ASR → LLM 评分 → evaluation 双写）

要点：
- 提交接口用 stub 替换后台执行器，避免异步时序影响断言（同 test_dialogue_task_routes）
- 执行器全链路用同步 _run() 直接驱动（patch LLM/ASR 替身）
- no_external_calls 默认屏蔽 _call_translation_llm（返回 None → EVAL_UNAVAILABLE）
"""
from __future__ import annotations

import asyncio
import base64
import time

from services.routes_eval import get_asr_service, router as eval_router
from services.translation_task import run_translation_task
from tests.fakes.fake_providers import FakeAsrService
from tests.fakes.seed_factory import seed_translation_task

TASK_TTL_MS = 24 * 60 * 60 * 1000
FAKE_AUDIO = base64.b64encode(b"fake-mp3-bytes").decode()
MODEL_OK = '{"status": 4, "feedback": "用词准确，注意时态", "confidence": 0.85}'


def _run(coro):
    return asyncio.run(coro)


def _client(make_client, asr=None, model_output=None, monkeypatch=None):
    """构建 v2 TestClient：ASR 走 dependency_overrides；模型输出可注入。"""
    if asr is None:
        asr = FakeAsrService()
    if model_output is not None:
        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm",
            lambda *a, **k: model_output,
        )
    return make_client(eval_router, overrides={get_asr_service: lambda: asr})


class TestEvalTranslateV2Submit:
    """POST /eval/translate/v2"""

    def test_ok_returns_task_id_pending(self, make_client, monkeypatch, fake_db):
        called = {}

        async def fake_run(task_id, **kwargs):
            called["task_id"] = task_id
            called.update(kwargs)

        monkeypatch.setattr(
            "services.routes_eval.run_translation_task", fake_run
        )
        client = _client(make_client, monkeypatch=monkeypatch)

        resp = client.post(
            "/eval/translate/v2",
            json={
                "original_text": "It is a watch.",
                "user_input": "它是一块手表。",
                "scholar_id": "s1",
                "sentence_id": "sent_1",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["task_id"].startswith("tr_")
        assert data["status"] == "pending"

        # 后台执行器已调度，参数正确
        assert called["task_id"] == data["task_id"]
        assert called["mode"] == "ec"  # 英文原句 → 英译中
        assert called["input_mode"] == "text"
        assert called["original_text"] == "It is a watch."
        assert called["user_input"] == "它是一块手表。"
        assert called["scholar_id"] == "s1"
        assert called["sentence_id"] == "sent_1"

        # 任务已落库且为 pending；audio_base64 不落库
        stored = fake_db.all("translation_task")
        assert len(stored) == 1
        assert stored[0]["status"] == "pending"
        assert stored[0]["mode"] == "ec"
        assert stored[0]["audio_base64"] is None

    def test_voice_submit_infers_ce_mode(self, make_client, monkeypatch, fake_db):
        called = {}

        async def fake_run(task_id, **kwargs):
            called.update(kwargs)

        monkeypatch.setattr(
            "services.routes_eval.run_translation_task", fake_run
        )
        client = _client(make_client, monkeypatch=monkeypatch)
        resp = client.post(
            "/eval/translate/v2",
            json={
                "original_text": "这是一块手表。",
                "audio_base64": FAKE_AUDIO,
                "voice_format": "mp3",
            },
        )
        assert resp.status_code == 200
        assert called["mode"] == "ce"  # 中文原句 → 中译英
        assert called["input_mode"] == "voice"
        assert called["audio_base64"] == FAKE_AUDIO  # 仅透传 worker，不落库

    def test_missing_original_text(self, make_client, monkeypatch, fake_db):
        client = _client(make_client, monkeypatch=monkeypatch)
        resp = client.post("/eval/translate/v2", json={"user_input": "hi"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"
        assert fake_db.all("translation_task") == []

    def test_missing_both_inputs(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch=monkeypatch)
        resp = client.post("/eval/translate/v2", json={"original_text": "Hello"})
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_invalid_base64(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch=monkeypatch)
        resp = client.post(
            "/eval/translate/v2",
            json={"original_text": "Hello", "audio_base64": "!!!not-base64!!!"},
        )
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_AUDIO"

    def test_empty_audio(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch=monkeypatch)
        resp = client.post(
            "/eval/translate/v2",
            json={"original_text": "Hello", "audio_base64": ""},
        )
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_AUDIO"

    def test_audio_too_large(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch=monkeypatch)
        huge = base64.b64encode(b"x" * (5 * 1024 * 1024 + 1)).decode()
        resp = client.post(
            "/eval/translate/v2",
            json={"original_text": "Hello", "audio_base64": huge},
        )
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_AUDIO"
        assert "5MB" in body["message"]


class TestEvalTranslateV2Get:
    """GET /eval/translate/v2/task/{task_id}"""

    def test_ok_pending(self, make_client, fake_db):
        client = _client(make_client)
        seed_translation_task(fake_db)
        resp = client.get("/eval/translate/v2/task/tr_test")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["task_id"] == "tr_test"
        assert data["status"] == "pending"
        assert data["result"] is None
        assert data["error"] is None

    def test_ok_success_with_result(self, make_client, fake_db):
        client = _client(make_client)
        result = {
            "transcription": "它是一块手表。",
            "status": 5,
            "feedback": "完全正确",
            "confidence": 0.9,
            "raw_model_output": '{"status": 5}',
        }
        seed_translation_task(fake_db, status="success", result=result)
        resp = client.get("/eval/translate/v2/task/tr_test")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "success"
        assert data["result"] == result
        assert data["error"] is None

    def test_ok_failed_error_shows_detail_string(self, make_client, fake_db):
        """error 对象 → 响应展示可读 error_detail 字符串（前端直接展示）。"""
        client = _client(make_client)
        seed_translation_task(
            fake_db,
            status="failed",
            error={
                "error_code": "LLM_TIMEOUT",
                "error_detail": "LLM 调用超过 300s 未返回",
                "failure_stage": "llm",
                "llm_timeout_seconds": 300,
                "raw": None,
            },
        )
        resp = client.get("/eval/translate/v2/task/tr_test")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "failed"
        assert data["error"] == "LLM 调用超过 300s 未返回"
        assert data["result"] is None

    def test_not_found(self, make_client, fake_db):
        client = _client(make_client)
        resp = client.get("/eval/translate/v2/task/tr_missing")
        assert resp.status_code == 404
        assert "任务不存在" in resp.json()["detail"]

    def test_expired_returns_404_doc_kept(self, make_client, fake_db):
        client = _client(make_client)
        seed_translation_task(fake_db, expires_at=int(time.time() * 1000) - 1000)
        resp = client.get("/eval/translate/v2/task/tr_test")
        assert resp.status_code == 404
        assert "已过期" in resp.json()["detail"]
        assert len(fake_db.all("translation_task")) == 1  # TTL 清理由提交接口巡检执行

    def test_query_revives_stale_processing_task(self, make_client, fake_db):
        """GET 定点自愈：被查询的卡死 processing 任务 → failed + LLM_TIMEOUT。"""
        client = _client(make_client)
        now = int(time.time() * 1000)
        seed_translation_task(
            fake_db,
            task_id="tr_stale",
            status="processing",
            updated_at=now - 130_000,
            expires_at=now + TASK_TTL_MS,
        )
        resp = client.get("/eval/translate/v2/task/tr_stale")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "failed"
        assert data["error"] == "执行超时"
        stored = fake_db.all("translation_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"]["error_code"] == "LLM_TIMEOUT"

    def test_query_fresh_processing_unaffected(self, make_client, fake_db):
        client = _client(make_client)
        now = int(time.time() * 1000)
        seed_translation_task(fake_db, task_id="tr_ok")
        seed_translation_task(
            fake_db,
            task_id="tr_stale",
            status="processing",
            updated_at=now - 130_000,
            expires_at=now + TASK_TTL_MS,
        )
        resp = client.get("/eval/translate/v2/task/tr_ok")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "pending"
        by_id = {d["task_id"]: d for d in fake_db.all("translation_task")}
        assert by_id["tr_stale"]["status"] == "processing"


class TestRunTranslationTask:
    """后台执行器 run_translation_task（patch LLM/ASR 替身，不触真实外部）"""

    def test_success_writes_result_and_evaluation(self, make_client, monkeypatch, fake_db):
        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm",
            lambda *a, **k: MODEL_OK,
        )
        task = seed_translation_task(fake_db)
        _run(
            run_translation_task(
                task["task_id"],
                original_text="It is a watch.",
                mode="ec",
                input_mode="text",
                user_input="它是一块手表。",
                scholar_id="s1",
                sentence_id="sent_1",
            )
        )

        stored = fake_db.all("translation_task")[0]
        assert stored["status"] == "success"
        assert stored["result"]["status"] == 4
        assert stored["result"]["feedback"] == "用词准确，注意时态"
        assert stored["result"]["transcription"] == "它是一块手表。"
        assert stored["error"] is None

        # 成功终态双写：evaluation 一条，succeeded=true
        evals = fake_db.all("evaluation")
        assert len(evals) == 1
        assert evals[0]["type"] == "translation"
        assert evals[0]["succeeded"] is True
        assert evals[0]["status"] == 4
        assert evals[0]["task_id"] == task["task_id"]
        assert evals[0]["scholar_id"] == "s1"
        assert evals[0]["mode"] == "ec"

    def test_voice_path_asr_and_score(self, make_client, monkeypatch, fake_db):
        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm",
            lambda *a, **k: '{"status": 5, "feedback": "完全正确"}',
        )
        asr = FakeAsrService()  # 默认转写 "it is a watch"
        # 后台执行器非 Depends 依赖：patch worker 模块命名空间的 get_asr_service
        monkeypatch.setattr(
            "services.translation_task.get_asr_service", lambda: asr, raising=False
        )
        task = seed_translation_task(fake_db, input_mode="voice", mode="ec")
        _run(
            run_translation_task(
                task["task_id"],
                original_text="It is a watch.",
                mode="ec",
                input_mode="voice",
                audio_base64=FAKE_AUDIO,
                voice_format="mp3",
                scholar_id="s1",
            )
        )
        stored = fake_db.all("translation_task")[0]
        assert stored["status"] == "success"
        assert stored["result"]["transcription"] == "it is a watch"
        assert stored["result"]["status"] == 5
        assert asr.call_count == 1

    def test_asr_unavailable_fails_with_trace(self, make_client, monkeypatch, fake_db):
        """语音路径 ASR 不可用 → failed + ASR_UNAVAILABLE + 失败留痕。"""
        monkeypatch.setattr(
            "services.translation_task.get_asr_service",
            lambda: FakeAsrService.unavailable(),
            raising=False,
        )
        task = seed_translation_task(fake_db, input_mode="voice", mode="ec")
        _run(
            run_translation_task(
                task["task_id"],
                original_text="It is a watch.",
                mode="ec",
                input_mode="voice",
                audio_base64=FAKE_AUDIO,
                scholar_id="s1",
            )
        )
        stored = fake_db.all("translation_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"]["error_code"] == "ASR_UNAVAILABLE"
        assert stored["error"]["failure_stage"] == "asr"
        assert stored["result"] is None

        # 失败留痕：evaluation succeeded=false + 同款 error 五字段
        evals = fake_db.all("evaluation")
        assert len(evals) == 1
        assert evals[0]["succeeded"] is False
        assert evals[0]["error_code"] == "ASR_UNAVAILABLE"
        assert evals[0]["failure_stage"] == "asr"
        assert evals[0]["llm_timeout_seconds"] == 300

    def test_llm_unavailable_fails(self, make_client, monkeypatch, fake_db):
        """no_external_calls 默认屏蔽 LLM（返回 None）→ EVAL_UNAVAILABLE，不降级。"""
        task = seed_translation_task(fake_db)
        _run(
            run_translation_task(
                task["task_id"],
                original_text="It is a watch.",
                mode="ec",
                input_mode="text",
                user_input="它是一块手表。",
            )
        )
        stored = fake_db.all("translation_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"]["error_code"] == "EVAL_UNAVAILABLE"
        assert stored["error"]["failure_stage"] == "llm"
        evals = fake_db.all("evaluation")
        assert evals[0]["succeeded"] is False
        assert evals[0]["error_code"] == "EVAL_UNAVAILABLE"

    def test_parse_error_fails(self, make_client, monkeypatch, fake_db):
        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm",
            lambda *a, **k: "完全无法解析",
        )
        task = seed_translation_task(fake_db)
        _run(
            run_translation_task(
                task["task_id"],
                original_text="It is a watch.",
                mode="ec",
                input_mode="text",
                user_input="它是一块手表。",
            )
        )
        stored = fake_db.all("translation_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"]["error_code"] == "LLM_PARSE_ERROR"
        assert stored["error"]["failure_stage"] == "parse"
        evals = fake_db.all("evaluation")
        assert evals[0]["succeeded"] is False
        assert evals[0]["error_code"] == "LLM_PARSE_ERROR"

    def test_llm_timeout_config_small_value_applies_and_records(
        self, make_client, monkeypatch, fake_db
    ):
        """§12：TRANSLATION_LLM_TIMEOUT_SECONDS 按配置生效（默认 300s 与较小值均须生效）。

        较小值用 0.05s（毫秒级，测试提速），使挂起的 LLM 调用（0.3s）必然被
        asyncio.wait_for 强制取消 → failed + LLM_TIMEOUT，且 task/evaluation 落库
        记录均带本次生效的 llm_timeout_seconds=0.05（审计闭环）。
        """
        import time as _time

        def hanging_llm(*a, **k):
            _time.sleep(0.3)  # 远超 0.05s 上限 → wait_for 强制取消（不返回 None）

        monkeypatch.setattr(
            "services.translation_task.TRANSLATION_LLM_TIMEOUT_SECONDS", 0.05
        )
        monkeypatch.setattr(
            "services.translation_eval.TRANSLATION_LLM_TIMEOUT_SECONDS", 0.05
        )
        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm", hanging_llm
        )
        task = seed_translation_task(fake_db)
        _run(
            run_translation_task(
                task["task_id"],
                original_text="It is a watch.",
                mode="ec",
                input_mode="text",
                user_input="它是一块手表。",
            )
        )
        stored = fake_db.all("translation_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"]["error_code"] == "LLM_TIMEOUT"
        assert stored["error"]["failure_stage"] == "llm"
        assert stored["error"]["llm_timeout_seconds"] == 0.05
        # 失败留痕带本次生效的超时配置值（审计闭环）
        evals = fake_db.all("evaluation")
        assert evals[0]["succeeded"] is False
        assert evals[0]["error_code"] == "LLM_TIMEOUT"
        assert evals[0]["llm_timeout_seconds"] == 0.05

    def test_network_error_fails_with_trace(
        self, make_client, monkeypatch, fake_db
    ):
        """§12：模型 5xx / 网络异常（LLM 调用抛通用异常）→ NETWORK_ERROR + 失败全量留痕。"""
        def boom(*a, **k):
            raise RuntimeError("connection reset by peer")

        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm", boom
        )
        task = seed_translation_task(fake_db)
        _run(
            run_translation_task(
                task["task_id"],
                original_text="It is a watch.",
                mode="ec",
                input_mode="text",
                user_input="它是一块手表。",
            )
        )
        stored = fake_db.all("translation_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"]["error_code"] == "NETWORK_ERROR"
        assert stored["error"]["failure_stage"] == "llm"
        assert "connection reset" in stored["error"]["error_detail"]
        evals = fake_db.all("evaluation")
        assert evals[0]["succeeded"] is False
        assert evals[0]["error_code"] == "NETWORK_ERROR"

    def test_empty_input_fails(self, make_client, monkeypatch, fake_db):
        task = seed_translation_task(fake_db, user_input="")
        _run(
            run_translation_task(
                task["task_id"],
                original_text="It is a watch.",
                mode="ec",
                input_mode="text",
                user_input="",
            )
        )
        stored = fake_db.all("translation_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"]["error_code"] == "EVAL_UNAVAILABLE"

    def test_claim_failed_skips_execution(self, make_client, monkeypatch, fake_db):
        """任务已被抢占（processing）→ 执行器直接返回，不调用 LLM。"""
        calls = {"llm": 0}

        def record(*a, **k):
            calls["llm"] += 1
            return MODEL_OK

        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm", record
        )
        task = seed_translation_task(fake_db, status="processing")
        _run(
            run_translation_task(
                task["task_id"],
                original_text="It is a watch.",
                mode="ec",
                input_mode="text",
                user_input="它是一块手表。",
            )
        )
        assert calls["llm"] == 0
        stored = fake_db.all("translation_task")[0]
        assert stored["status"] == "processing"


class TestEvalTranslateV2ZhSubmit:
    """POST /eval/translate/v2/zh — 中文语音识别版（2026-09-02，英译中语音作答专用）。

    与 POST /eval/translate/v2 唯一差异 = 语音路径 ASR 固定中文引擎 16k_zh
    （任务落库 asr_engine=16k_zh，执行器据此取中文引擎转写）。
    入参/校验/状态机/TTL/失败留痕与 v2 同构；查询复用 GET /eval/translate/v2/task/{task_id}。
    """

    def test_zh_text_submit_ok_records_asr_engine(
        self, make_client, monkeypatch, fake_db
    ):
        """文字路径：200 + pending + 任务记 asr_engine=16k_zh（文字路径无 ASR，行为同 v2）。"""
        called = {}

        async def fake_run(task_id, **kwargs):
            called["task_id"] = task_id
            called.update(kwargs)

        monkeypatch.setattr("services.routes_eval.run_translation_task", fake_run)
        client = _client(make_client, monkeypatch=monkeypatch)

        resp = client.post(
            "/eval/translate/v2/zh",
            json={
                "original_text": "It is a watch.",  # 英译中：英文原句
                "user_input": "它是一块手表。",  # 中文译文（文字路径）
                "scholar_id": "s1",
                "sentence_id": "sent_1",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["task_id"].startswith("tr_")
        assert data["status"] == "pending"

        # 后台执行器已调度：mode=ec（英文原句），唯一差异 asr_engine=16k_zh
        assert called["task_id"] == data["task_id"]
        assert called["mode"] == "ec"
        assert called["input_mode"] == "text"
        assert called["original_text"] == "It is a watch."
        assert called["user_input"] == "它是一块手表。"
        assert called["asr_engine"] == "16k_zh"

        # 任务落库记录中文引擎
        stored = fake_db.all("translation_task")
        assert len(stored) == 1
        assert stored[0]["status"] == "pending"
        assert stored[0]["mode"] == "ec"
        assert stored[0]["asr_engine"] == "16k_zh"

    def test_zh_voice_submit_passes_audio_with_zh_engine(
        self, make_client, monkeypatch, fake_db
    ):
        """语音路径：audio_base64 仅透传 worker（不落库），asr_engine=16k_zh 即本接口语义。"""
        called = {}

        async def fake_run(task_id, **kwargs):
            called.update(kwargs)

        monkeypatch.setattr("services.routes_eval.run_translation_task", fake_run)
        client = _client(make_client, monkeypatch=monkeypatch)

        resp = client.post(
            "/eval/translate/v2/zh",
            json={
                "original_text": "It is a watch.",
                "audio_base64": FAKE_AUDIO,
                "voice_format": "mp3",
            },
        )
        assert resp.status_code == 200
        assert called["mode"] == "ec"
        assert called["input_mode"] == "voice"
        assert called["audio_base64"] == FAKE_AUDIO  # 仅透传 worker，不落库
        assert called["asr_engine"] == "16k_zh"
        stored = fake_db.all("translation_task")[0]
        assert stored["audio_base64"] is None
        assert stored["asr_engine"] == "16k_zh"

    def test_zh_voice_worker_transcribes_chinese(
        self, make_client, monkeypatch, fake_db
    ):
        """执行器闭环：asr_engine=16k_zh → 取中文引擎服务 → 转写为正确中文 → LLM 评分 → 双写。"""
        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm",
            lambda *a, **k: '{"status": 5, "feedback": "翻译准确"}',
        )
        zh_asr = FakeAsrService(result="它是一块手表。")  # 中文引擎替身
        monkeypatch.setattr(
            "services.translation_task.get_asr_service_for",
            lambda engine: zh_asr,
            raising=False,
        )
        task = seed_translation_task(fake_db, input_mode="voice", mode="ec")
        _run(
            run_translation_task(
                task["task_id"],
                original_text="It is a watch.",
                mode="ec",
                input_mode="voice",
                audio_base64=FAKE_AUDIO,
                voice_format="mp3",
                scholar_id="s1",
                asr_engine="16k_zh",
            )
        )
        stored = fake_db.all("translation_task")[0]
        assert stored["status"] == "success"
        # 与 16k_en 分支的差异核心：transcription 为正确中文（不再被英文引擎转写为错文本）
        assert stored["result"]["transcription"] == "它是一块手表。"
        assert stored["result"]["status"] == 5
        assert zh_asr.call_count == 1

        # 终态双写 evaluation（type=translation, succeeded=true, user_input=中文转写）
        evals = fake_db.all("evaluation")
        assert len(evals) == 1
        assert evals[0]["succeeded"] is True
        assert evals[0]["mode"] == "ec"
        assert evals[0]["user_input"] == "它是一块手表。"

    def test_zh_voice_worker_asr_unavailable_fails_with_trace(
        self, make_client, monkeypatch, fake_db
    ):
        """zh 语音路径 ASR 不可用 → failed + ASR_UNAVAILABLE + 失败留痕（与 v2 同构）。"""
        monkeypatch.setattr(
            "services.translation_task.get_asr_service_for",
            lambda engine: FakeAsrService.unavailable(),
            raising=False,
        )
        task = seed_translation_task(fake_db, input_mode="voice", mode="ec")
        _run(
            run_translation_task(
                task["task_id"],
                original_text="It is a watch.",
                mode="ec",
                input_mode="voice",
                audio_base64=FAKE_AUDIO,
                voice_format="mp3",
                asr_engine="16k_zh",
            )
        )
        stored = fake_db.all("translation_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"]["error_code"] == "ASR_UNAVAILABLE"
        assert stored["error"]["failure_stage"] == "asr"
        evals = fake_db.all("evaluation")
        assert evals[0]["succeeded"] is False
        assert evals[0]["error_code"] == "ASR_UNAVAILABLE"
