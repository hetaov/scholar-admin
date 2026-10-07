"""v2 会话 LLM 的「思考」开关（2026-10-07 Z07 真机走查补齐）

背景：小程序 AI 英文会话走 `POST /ai/session/v2` → `services/learning/session_task.py`
→ `services/providers/session_gen.py::_call_session_llm`。该 provider 此前**完全没有透传
`thinking` 字段**（payload 只有 model / messages / temperature / response_format），而它用的
`VOLCANO_CHAT_MODEL` 是**推理型模型**（先产出 reasoning_content，content 几乎最后才出）
⇒ 与 v4 同因同果的慢（v4 实测首帧 ~12s → ~1.4s）。

本文件锁定：① 默认（`SESSION_V2_THINKING_DISABLED=1`）payload 带 `thinking.type=disabled`；
② 置 0 时**不带**该字段（可复现「思考开」对照）；③ 既有字段不被改动。
"""

import json

import pytest

import services.providers.session_gen as sg


class _FakeResp:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": json.dumps({"ok": True})}}]}


def _capture_payload(monkeypatch) -> dict:
    """桩掉 requests.post，返回被调用时捕获的 payload（dict）。"""
    captured: dict = {}

    monkeypatch.setattr(sg, "VOLCANO_API_KEY", "test-key")
    monkeypatch.setattr(sg, "VOLCANO_CHAT_MODEL", "test-model")
    monkeypatch.setattr(sg, "VOLCANO_BASE_URL", "http://volcano.test")
    monkeypatch.setattr(sg, "SESSION_LLM_TIMEOUT_SECONDS", 5)

    def _post(url, headers=None, json=None, timeout=None):  # noqa: A002
        captured["url"] = url
        captured.update(json or {})
        return _FakeResp()

    import requests

    monkeypatch.setattr(requests, "post", _post)
    return captured


def test_thinking_disabled_by_default(monkeypatch):
    """默认关思考：payload 带 thinking.type=disabled。"""
    captured = _capture_payload(monkeypatch)
    monkeypatch.setattr(sg, "SESSION_V2_THINKING_DISABLED", 1)

    out = sg._call_session_llm([{"role": "user", "content": "hi"}])

    assert captured["thinking"] == {"type": "disabled"}
    # 既有字段零改动
    assert captured["model"] == "test-model"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["response_format"] == {"type": "json_object"}
    assert isinstance(captured["temperature"], float)
    assert out == json.dumps({"ok": True})


def test_thinking_flag_zero_reverts_to_current_behavior(monkeypatch):
    """置 0 → 不带 thinking 字段（与补齐前逐字一致，可作对照实验）。"""
    captured = _capture_payload(monkeypatch)
    monkeypatch.setattr(sg, "SESSION_V2_THINKING_DISABLED", 0)

    sg._call_session_llm([{"role": "user", "content": "hi"}])

    assert "thinking" not in captured


def test_thinking_field_helper_contract():
    """字段形态与 v4 / extension 两处同名 helper 保持一致。"""
    assert sg._thinking_field(True) == {"thinking": {"type": "disabled"}}
    assert sg._thinking_field(False) == {}


@pytest.mark.parametrize("status", [400, 500])
def test_non_200_returns_none_without_thinking_regression(monkeypatch, status):
    """非 200 仍返回 None（错误路径不受本次改动影响）。"""
    monkeypatch.setattr(sg, "VOLCANO_API_KEY", "test-key")
    monkeypatch.setattr(sg, "VOLCANO_CHAT_MODEL", "test-model")
    monkeypatch.setattr(sg, "VOLCANO_BASE_URL", "http://volcano.test")
    monkeypatch.setattr(sg, "SESSION_V2_THINKING_DISABLED", 1)

    class _Bad:
        status_code = status
        text = "boom"

    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Bad())

    assert sg._call_session_llm([{"role": "user", "content": "hi"}]) is None
