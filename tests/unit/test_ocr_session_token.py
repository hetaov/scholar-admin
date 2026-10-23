"""腾讯云 OCR 凭据必须带 SESSION_TOKEN（2026-10-23 修复：线上 ocr_status 恒 failed）

被测对象：services/math/ocr.py

背景（线上实测）：拍照上传后 scan 记录 `ocr_status=failed`（而非 provider 未接入的
`pending`）→ classify / quiz-recognize 永远拿不到 OCR 文本 → 整条扫描链路不可用。

根因：云托管（CloudRun）注入的是 **STS 临时密钥**（TENCENTCLOUD_SECRETID/SECRETKEY
+ SESSIONTOKEN），临时密钥签名必须带 token。而 OCR 是**唯一**漏传该参数的腾讯云 SDK
调用 —— ASR（providers/asr.py:77）与 TTS（providers/tts.py:132）都传了
`session_token or None`。

覆盖用例：
  - 显式传 session_token → Credential 第三参 == 该 token
  - 未显式传 → 取 config.TENCENT_OCR_SESSION_TOKEN（配置路径）
  - 无 token → 传 **None**（不是空串；空串会被 SDK 当作有效 token 参与签名）
"""
from __future__ import annotations

import sys
import types

import pytest

from config import SESSION_TOKEN as CONFIG_SESSION_TOKEN
from services.math import ocr


def _install_fake_sdk(monkeypatch, captured: dict) -> None:
    """注入假 tencentcloud SDK：只记录 Credential 入参，不发任何网络请求"""

    def mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    mod("tencentcloud")
    mod("tencentcloud.common")
    cred_mod = mod("tencentcloud.common.credential")

    class Credential:
        def __init__(self, secret_id, secret_key, token=None):  # noqa: ANN001
            captured["cred"] = (secret_id, secret_key, token)

    cred_mod.Credential = Credential

    mod("tencentcloud.common.profile")
    cp = mod("tencentcloud.common.profile.client_profile")

    class ClientProfile:
        httpProfile = None

    cp.ClientProfile = ClientProfile
    hp = mod("tencentcloud.common.profile.http_profile")

    class HttpProfile:
        endpoint = ""
        reqTimeout = 0

    hp.HttpProfile = HttpProfile

    mod("tencentcloud.ocr")
    mod("tencentcloud.ocr.v20181119")
    oc = mod("tencentcloud.ocr.v20181119.ocr_client")

    class OcrClient:
        def __init__(self, cred, region, profile):  # noqa: ANN001
            captured["client"] = (cred, region, profile)

    oc.OcrClient = OcrClient


def test_credential_carries_session_token(monkeypatch):
    """显式 token → 原样传给 Credential（修复点）"""
    captured: dict = {}
    _install_fake_sdk(monkeypatch, captured)

    provider = ocr.TencentGeneralOcrProvider(
        secret_id="sid", secret_key="sk", session_token="sts-token-abc"
    )
    provider._build_client()

    assert captured["cred"] == ("sid", "sk", "sts-token-abc")


def test_falls_back_to_config_token(monkeypatch):
    """未显式传 → 取 config.TENCENT_OCR_SESSION_TOKEN"""
    captured: dict = {}
    _install_fake_sdk(monkeypatch, captured)
    monkeypatch.setattr("config.TENCENT_OCR_SESSION_TOKEN", "token-from-config")

    ocr.TencentGeneralOcrProvider(secret_id="sid", secret_key="sk")._build_client()

    assert captured["cred"][2] == "token-from-config"


def test_no_token_passes_none(monkeypatch):
    """无 token → 传 None（空串会被 SDK 当成有效 token，与 ASR/TTS 口径一致）"""
    captured: dict = {}
    _install_fake_sdk(monkeypatch, captured)
    monkeypatch.setattr("config.TENCENT_OCR_SESSION_TOKEN", "")

    ocr.TencentGeneralOcrProvider(secret_id="sid", secret_key="sk")._build_client()

    assert captured["cred"] == ("sid", "sk", None)


def test_config_default_inherits_cloudrun_token():
    """config 默认值应继承 CloudRun 注入的 SESSION_TOKEN（防回归：勿写成空串默认）"""
    import config

    assert config.TENCENT_OCR_SESSION_TOKEN == (
        config.TENCENT_OCR_SESSION_TOKEN or CONFIG_SESSION_TOKEN
    )
