"""F4.2 OCR Provider 抽象与腾讯云通用印刷体 OCR 实现。

契约：
- service-contract.md §7.4：OcrProvider 抽象（recognize(image_bytes) -> OcrResult(text, blocks)）
- ADR-0020：MVP 选型腾讯云通用印刷体 OCR，凭据复用 config.py（TENCENTCLOUD_SECRETID 同源）
- data-model-contract.md §4.12.9：结果回写 math_scan_upload.ocr_text / ocr_blocks[] / ocr_status

设计：
- error_scanner._run_ocr_job 只依赖 OcrProvider 抽象，厂商可替换（P2+ 手写体/公式专项）。
- TencentGeneralOcrProvider 内部失败自动重试 1 次；仍失败抛 OcrError，
  由调用方（error_scanner）降级 ocr_status=failed，不阻断上传链路。
- 无凭据时抛 OcrConfigError（OcrError 子类），同样触发 failed 降级而非崩溃。

使用：
    from services.math import ocr
    provider = ocr.get_provider()
    result = await provider.recognize(image_bytes)   # result.text / result.blocks
"""

from __future__ import annotations

import asyncio
import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 腾讯云 OCR 接口与请求类名映射（ADR-0020：MVP 通用印刷体二选一）
_OCR_ENGINES = {
    "general_accurate": "GeneralAccurateOCR",
    "general_fast": "GeneralFastOCR",
}


class OcrError(Exception):
    """OCR 调用失败（网络/超时/服务端错误），调用方应降级 ocr_status=failed。"""


class OcrConfigError(OcrError):
    """OCR 未配置可用凭据，无法发起调用。"""


@dataclass
class OcrResult:
    """OCR 识别结果。

    text:   全文文本（检测块按自上而下顺序以换行拼接，保留题号行结构）
    blocks: 检测块列表，契约形状 {block_id, text, bbox, image_url_crop}
            （image_url_crop 由 P2 人工修正链路填充，MVP 恒为 None）
    """

    text: str
    blocks: list[dict[str, Any]] = field(default_factory=list)


class OcrProvider(ABC):
    """OCR Provider 抽象（service-contract.md §7.4）。"""

    @property
    @abstractmethod
    def available(self) -> bool:
        """是否具备可用的 OCR 凭据。"""

    @abstractmethod
    async def recognize(self, image_bytes: bytes) -> OcrResult:
        """识别图片，返回结构化文本与检测块。失败抛 OcrError。"""


class TencentGeneralOcrProvider(OcrProvider):
    """腾讯云通用印刷体 OCR（GeneralAccurateOCR / GeneralFastOCR）。

    凭据复用 config（TENCENT_OCR_SECRET_ID/SECRET_KEY，默认回退 SECRET_ID/SECRET_KEY，
    即 CloudRun 注入的 TENCENTCLOUD_SECRETID/SECRETKEY），可用环境变量独立覆盖。
    """

    def __init__(
        self,
        secret_id: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
        engine: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        import config

        self._secret_id = secret_id if secret_id else config.TENCENT_OCR_SECRET_ID
        self._secret_key = secret_key if secret_key else config.TENCENT_OCR_SECRET_KEY
        self._region = region if region else config.TENCENT_OCR_REGION
        self._engine = engine if engine else config.TENCENT_OCR_ENGINE
        self._timeout_seconds = (
            timeout_seconds if timeout_seconds else config.TENCENT_OCR_TIMEOUT_SECONDS
        )

    @property
    def available(self) -> bool:
        return bool(self._secret_id and self._secret_key)

    def _build_client(self) -> Any:
        """延迟导入并构造腾讯云 OCR client（失败信息隐藏凭据）。"""
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.ocr.v20181119 import ocr_client

        cred = credential.Credential(self._secret_id, self._secret_key)
        http_profile = HttpProfile()
        http_profile.endpoint = "ocr.tencentcloudapi.com"
        http_profile.reqTimeout = self._timeout_seconds
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        return ocr_client.OcrClient(cred, self._region, client_profile)

    def _invoke_once(self, client: Any, image_b64: str) -> Any:
        """同步执行一次 OCR 调用并返回原始响应。

        正常生产路径：通过腾讯云 SDK 的 models 模块构造严格类型的 req。
        测试/降级路径（SDK 不可用，如本地 pytest + FakeClient）：
            用 SimpleNamespace 最小对象承载 ImageBase64 属性，直接调用
            传入的 client.engine_method(req) —— FakeClient 仅断言方法名，
            不校验 req 类型。
        """
        engine_method = _OCR_ENGINES.get(self._engine, "GeneralAccurateOCR")
        try:
            from tencentcloud.ocr.v20181119 import models  # type: ignore

            req = getattr(models, f"{engine_method}Request")()
        except Exception:  # noqa: BLE001 - 降级：允许测试环境无 SDK 时通过 mock
            from types import SimpleNamespace

            req = SimpleNamespace()
        req.ImageBase64 = image_b64
        return getattr(client, engine_method)(req)

    def _call_once(self, image_bytes: bytes) -> OcrResult:
        client = self._build_client()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = self._invoke_once(client, image_b64)
        return self._parse(resp)

    @staticmethod
    def _parse(resp: Any) -> OcrResult:
        """解析 TextDetections -> OcrResult（文本行 + 检测块）。"""
        detections = getattr(resp, "TextDetections", None) or []
        lines: list[str] = []
        blocks: list[dict[str, Any]] = []
        for index, item in enumerate(detections, start=1):
            text = (item.get("DetectedText") or "").strip()
            if not text:
                continue
            lines.append(text)
            bbox = [
                [float(point.get("X", 0)), float(point.get("Y", 0))]
                for point in (item.get("Polygon") or [])
            ]
            blocks.append(
                {
                    "block_id": f"blk_{index:04d}",
                    "text": text,
                    "bbox": bbox,
                    "image_url_crop": None,
                }
            )
        return OcrResult(text="\n".join(lines), blocks=blocks)

    async def recognize(self, image_bytes: bytes) -> OcrResult:
        if not self.available:
            raise OcrConfigError("OCR 未配置凭据（TENCENT_OCR_SECRET_ID/SECRET_KEY）")
        if not image_bytes:
            raise OcrError("OCR 入参为空：image_bytes")

        last_error: Exception | None = None
        for attempt in (1, 2):  # 首次 + 自动重试 1 次
            try:
                return await asyncio.to_thread(self._call_once, image_bytes)
            except OcrConfigError:
                raise
            except Exception as exc:  # 网络/超时/服务端异常 → 重试
                last_error = exc
                logger.warning("OCR 调用失败（第 %d 次），即将重试：%s", attempt, exc)
        raise OcrError(f"OCR 调用失败（已重试 1 次）：{last_error}") from last_error


_provider: OcrProvider | None = None


def get_provider() -> OcrProvider:
    """返回进程级 OCR Provider 单例（与 dependencies.get_ocr() 同源）。"""
    global _provider
    if _provider is None:
        _provider = TencentGeneralOcrProvider()
    return _provider
