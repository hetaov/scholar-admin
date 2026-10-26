"""音源统一取值：**fileID 优先**（2026-10-26 长音频修复）。

背景（硬约束，非代码 bug）
--------------------------
微信 `wx.cloud.callContainer` 的**请求包体上限仅 100KB**。官方《调用服务问题排查指引》：

    ERRCODE: -606001 | ERRMSG: 请求包大小超过 100KB
    解决方法：压缩请求包大小。请勿在请求体中加入图片/视频/文件。
    上述类型请先上传到「对象存储」后，仅将 fileID 通过请求体传给后端。

音频 base64 还要多 33% 开销 ⇒ **原始音频只能约 75KB**，录音稍长必报 -606001。

由此确立全项目语音入参的统一口径：

    请求体只带 `voice_file_id`（对象存储 fileID，几十字节）
    → 服务端用本模块自行从对象存储拉取字节
    ⇒ 彻底绕开 100KB 上限，任意时长可用。

`audio_base64` 分支**仅为兼容未升级客户端**保留，新功能一律不要再走它。

放这里的理由
------------
`services/routes/eval.py` 与 `services/learning/translation_task.py` **都需要**这套逻辑
（后者是 v2 的后台 worker），而 `translation_task` 反向导入 `routes.eval` 会成环
（eval 已依赖 translation_task）⇒ 下沉到本中立模块，两边都从这里导入。
"""

import base64
import binascii
import logging
from typing import Optional

from services.infra.tcb_storage import CloudBaseStorageClient, StorageAPIError

logger = logging.getLogger(__name__)

# 音频上限：16k 单声道 60s 约 1.9MB(mp3)，留冗余到 5MB（由来：既有实现同值）
MAX_AUDIO_BYTES = 5 * 1024 * 1024


class InvalidAudio(Exception):
    """音频不可用 → 由各路由映射为业务错误 `INVALID_AUDIO`（200 + success=false）。"""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def decode_audio_base64(raw_b64: str) -> Optional[bytes]:
    """解码音频 base64（兼容 data URL 前缀）；非法返回 None。"""
    s = raw_b64.strip()
    if "," in s and s.split(",", 1)[0].startswith("data:"):
        s = s.split(",", 1)[1]
    if not s:
        return None
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return None


async def resolve_audio_bytes(
    voice_file_id: Optional[str],
    audio_base64: Optional[str],
    *,
    route: str,
    oversize_message: str = "音频过大（16k/60s 上限约 5MB），请分段评测",
) -> Optional[bytes]:
    """音频**二选一**取值：fileID 优先，base64 兜底。

    Args:
        voice_file_id: 云存储 fileID（`wx.cloud.uploadFile` **回传**的完整 cloud:// ID）
        audio_base64: 音频 base64（旧客户端兼容路径）
        route: 调用方路由名，仅用于日志定位
        oversize_message: 超限文案（各路由措辞不同，故由调用方给）

    Returns:
        音频字节；**两者都未提供**时返回 `None` —— 交由各路由自己的
        「文本 / 语音二选一」或「必填」校验处理（保持既有契约不变）。

    Raises:
        InvalidAudio: 提供了但取不到 / 不合法 / 超限。
    """
    file_id = (voice_file_id or "").strip()
    raw_b64 = (audio_base64 or "").strip()

    if file_id:
        try:
            audio_bytes = await CloudBaseStorageClient().download_file(file_id)
        except StorageAPIError as e:
            logger.warning(f"[audio] {route} 云存储下载失败 file_id={file_id}: {e}")
            raise InvalidAudio("语音文件已失效，请重新录制") from e
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"[audio] {route} 下载异常 file_id={file_id}: {e}", exc_info=True
            )
            raise InvalidAudio("语音读取失败，请重试") from e
        if not audio_bytes:
            raise InvalidAudio("音频内容为空")
        # fileID 路径必须自设上限：否则后端会被动下载超大对象
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise InvalidAudio(oversize_message)
        return audio_bytes

    if raw_b64:
        # ⚠️ 兼容旧客户端：本分支**容纳不了长音频**（见模块文档 100KB 说明）
        audio_bytes = decode_audio_base64(raw_b64)
        if audio_bytes is None:
            raise InvalidAudio("audio_base64 不是合法 base64")
        if not audio_bytes:
            raise InvalidAudio("音频内容为空")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise InvalidAudio(oversize_message)
        return audio_bytes

    return None
