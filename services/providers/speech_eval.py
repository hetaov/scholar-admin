"""SOE-N 句级口语评测 Provider（F2/2.2）

SpeechProvider 抽象接口 + TencentSoeNProvider 实现（WSS 录音模式 rec_mode=1）。

- 调用面复用 F1-2 已验证的 scripts/soe_n_verify.py：官方 tencentcloud-speech-sdk-python
  为纯源码分发（PyPI 无包，2026-08-17 实测），vendor 目录随镜像发布，运行时 sys.path 引用仓库根目录
- 注入模式对齐 services/asr.py 的 get_asr_service（模块级单例）
- evaluate 返回 SOE-N 完整原始 JSON（供路由层落 speech_evaluation.raw）；
  normalize_soe_result 按 api-contract §3.4.2 给出 parsed 归一口径
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from config import (
    TCB_APPID,
    SESSION_TOKEN,
    SECRET_ID,
    SECRET_KEY,
    SPEECH_EVAL_MODE,   # v4 后修：评测模式 auto/1(句子)/2(段落)，缺省 auto 按句长自适应
)

logger = logging.getLogger("scholar-admin.speech_eval")

# 官方 SDK 源码分发目录（随镜像发布，见 scripts/soe_n_verify.py 安装说明）
# 本文件在 services/providers/ 下 → 仓库根是 parents[2]（原在 services/ 时为 parent.parent，搬入
# providers/ 后少算一层，于是去找不存在的 services/vendor/ → 评测恒报 sdk_missing；2026-09-22 修正）
SDK_DIR = Path(__file__).resolve().parents[2] / "vendor" / "tencentcloud-speech-sdk-python"

# SOE-N 评测原始结果存档集合（data-model-contract §4.9；需 scripts/init_speech_evaluation_collection.py 幂等建表）
SPEECH_EVALUATION_COLLECTION = "speech_evaluation"

# SOE-N 语音评测参数（与 F1-2 实测一致）
SOE_ENGINE = "16k_en"          # 英语 16k
SOE_REC_MODE = 1               # 1 = 录音模式（一次性上传完整音频，≤60s）
SOE_TEXT_MODE = 0              # 0 = 普通文本

# 评测模式（官方 EvalMode：0=单词 / 1=句子 / 2=段落 / 3=自由说）
SOE_EVAL_MODE_SENTENCE = 1
SOE_EVAL_MODE_PARAGRAPH = 2
# 官方文本长度上限：句子模式 ≤30 词、段落模式 ≤120 词（2026-09-21 查证）
SOE_SENTENCE_MAX_WORDS = 30
# 我们的接受上限（2026-09-21 用户拍板 30 → 90）：≤30 走句子模式，31~90 自动切**段落模式**，
# >90 直接判「本句过长」不发起评测。低于官方段落上限 120，留余量避免边界抖动。
SOE_MAX_REF_WORDS = 90

# 语音评测失败原因（2026-09-21 后修）：token 为机器可读（客户端据此分流），text 为可读文案（拼进响应 message）。
# 背景：此前 5 种失败在路由层被折叠成同一句 message，客户端无法区分「服务不可用」与「本句不支持评测」。
SPEECH_FAIL_REASON_TEXT = {
    "no_credentials": "SOE-N 凭据未配置",
    "empty_audio": "音频内容为空",
    "ref_text_too_long": f"本句过长（超 {SOE_MAX_REF_WORDS} 词的评测上限）",
    "ref_text_empty": "缺少参考文本",
    "sdk_missing": "服务端缺少 SOE-N SDK",
    "eval_failed": "SOE 评测未通过",
    "timeout": "SOE 评测超时",
    "call_error": "SOE 调用异常",
    # 2026-10-15：aac → mp3 转码失败（见 _transcode_to_mp3 说明）
    "transcode_failed": "音频转码失败（服务端缺少 ffmpeg 或音频损坏）",
}
EVALUATE_TIMEOUT = 65.0        # 录音 ≤60s + 缓冲，超出视为评测失败

# 契约 voice_format 字符串 → SOE-N int（0=pcm / 1=wav / 2=mp3 / 4=speex）
VOICE_FORMAT_MAP = {"mp3": 2, "wav": 1}
DEFAULT_VOICE_FORMAT = "mp3"

# 2026-10-15：SOE-N 的 VoiceFileType **不含 aac**，但微信小程序在部分真机上
# 录不出 mp3 —— `RecorderManager` 的原生/默认格式是 aac，mp3 需额外转码，
# 而这些机器上该转码失败：起录正常（onStart 有）、能录满时长，但 `stop()` 后
# 微信 `onError` 报 **file error**（编码器产不出文件）⇒ 观感「所有录音都失败」。
# 客户端已改为「默认 mp3，file error 时自适应降级 aac」，故服务端必须能吃下 aac：
#   1. 路由层白名单放行 aac；
#   2. 本 Provider 在送 SOE 前把 aac 转成 mp3（ffmpeg）。
# ASR 侧（services/providers/asr.py）原生支持 aac，透传即可，无需转码。
SOE_TRANSCODE_TO_MP3 = {"aac"}
TRANSCODE_TIMEOUT = 20.0   # 单句 ≤60s，转码远快于此；超时即判失败，避免拖死评测


def _transcode_to_mp3(audio_bytes: bytes, src_format: str) -> Optional[bytes]:
    """把 SOE 不支持的格式（aac）转成 mp3（16k/单声道），失败返回 None。

    走 ffmpeg 子进程、纯管道输入输出（不落盘，避免并发下的临时文件清理问题）。
    ffmpeg 缺失时**直接返回 None** ⇒ 上层给出 `transcode_failed`，
    而不是让 SOE 收到无法解码的字节后超时（那样要等满 65s 才失败）。
    """
    if shutil.which("ffmpeg") is None:
        logger.error(
            "[speech_eval] 需要转码 %s→mp3，但镜像内缺少 ffmpeg（见 Dockerfile apt 依赖）。"
            "注意：改了 Dockerfile 必须**重新构建镜像**，只同步代码不会装上 ffmpeg。",
            src_format,
        )
        return None

    # 输入格式候选：**先自动探测**，失败再按声明格式强制解复用。
    # 原因：微信 RecorderManager(format:'aac') 产出的虽然扩展名是 .aac，但实际容器
    # 不是裸 ADTS —— 真机实测强制 `-f aac` 恒报 `Error opening input: End of file`，
    # 而自动探测能正常解出。顺序反过来会每次都白跑一次 ffmpeg（实测约 70ms）。
    for in_fmt in (None, src_format):
        try:
            cmd = ["ffmpeg", "-nostdin", "-y"]
            if in_fmt:
                cmd += ["-f", in_fmt]
            cmd += [
                "-i", "pipe:0",
                "-codec:a", "libmp3lame", "-ar", "16000", "-ac", "1",
                "-f", "mp3", "pipe:1",
            ]
            proc = subprocess.run(
                cmd,
                input=audio_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=TRANSCODE_TIMEOUT,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.error("[speech_eval] %s→mp3 转码异常: %s", src_format, e)
            continue

        if proc.returncode == 0 and proc.stdout:
            logger.info(
                "[speech_eval] %s→mp3 转码成功（%d → %d 字节，输入格式=%s）",
                src_format, len(audio_bytes), len(proc.stdout), in_fmt or "自动探测",
            )
            return proc.stdout
        logger.error(
            "[speech_eval] %s→mp3 转码失败（输入格式=%s）rc=%s stderr=%s",
            src_format, in_fmt or "自动探测", proc.returncode,
            (proc.stderr or b"")[-300:],
        )

    # 全部候选失败：打印魔数，一眼看出这段音频到底是什么容器
    # （裸 ADTS AAC 以 FF F1 / FF F9 开头；mp4/m4a 以 ....ftyp 开头）
    logger.error(
        "[speech_eval] 转码彻底失败：%d 字节，前 16 字节=%s",
        len(audio_bytes), audio_bytes[:16].hex(),
    )
    return None

# SDK 模块惰性加载缓存
_sdk_credential = None
_sdk_speaking_assessment = None


def _load_sdk() -> bool:
    """加载官方 SOE-N SDK（源码分发，vendor 目录加入 sys.path）。成功返回 True。"""
    global _sdk_credential, _sdk_speaking_assessment
    if _sdk_credential is not None:
        return True
    if not SDK_DIR.is_dir():
        logger.error(
            "[speech_eval] 缺少官方 SDK 源码目录: %s（镜像/部署产物未含 vendor/tencentcloud-speech-sdk-python；"
            "git 部署时 .gitignore 排除了 vendor/，需 Dockerfile 构建期在线拉取或 git add -f vendor 入库后重部署）",
            SDK_DIR,
        )
        return False
    if str(SDK_DIR) not in sys.path:
        sys.path.insert(0, str(SDK_DIR))
    try:
        from common import credential as _c
        from soe import speaking_assessment as _sa

        _sdk_credential = _c
        _sdk_speaking_assessment = _sa
        return True
    except ImportError:
        logger.error(
            "[speech_eval] 官方 SDK 源码目录存在但导入失败: %s（目录结构不完整？traceback 见下）",
            SDK_DIR,
            exc_info=True,
        )
        return False


def select_eval_mode(ref_text: str, override: str | int | None = None) -> int:
    """按句长选择 SOE 评测模式（纯函数，便于单测）。

    - `override` 为 `1`/`2`/`"1"`/`"2"` 时**强制**该模式（运维配置 `SPEECH_EVAL_MODE`）；
    - 否则自动：`ref_text` ≤ `SOE_SENTENCE_MAX_WORDS` 词 → 句子模式；否则 → **段落模式**
      （段落模式官方上限 120 词，故 31~90 词的句子不再被判「过长」而是走段落评测）。

    背景（2026-09-21）：句子模式官方文本上限仅 30 词，此前 30 也被当作我们的接受上限 →
    31 词以上的句子直接不可评测。放开到 90 词并自动切段落模式后，长句可正常评测。
    """
    if override is not None and str(override).strip() in ("1", "2"):
        return SOE_EVAL_MODE_SENTENCE if str(override).strip() == "1" else SOE_EVAL_MODE_PARAGRAPH
    words = len(str(ref_text or "").split())
    return SOE_EVAL_MODE_SENTENCE if words <= SOE_SENTENCE_MAX_WORDS else SOE_EVAL_MODE_PARAGRAPH


class SpeechProvider(ABC):
    """语音评测 Provider 抽象接口（供路由层依赖注入）"""

    @property
    @abstractmethod
    def available(self) -> bool:
        """是否具备可用凭据"""

    @abstractmethod
    def evaluate(
        self, audio_bytes: bytes, ref_text: str, voice_format: str = DEFAULT_VOICE_FORMAT
    ) -> Optional[dict]:
        """对一段音频做句级口语评测，返回 SOE-N 完整原始 JSON；失败返回 None。"""

    def evaluate_with_reason(
        self, audio_bytes: bytes, ref_text: str, voice_format: str = DEFAULT_VOICE_FORMAT
    ) -> tuple[Optional[dict], Optional[str]]:
        """带失败原因的评测（2026-09-21 后修）：返回 `(raw, reason)`。

        成功 → `(原始 JSON, None)`；失败 → `(None, reason_token)`，token 取自
        `SPEECH_FAIL_REASON_TEXT`，供路由层拼可读 message、客户端按 token 分流处置。
        默认实现委托 `evaluate`（reason 恒 None），故既有 fake / 存量实现无需改造。
        """
        return self.evaluate(audio_bytes, ref_text, voice_format), None


class _SoeNListener:
    """评测结果回调：最终结果在 on_recognition_complete 中返回（对齐 F1-2 SoeNListener）"""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Optional[dict] = None
        self.fail_reason: Optional[str] = None

    def on_recognition_start(self, response) -> None:
        logger.info("[speech_eval] 评测开始 voice_id=%s", response.get("voice_id"))

    def on_intermediate_result(self, response) -> None:
        pass  # 录音模式（rec_mode=1）不关注中间结果

    def on_recognition_complete(self, response) -> None:
        self.result = response
        self.event.set()

    def on_fail(self, response) -> None:
        self.fail_reason = str(response)
        self.event.set()


class TencentSoeNProvider(SpeechProvider):
    """SOE-N 口语评测客户端（WSS + HMAC-SHA1 签名，官方 SDK 内部处理）"""

    def __init__(
        self,
        appid: str = TCB_APPID,
        secret_id: str = SECRET_ID,
        secret_key: str = SECRET_KEY,
        session_token: str = SESSION_TOKEN,
    ) -> None:
        self._appid = appid
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._session_token = session_token

    @property
    def available(self) -> bool:
        """需要 AppID + SecretId + SecretKey（CloudRun 注入 / 本地 .env）"""
        ok = bool(self._appid and self._secret_id and self._secret_key)
        if not ok:
            # 详细诊断日志：区分缺失项（不打印密钥值，仅打印存在性）
            logger.warning(
                "[speech_eval] SOE-N 凭据不完整 → available=False "
                "（appid=%s secret_id=%s secret_key=%s；"
                "cloudrun 自动注入 TENCENTCLOUD_*，TCB_APPID 需在服务环境变量手动配置，改后需重启服务）",
                "OK" if self._appid else "MISSING",
                "OK" if self._secret_id else "MISSING",
                "OK" if self._secret_key else "MISSING",
            )
        return ok

    def evaluate(
        self, audio_bytes: bytes, ref_text: str, voice_format: str = DEFAULT_VOICE_FORMAT
    ) -> Optional[dict]:
        """兼容旧签名（失败返回 None）；内部委托 evaluate_with_reason。"""
        raw, _ = self.evaluate_with_reason(audio_bytes, ref_text, voice_format)
        return raw

    def evaluate_with_reason(
        self, audio_bytes: bytes, ref_text: str, voice_format: str = DEFAULT_VOICE_FORMAT
    ) -> tuple[Optional[dict], Optional[str]]:
        """录音模式（rec_mode=1）一次性上传完整音频做句级评测：`(原始 JSON, None)` 或 `(None, reason)`。

        失败原因逐条区分（2026-09-21 后修），路由层据此拼可读 message、客户端据 token 分流——
        其中 `ref_text_too_long` 是**唯一与句子内容相关**的成因（SOE 句级硬上限 30 词），
        其余为环境类（凭据/SDK/调用/超时），处置方式不同。
        """
        if not self.available:
            logger.warning(
                "[speech_eval] 未配置 TCB_APPID / TENCENTCLOUD_SECRETID/SECRETKEY，无法调用 SOE-N"
            )
            return None, "no_credentials"
        if not audio_bytes:
            logger.warning("[speech_eval] 音频内容为空")
            return None, "empty_audio"
        if not ref_text or len(ref_text.split()) > SOE_MAX_REF_WORDS:
            logger.warning(
                "[speech_eval] ref_text 为空或超 %d 词（句级评测上限；当前 %d 词）",
                SOE_MAX_REF_WORDS,
                len(str(ref_text or "").split()),
            )
            return None, ("ref_text_empty" if not ref_text else "ref_text_too_long")
        # 2026-10-15：aac 必须先转 mp3 才能送 SOE（VoiceFileType 无 aac）。
        # 位置在 _load_sdk() **之前**：转不了码就该以 transcode_failed 快速失败，
        # 而不是先撞上 sdk_missing（后者会误导读数、掩盖真正的转码问题）。
        if (voice_format or "").lower() in SOE_TRANSCODE_TO_MP3:
            mp3_bytes = _transcode_to_mp3(audio_bytes, voice_format.lower())
            if mp3_bytes is None:
                return None, "transcode_failed"
            audio_bytes, voice_format = mp3_bytes, "mp3"

        if not _load_sdk():
            return None, "sdk_missing"

        voice_format_int = VOICE_FORMAT_MAP.get(voice_format)
        if voice_format_int is None:
            logger.warning(
                "[speech_eval] 未知 voice_format=%r，回退 mp3", voice_format
            )
            voice_format_int = VOICE_FORMAT_MAP[DEFAULT_VOICE_FORMAT]

        eval_mode = select_eval_mode(ref_text, SPEECH_EVAL_MODE)
        logger.info(
            "[speech_eval] 评测模式=%s（ref_words=%d，句子上限 %d / 接受上限 %d）",
            "句子" if eval_mode == SOE_EVAL_MODE_SENTENCE else "段落",
            len(ref_text.split()), SOE_SENTENCE_MAX_WORDS, SOE_MAX_REF_WORDS,
        )
        try:
            cred = _sdk_credential.Credential(
                self._secret_id, self._secret_key, self._session_token or None
            )
            listener = _SoeNListener()
            recognizer = _sdk_speaking_assessment.SpeakingAssessment(
                self._appid, cred, SOE_ENGINE, listener
            )
            recognizer.set_eval_mode(eval_mode)            # 1=句子 / 2=段落（按句长自动，见 select_eval_mode）
            recognizer.set_rec_mode(SOE_REC_MODE)          # 1 = 录音模式
            recognizer.set_ref_text(ref_text)              # 句级 ≤30 词
            recognizer.set_text_mode(SOE_TEXT_MODE)        # 0 = 普通文本
            recognizer.set_voice_format(voice_format_int)  # 2=mp3 / 1=wav

            recognizer.start()
            try:
                recognizer.write(audio_bytes)
            finally:
                recognizer.stop()

            if listener.event.wait(EVALUATE_TIMEOUT):
                if listener.result is not None:
                    logger.info(
                        "[speech_eval] 评测成功 size=%dB ref_text=%r",
                        len(audio_bytes), ref_text[:40],
                    )
                    return listener.result, None
                logger.warning("[speech_eval] 评测失败: %s", listener.fail_reason)
                return None, "eval_failed"
            logger.error("[speech_eval] 评测超时（%ss）", EVALUATE_TIMEOUT)
            return None, "timeout"
        except Exception as e:  # noqa: BLE001 — 任何 SDK/网络异常都降级为 None（路由层回退旧链路）
            logger.error(
                "[speech_eval] SOE-N 调用异常: %s（appid=%s, voice_format=%s, ref_words=%d, "
                "session_token=%s）",
                e,
                self._appid,
                voice_format,
                len(ref_text.split()),
                "已配置" if self._session_token else "无",
                exc_info=True,
            )
            return None, "call_error"


def _pick(raw: dict, key: str):
    """兼容 SOE-N 响应两种形态：顶层扁平字段 或 result 子对象内嵌。"""
    if key in raw:
        return raw[key]
    result = raw.get("result")
    if isinstance(result, dict) and key in result:
        return result[key]
    return None


def normalize_soe_result(raw: dict) -> dict:
    """把 SOE-N 原始 JSON 归一为 api-contract §3.4 / data-model §4.9 parsed 口径。

    **四个数值字段的官方量纲并不统一**（F1-3 定标实测为准，勿凭字段名猜）：
    - accuracy：0~100（SOE-N `PronAccuracy` 原值）
    - suggested_score：0~100（SOE-N `SuggestedScore` 原值）
    - fluency：0~100（**= `PronFluency` × 100**，官方原值 0~1）
    - completion：0~100（**= `PronCompletion` × 100**，官方原值 0~1）
    - words[].match_tag：0=命中 / 2=未命中（SOE-N 原始语义，前端词级高亮）

    定标铁证（`03-change/proposals/2026-08-17-F1-3-定标音频规范.md` §2，同链路同 ref_text 仅换音频）：
    干净音频 → `PronAccuracy 98.46` / `PronFluency 0.986` / `PronCompletion 1.0`；
    低质音频 → 0.042 / 0.986 / 0.444。⇒ accuracy 是 0~100，而 fluency 与 completion 都是 0~1。

    ⚠️ 2026-09-22 修正：此前**只对 fluency 做了 ×100，completion 原值透传**，导致
    `completion` 对外恒为 0~1（干净跟读显示「完整度 1」），并连带 `evaluate_speech` 的
    `anomaly = completion < 10` 恒为真、`confidence` 恒取最低档（见 evaluation_engine.evaluate_speech）。
    """
    pron_accuracy = _pick(raw, "PronAccuracy")
    pron_fluency = _pick(raw, "PronFluency")
    pron_completion = _pick(raw, "PronCompletion")
    suggested_score = _pick(raw, "SuggestedScore")

    # PronFluency / PronCompletion 官方量纲为 0~1，统一 ×100 归一到 0~100（与 accuracy 同口径）
    def _scale01(value):
        if isinstance(value, (int, float)):
            return float(value) * 100.0
        return None

    fluency = _scale01(pron_fluency)
    completion = _scale01(pron_completion)

    words = _pick(raw, "Words")
    normalized_words = []
    if isinstance(words, list):
        for w in words:
            if not isinstance(w, dict):
                continue
            word = w.get("Word")
            if not word:
                continue
            normalized_words.append(
                {"word": word, "match_tag": int(w.get("MatchTag", 0) or 0)}
            )

    return {
        "accuracy": float(pron_accuracy or 0.0),
        "fluency": fluency if fluency is not None else 0.0,
        "completion": completion if completion is not None else 0.0,
        "suggested_score": float(suggested_score or 0.0),
        "words": normalized_words,
    }


# 模块级单例（与 asr.py get_asr_service 注入风格一致）
_speech_provider: TencentSoeNProvider | None = None


def get_speech_provider() -> SpeechProvider:
    global _speech_provider
    if _speech_provider is None:
        _speech_provider = TencentSoeNProvider()
    return _speech_provider
