"""火山引擎豆包视觉模型服务 — 英文教材图片识别与语句提取

基于 OpenAI 兼容接口调用豆包视觉模型，识别图片中的英文教学内容，
提取语句并按 JSON 结构化输出。
"""
from __future__ import annotations

import base64
import json
import re
from typing import Optional

from openai import OpenAI

from config import (
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
    VOLCANO_IMAGE_FORMATS,
    VOLCANO_MAX_IMAGE_SIZE,
    VOLCANO_VISION_MODEL,
)

_SYSTEM_PROMPT = """You are an English teaching material analyzer. Your task:

1. Identify if the image is **English learning/teaching material** (textbook, workbook, handout, exam paper, flashcard, reading passage, etc).
2. Extract **all English sentences, phrases, or paragraphs** that appear in the image. Do NOT skip any.
3. For each extracted sentence:
   - Preserve the original English text exactly as it appears
   - Provide a natural Chinese translation (简体中文)
   - Estimate the difficulty level as one of: A1, A2, B1, B2, C1, C2 (CEFR scale)
   - Identify key vocabulary words (up to 5 per sentence)

4. Output ONLY valid JSON in the exact format below. No markdown code fences, no extra text.

Output JSON format:
{
  "language": "en",
  "material_type": "one of: textbook, workbook, handout, exam_paper, flashcard, reading, vocabulary, grammar, other",
  "title": "extracted or inferred title of the material, or empty string",
  "sentences": [
    {
      "index": 1,
      "text": "original English sentence exactly as shown",
      "translation": "natural Chinese translation",
      "level": "CEFR level (A1/A2/B1/B2/C1/C2)",
      "keywords": ["word1", "word2", "word3"]
    }
  ],
  "total_sentences": 0,
  "summary": "one-sentence summary in Chinese describing the image content"
}

If the image is NOT English learning material, return:
{
  "language": null,
  "material_type": "not_english_material",
  "title": "",
  "sentences": [],
  "total_sentences": 0,
  "summary": "Why this is not English learning material."
}"""


class VolcanoVisionService:
    """火山引擎豆包视觉识别服务"""

    def __init__(
        self,
        api_key: str = VOLCANO_API_KEY,
        base_url: str = VOLCANO_BASE_URL,
        model: str = VOLCANO_VISION_MODEL,
    ):
        if not api_key:
            raise ValueError(
                "缺少火山方舟 API Key。请设置环境变量：\n"
                "  export VOLCANO_API_KEY='你的API Key'\n"
                "获取地址：https://console.volcengine.com/ark"
            )
        if not model:
            raise ValueError(
                "缺少火山方舟推理接入点 ID (Endpoint ID)。\n"
                "请按以下步骤创建：\n"
                "  1. 打开 https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint\n"
                "  2. 点击「创建推理接入点」\n"
                "  3. 选择模型 doubao-1.5-vision-pro-32k\n"
                "  4. 创建完成后复制 Endpoint ID（格式：ep-2024xxxxxxxx）\n"
                "  5. 设置环境变量：export VOLCANO_VISION_MODEL='ep-2024xxxxxxxx'\n"
                '注意：model 参数必须填 Endpoint ID，不能填模型名称 "doubao-1.5-vision-pro-32k"'
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_image_size = VOLCANO_MAX_IMAGE_SIZE
        self.valid_formats = VOLCANO_IMAGE_FORMATS

    # ==================== 图片校验 ====================

    @staticmethod
    def validate_image(
        image_bytes: bytes | None = None,
        image_url: str | None = None,
    ) -> None:
        """校验图片输入"""
        if not image_bytes and not image_url:
            raise ValueError("必须提供 image_bytes（本地图片）或 image_url（在线图片）")

        if image_url and image_bytes:
            raise ValueError("只需提供 image_bytes 或 image_url 之一")

    @staticmethod
    def _encode_image(image_bytes: bytes) -> str:
        """将图片 bytes 编码为 base64 data URL"""
        # 简单检测图片格式
        fmt = "jpeg"
        if image_bytes[:4] == b"\x89PNG":
            fmt = "png"
        elif image_bytes[:2] == b"\xff\xd8":
            fmt = "jpeg"
        elif image_bytes[:4] == b"RIFF":
            fmt = "webp"
        elif image_bytes[:2] in (b"BM",):
            fmt = "bmp"

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/{fmt};base64,{b64}"

    # ==================== 图片识别 ====================

    def recognize(
        self,
        image_bytes: bytes | None = None,
        image_url: str | None = None,
    ) -> dict:
        """识别图片中的英文教材内容

        Args:
            image_bytes: 图片二进制数据（本地上传）
            image_url:   图片在线地址

        Returns:
            JSON 结构化的识别结果
        """
        self.validate_image(image_bytes=image_bytes, image_url=image_url)

        # 构建图片 content
        if image_bytes:
            if len(image_bytes) > self.max_image_size:
                raise ValueError(
                    f"图片大小 {len(image_bytes)} 超过限制 {self.max_image_size} 字节"
                )
            image_content = {
                "type": "image_url",
                "image_url": {"url": self._encode_image(image_bytes)},
            }
        else:
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_url},
            }

        # 调用豆包视觉模型
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please analyze this image and extract all English learning content."},
                        image_content,
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=4096,
        )

        raw_text = response.choices[0].message.content or ""

        # 解析模型返回的 JSON
        return self._parse_response(raw_text)

    # ==================== 响应解析 ====================

    @staticmethod
    def _parse_response(raw_text: str) -> dict:
        """从模型返回的文本中提取并验证 JSON"""
        # 尝试直接解析
        try:
            result = json.loads(raw_text)
            return VolcanoVisionService._normalize(result)
        except json.JSONDecodeError:
            pass

        # 尝试提取 markdown 代码块中的 JSON
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
        if code_block:
            try:
                result = json.loads(code_block.group(1))
                return VolcanoVisionService._normalize(result)
            except json.JSONDecodeError:
                pass

        # 尝试提取第一个 { ... } 块
        brace_match = re.search(r"\{[\s\S]*\}", raw_text)
        if brace_match:
            try:
                result = json.loads(brace_match.group(0))
                return VolcanoVisionService._normalize(result)
            except json.JSONDecodeError:
                pass

        # 解析失败，返回原始文本
        return {
            "language": None,
            "material_type": "parse_error",
            "title": "",
            "sentences": [],
            "total_sentences": 0,
            "summary": raw_text,
            "error": "Failed to parse model response as JSON",
        }

    @staticmethod
    def _normalize(result: dict) -> dict:
        """标准化输出字段"""
        # 兼容模型返回 JSON 数组的情况（取第一个元素）
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            return {
                "language": None,
                "material_type": "parse_error",
                "title": "",
                "sentences": [],
                "total_sentences": 0,
                "summary": str(result),
                "error": f"Expected JSON object but got {type(result).__name__}",
            }
        result.setdefault("language", "en")
        result.setdefault("material_type", "other")
        result.setdefault("title", "")
        result.setdefault("summary", "")
        sentences = result.get("sentences", [])
        if isinstance(sentences, list):
            for i, s in enumerate(sentences):
                s.setdefault("index", i + 1)
                s.setdefault("text", "")
                s.setdefault("translation", "")
                s.setdefault("level", "")
                s.setdefault("keywords", [])
            result["total_sentences"] = len(sentences)
        else:
            result["sentences"] = []
            result["total_sentences"] = 0
        return result
