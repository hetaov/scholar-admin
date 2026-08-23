"""英语语句管理服务包（scholar-admin /english 模块）

承载英语教材语句数据后台管理（SOP §5 E1.1 / E1.2 + M3 G1.3）：
- E1.1 sentence CRUD 4 服务函数（list/create/edit/delete + 级联清理）→ sentence_management.py
- E1.2 stats/chapter_tree/validate 3 服务函数 → validation.py
- M3 G1.3 sentence_group CRUD 4 服务函数（list/create/edit/delete + 审计）→ sentence_group.py

路由统一挂载在 services/routes_english.py（prefix="/english"）。
规格：service-contract.md §8（§8.1/§8.2 + §8.5）；
契约：api-contract.md §3.11 E-API-1~E-API-11。
"""
from __future__ import annotations


# ===========================================================================
# E1.1/E1.2 管理端错误码（service-contract §8.2 异常层级对齐）
# ===========================================================================

ERR_TEXTBOOK_NOT_FOUND = "TEXTBOOK_NOT_FOUND"            # 教材不存在（404）
ERR_LESSON_NOT_FOUND = "LESSON_NOT_FOUND"                # 课时不存在（404）
ERR_SENTENCE_NOT_FOUND = "SENTENCE_NOT_FOUND"            # 语句不存在（404）
ERR_SENTENCE_PAYLOAD_INVALID = "SENTENCE_PAYLOAD_INVALID"  # 入参校验失败（400）
ERR_CONFIRM_TEXT_MISMATCH = "CONFIRM_TEXT_MISMATCH"      # 删除二次确认失败（400）
ERR_GROUP_NOT_FOUND = "GROUP_NOT_FOUND"                  # 语句分组不存在（404，M3 G1.3）


class EnglishManagementError(ValueError):
    """英语语句管理异常基类（routes 层 _english_error_to_http 统一映射）。"""

    code = "ENGLISH_MANAGEMENT_ERROR"

    def __init__(self, message: str, code: str | None = None):
        code = code or self.code
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class TextbookNotFoundError(LookupError):
    """英语教材不存在（routes 层映射 → HTTP 404）。"""

    code = ERR_TEXTBOOK_NOT_FOUND

    def __init__(self, textbook_id: str, message: str | None = None):
        msg = message or f"教材不存在：{textbook_id!r}"
        super().__init__(f"{self.code}: {msg}")
        self.textbook_id = textbook_id
        self.message = msg


class LessonNotFoundError(LookupError):
    """课时不存在（routes 层映射 → HTTP 404）。"""

    code = ERR_LESSON_NOT_FOUND

    def __init__(self, lesson_id: str, message: str | None = None):
        msg = message or f"课时不存在：{lesson_id!r}"
        super().__init__(f"{self.code}: {msg}")
        self.lesson_id = lesson_id
        self.message = msg


class SentenceNotFoundError(LookupError):
    """语句不存在（routes 层映射 → HTTP 404）。"""

    code = ERR_SENTENCE_NOT_FOUND

    def __init__(self, sentence_id: str, message: str | None = None):
        msg = message or f"语句不存在：{sentence_id!r}"
        super().__init__(f"{self.code}: {msg}")
        self.sentence_id = sentence_id
        self.message = msg


class SentencePayloadError(EnglishManagementError):
    """语句入参校验异常（routes 层映射 → HTTP 400）。"""

    code = ERR_SENTENCE_PAYLOAD_INVALID


class ConfirmTextMismatchError(EnglishManagementError):
    """删除二次确认失败（routes 层映射 → HTTP 400）。"""

    code = ERR_CONFIRM_TEXT_MISMATCH


class GroupNotFoundError(LookupError):
    """语句分组不存在（M3 G1.3，routes 层映射 → HTTP 404）。"""

    code = ERR_GROUP_NOT_FOUND

    def __init__(self, group_id: str, message: str | None = None):
        msg = message or f"分组不存在：{group_id!r}"
        super().__init__(f"{self.code}: {msg}")
        self.group_id = group_id
        self.message = msg


class MissingFieldError(SentencePayloadError):
    """校验范围参数缺失（scope=chapter/lesson 缺对应 ID，routes 层映射 → HTTP 400）。

    契约 service-contract §8.1 validateEnglishSentences 异常语义：
    scope=chapter 缺 chapter_id / scope=lesson 缺 lesson_id → MissingFieldError。
    复用 SENTENCE_PAYLOAD_INVALID 错误码（400）。
    """
