"""G0.1 单元测试 — textbook subject_type 多学科扩展 + getter 兼容层

覆盖 SOP 验收标准 5 条：
  1. 存量 textbook（无 subject_type 字段）→ getter 返回 subject_type='english'
  2. build_textbook_v2_doc 默认写入 subject_type='english'（英语 0 迁移）
  3. 常量齐全：SUBJECT_TYPE_ENGLISH/MATH/CHINESE + VALID_SUBJECT_TYPES
  4. POST 数学教材缺 semester → 抛 MATH_TEXTBOOK_SEMESTER_REQUIRED 异常（400）
  5. POST subject_type 非法 → 抛 INVALID_SUBJECT_TYPE 异常（400）
  6. POST 数学教材合法（subject_type=math + semester + grade）→ validate 通过，输出 cleaned payload
"""
from __future__ import annotations

import pytest


# ===========================================================================
# 1. 常量存在性 + 取值
# ===========================================================================


class TestSubjectTypeConstants:
    def test_subject_type_constants_exist_and_have_correct_values(self):
        """SUBJECT_TYPE_* 三常量存在，取值 = english/math/chinese。"""
        from services.math import (
            SUBJECT_TYPE_CHINESE,
            SUBJECT_TYPE_ENGLISH,
            SUBJECT_TYPE_MATH,
            VALID_SUBJECT_TYPES,
        )

        assert SUBJECT_TYPE_ENGLISH == "english"
        assert SUBJECT_TYPE_MATH == "math"
        assert SUBJECT_TYPE_CHINESE == "chinese"
        # VALID_SUBJECT_TYPES 为集合或元组/列表的等价可迭代容器
        assert set(VALID_SUBJECT_TYPES) == {"english", "math", "chinese"}

    def test_default_textbook_subject_is_english(self):
        """契约 §4.1：缺省 subject_type = english。"""
        from services.math import DEFAULT_SUBJECT_TYPE, SUBJECT_TYPE_ENGLISH

        assert DEFAULT_SUBJECT_TYPE == SUBJECT_TYPE_ENGLISH


# ===========================================================================
# 2. DB getter 兼容层 — 存量记录（无 subject_type 字段）自动注入 english
# ===========================================================================


class TestNormalizeTextbookDocGetterCompat:
    """normalize_textbook_doc 是读侧兼容入口：对 textbook_v2 单条记录打标。"""

    def test_missing_subject_type_defaults_to_english(self):
        """存量老数据（完全无 subject_type）→ 返回 subject_type='english'。"""
        from services.models_content import normalize_textbook_doc

        legacy = {
            "_id": "nce1",
            "textbook_id": "nce1",
            "title": "新概念第一册",
            "grade": "",
            "level": "Book 1",
            "version": 1,
        }
        normalized = normalize_textbook_doc(legacy)
        assert normalized["subject_type"] == "english"

    def test_subject_type_none_defaults_to_english(self):
        """明确 subject_type=None（非 missing 但空）→ 返回 english。"""
        from services.models_content import normalize_textbook_doc

        doc = {
            "textbook_id": "rjsx_4_up",
            "title": "人教版四年级上册",
            "subject_type": None,
        }
        normalized = normalize_textbook_doc(doc)
        assert normalized["subject_type"] == "english"

    def test_subject_type_explicit_keeps_value_math(self):
        """显式传 math 保留原值不覆盖（写入合法 math）。"""
        from services.models_content import normalize_textbook_doc

        doc = {
            "textbook_id": "rjsx_3_up",
            "title": "人教版三年级上册",
            "grade": "三年级",
            "subject_type": "math",
            "semester": "up",
            "publisher": "人民教育出版社",
        }
        normalized = normalize_textbook_doc(doc)
        assert normalized["subject_type"] == "math"
        assert normalized["semester"] == "up"
        assert normalized["publisher"] == "人民教育出版社"

    def test_subject_type_explicit_keeps_value_chinese(self):
        """显式传 chinese 保留原值（留作未来学科扩展）。"""
        from services.models_content import normalize_textbook_doc

        normalized = normalize_textbook_doc({"textbook_id": "yw1", "subject_type": "chinese"})
        assert normalized["subject_type"] == "chinese"

    def test_normalize_returns_copy_does_not_mutate_input(self):
        """兼容层不应修改传入对象（避免副作用污染调用方原始对象）。"""
        from services.models_content import normalize_textbook_doc

        legacy = {"textbook_id": "nce1", "title": "NCE1"}
        legacy_keys_before = set(legacy.keys())
        out = normalize_textbook_doc(legacy)
        assert out is not legacy
        # 原对象未被加入 subject_type
        assert set(legacy.keys()) == legacy_keys_before
        assert "subject_type" not in legacy


# ===========================================================================
# 3. 写入侧 build_textbook_v2_doc 扩展（英语默认 subject_type=english）
# ===========================================================================


class TestBuildTextbookV2DocSubjectType:
    def test_default_subject_type_english(self):
        """原函数不传 subject_type → 默认写入 english（英语 0 迁移）。"""
        from services.models_content import build_textbook_v2_doc

        doc = build_textbook_v2_doc(
            "nce1", "NCE", grade="1", level="Book 1", chapter_count=2, now=1000,
        )
        assert doc["subject_type"] == "english"
        # 原字段不变（向后兼容零破坏）
        assert doc["_id"] == "nce1"
        assert doc["version"] == 1
        assert doc["chapter_count"] == 2

    def test_explicit_subject_type_math_includes_semester(self):
        """显式传 math + semester（写入时由调用方负责，build_doc 不校验 — 校验是 validate_* 职责）。"""
        from services.models_content import build_textbook_v2_doc

        doc = build_textbook_v2_doc(
            "rjsx_3_up", "人教版三年级上册", grade="三年级", level="",
            subject_type="math", semester="up", publisher="人民教育出版社", isbn="9787107000001", now=1000,
        )
        assert doc["subject_type"] == "math"
        assert doc["semester"] == "up"
        assert doc["publisher"] == "人民教育出版社"
        assert doc["isbn"] == "9787107000001"
        # 保留既有 counter 字段（默认 0，向后兼容）
        assert doc["chapter_count"] == 0


# ===========================================================================
# 4. 入参校验 validate_textbook_payload —— math 必填 / subject_type 合法
# ===========================================================================


class TestValidateTextbookPayload:
    """校验函数：返回 cleaned dict，失败抛 ValueError 含契约错误码。"""

    def test_subject_type_invalid_raises_with_error_code(self):
        """subject_type ∉ {english,math,chinese} → ValueError，code=INVALID_SUBJECT_TYPE。"""
        from services.models_content import validate_textbook_payload

        with pytest.raises(ValueError) as exc_info:
            validate_textbook_payload(title="x", grade="三", subject_type="physics")
        err = exc_info.value
        assert hasattr(err, "code") or "INVALID_SUBJECT_TYPE" in str(err)
        # 两种表达方式都可接受（契约要求 400 + code）
        assert "INVALID_SUBJECT_TYPE" in (
            getattr(err, "code", "") + str(err)
        )

    def test_math_missing_semester_raises_math_semester_required(self):
        """subject_type=math 且缺 semester → ValueError，code=MATH_TEXTBOOK_SEMESTER_REQUIRED。"""
        from services.models_content import validate_textbook_payload

        with pytest.raises(ValueError) as exc_info:
            validate_textbook_payload(
                title="人教版三年级上册", grade="三年级", subject_type="math",
            )
        assert "MATH_TEXTBOOK_SEMESTER_REQUIRED" in (
            getattr(exc_info.value, "code", "") + str(exc_info.value)
        )

    def test_math_invalid_semester_raises(self):
        """subject_type=math 且 semester 非法（非 up/down）→ ValueError。"""
        from services.models_content import validate_textbook_payload

        with pytest.raises(ValueError) as exc_info:
            validate_textbook_payload(
                title="x", grade="三", subject_type="math", semester="spring",
            )
        # 错误内容包含 semester 取值范围提示
        assert "up" in str(exc_info.value) or "down" in str(exc_info.value) or "semester" in str(exc_info.value).lower()

    def test_math_valid_payload_returns_cleaned_and_defaults_applied(self):
        """数学教材合法 → 返回 cleaned dict，缺省 subject_type/math 字段齐全。"""
        from services.models_content import validate_textbook_payload

        result = validate_textbook_payload(
            title="人教版四年级上册",
            grade="四年级",
            subject_type="math",
            semester="up",
            publisher="人民教育出版社",
        )
        assert result["subject_type"] == "math"
        assert result["semester"] == "up"
        assert result["grade"] == "四年级"
        assert result["publisher"] == "人民教育出版社"
        # isbn/cover_url 未传 → 不出现在结果中（或 None；契约为可选）
        assert result.get("isbn") in (None, "no-key") or "isbn" not in result

    def test_english_valid_payload_default_subject_type_applied(self):
        """英语不传 subject_type → 默认 english，不要求 semester（无错）。"""
        from services.models_content import validate_textbook_payload

        result = validate_textbook_payload(title="NCE Book 2", grade="2", level="Book 2")
        assert result["subject_type"] == "english"
        # 英语侧 semester 为空/无 key（允许）
        assert result.get("semester") in (None, "no-key") or "semester" not in result

    def test_subject_type_math_explicit_down_valid(self):
        """math + semester=down（下册）正确通过。"""
        from services.models_content import validate_textbook_payload

        result = validate_textbook_payload(
            title="人教版六年级下册", grade="六年级", subject_type="math", semester="down",
        )
        assert result["subject_type"] == "math"
        assert result["semester"] == "down"
