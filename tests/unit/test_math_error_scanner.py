"""单元测试：错题扫描归类 services/math/error_scanner（F4.3 + F4.4）

F4.3 覆盖：
- 前置校验：scan_id 不存在 → ScanNotFoundError；OCR 未完成 → OcrNotReadyError
- 幂等：classify_status ∈ {success, needs_review} 且非 force_reclassify
  → 直接返回已有结果（Judge 不被调用）
- 高置信 + 知识点匹配 → 写 error_record + classify_status=success
- 低置信 / 知识点未匹配 → 不写 error_record + classify_status=needs_review
- Judge 未配置 → JudgeNotConfiguredError + classify_status=failed + 失败审计
- Judge 响应解析失败 → JudgeResponseError + classify_status=failed + 失败审计
- force_reclassify=true → 忽略已有结果，重新调用 Judge

F4.4 覆盖：
- 前置校验：scan_id 不存在 → ScanNotFoundError；items 为空 → ScanCorrectValidationError
- 已有 error_record_id → 更新归类（classify_method=manual_corrected）
- 无 error_record_id → 新建 error_record（classify_method=manual_corrected, source=manual_corrected）
- 修正后 classify_status 回写 success（契约五态无 corrected）
- 写审计 scan_correct
- error_type 非法 → ScanCorrectValidationError
"""
from __future__ import annotations

import pytest

from services.audit import (
    AUDIT_ACTION_SCAN_CLASSIFY,
    AUDIT_ACTION_SCAN_CORRECT,
    AUDIT_LOG_COLLECTION,
)
from services.database import (
    CURRICULUM_NODE_COLLECTION,
    ERROR_RECORD_COLLECTION,
    MATH_SCAN_UPLOAD_COLLECTION,
)
from services.math import error_scanner
from services.math.error_scanner import (
    CLASSIFY_METHOD_MANUAL_CORRECTED,
    CLASSIFY_STATUS_CLASSIFYING,
    CLASSIFY_STATUS_FAILED,
    CLASSIFY_STATUS_NEEDS_REVIEW,
    CLASSIFY_STATUS_PENDING,
    CLASSIFY_STATUS_SUCCESS,
    OCR_STATUS_PENDING,
    OCR_STATUS_SUCCESS,
    SOURCE_MANUAL_CORRECTED,
    JudgeNotConfiguredError,
    JudgeResponseError,
    OcrNotReadyError,
    ScanCorrectValidationError,
    ScanNotFoundError,
    classify_scan_upload,
    correct_scan_classify,
)
from tests.fakes.fake_db import FakeDB

SCAN_ID = "scan_test_001"
SCHOLAR_ID = "scholar_test_001"


# ---------------------------------------------------------------------------
# 种子数据
# ---------------------------------------------------------------------------


def _seed_scan(
    fake_db: FakeDB,
    *,
    scan_id: str = SCAN_ID,
    ocr_status: str = OCR_STATUS_SUCCESS,
    classify_status: str = CLASSIFY_STATUS_PENDING,
    classify_result: list | None = None,
    ocr_text: str = "1. 计算 3+5=___\n2. 下列哪个是质数？",
    **overrides,
) -> dict:
    doc = {
        "scan_id": scan_id,
        "scholar_id": SCHOLAR_ID,
        "image_url": "https://example.com/scan.jpg",
        "image_file_id": "cloud://test/scan.jpg",
        "image_hash": "abc123",
        "ocr_status": ocr_status,
        "ocr_text": ocr_text,
        "ocr_blocks": [
            {"block_id": "blk_0001", "text": "1. 计算 3+5=___"},
            {"block_id": "blk_0002", "text": "2. 下列哪个是质数？"},
        ],
        "classify_status": classify_status,
        "classify_result": classify_result or [],
        "note": "",
        "created_at": 1700000000000,
        "completed_at": None,
        "audit_log_id": "",
    }
    doc.update(overrides)
    fake_db.add(MATH_SCAN_UPLOAD_COLLECTION, doc)
    return doc


def _seed_knowledge_nodes(fake_db: FakeDB) -> None:
    """种子 curriculum_node（含 ai_summary.knowledge_points）"""
    fake_db.add(
        CURRICULUM_NODE_COLLECTION,
        {
            "node_id": "n1",
            "code": "u1",
            "grade": "3",
            "textbook_id": "tb1",
            "ai_summary": {
                "status": "success",
                "knowledge_points": [
                    {"name": "加法运算"},
                    {"name": "质数与合数"},
                ],
            },
        },
    )


def _judge_result_high_confidence() -> dict:
    """高置信 Judge 输出（知识点匹配 + confidence >= 0.6）"""
    return {
        "items": [
            {
                "knowledge_point_name": "加法运算",
                "error_type": "computation",
                "confidence": 0.9,
                "ocr_block_id": "blk_0001",
            },
            {
                "knowledge_point_name": "质数与合数",
                "error_type": "concept",
                "confidence": 0.85,
                "ocr_block_id": "blk_0002",
            },
        ]
    }


def _judge_result_low_confidence() -> dict:
    """低置信 Judge 输出（confidence < 0.6 或知识点无法定位）"""
    return {
        "items": [
            {
                "knowledge_point_name": "加法运算",
                "error_type": "computation",
                "confidence": 0.9,
                "ocr_block_id": "blk_0001",
            },
            {
                "knowledge_point_name": "未知知识点",
                "error_type": "concept",
                "confidence": 0.3,
                "ocr_block_id": "blk_0002",
            },
        ]
    }


# ---------------------------------------------------------------------------
# 前置校验
# ---------------------------------------------------------------------------


class TestClassifyPreChecks:
    @pytest.mark.asyncio
    async def test_scan_not_found_raises(self):
        """scan_id 不存在 → ScanNotFoundError"""
        db = FakeDB()
        with pytest.raises(ScanNotFoundError):
            await classify_scan_upload(db, scan_id="no_such_scan", actor="u1")

    @pytest.mark.asyncio
    async def test_ocr_not_ready_raises(self):
        """OCR 未完成（status=pending）→ OcrNotReadyError"""
        db = FakeDB()
        _seed_scan(db, ocr_status=OCR_STATUS_PENDING)
        with pytest.raises(OcrNotReadyError):
            await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")


# ---------------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------------


class TestClassifyIdempotency:
    @pytest.mark.asyncio
    async def test_already_success_returns_existing(self, monkeypatch):
        """classify_status=success 且非 force → 直接返回，Judge 不调用"""
        db = FakeDB()
        existing_items = [
            {
                "error_record_id": "er_existing",
                "knowledge_point_name": "加法运算",
                "error_type": "computation",
                "confidence": 0.9,
                "ocr_block_id": "blk_0001",
            }
        ]
        _seed_scan(
            db,
            classify_status=CLASSIFY_STATUS_SUCCESS,
            classify_result=existing_items,
        )
        calls: list = []

        async def fake_judge(ocr_text, candidates):
            calls.append(1)
            return _judge_result_high_confidence()

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert len(calls) == 0  # Judge 未被调用
        assert result["scan_id"] == SCAN_ID
        assert result["status"] == "success"
        assert result["items"] == existing_items

    @pytest.mark.asyncio
    async def test_already_needs_review_returns_existing(self, monkeypatch):
        """classify_status=needs_review 且非 force → 直接返回"""
        db = FakeDB()
        _seed_scan(
            db,
            classify_status=CLASSIFY_STATUS_NEEDS_REVIEW,
            classify_result=[],
        )
        calls: list = []

        async def fake_judge(ocr_text, candidates):
            calls.append(1)
            return _judge_result_high_confidence()

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert len(calls) == 0
        assert result["status"] == "needs_review"

    @pytest.mark.asyncio
    async def test_force_reclassify_re_invokes_judge(self, monkeypatch):
        """force_reclassify=true → 忽略已有结果，重新调用 Judge"""
        db = FakeDB()
        _seed_scan(
            db,
            classify_status=CLASSIFY_STATUS_SUCCESS,
            classify_result=[],
        )
        _seed_knowledge_nodes(db)
        calls: list = []

        async def fake_judge(ocr_text, candidates):
            calls.append(1)
            return _judge_result_high_confidence()

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(
            db, scan_id=SCAN_ID, force_reclassify=True, actor="u1"
        )
        assert len(calls) == 1
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# 置信度门控
# ---------------------------------------------------------------------------


class TestClassifyConfidenceGating:
    @pytest.mark.asyncio
    async def test_high_confidence_writes_error_record(self, monkeypatch):
        """高置信 + 知识点匹配 → 写 error_record + classify_status=success"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            return _judge_result_high_confidence()

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert result["status"] == "success"
        assert len(result["items"]) == 2
        # 每道题都写了 error_record
        for item in result["items"]:
            assert item["error_record_id"]

        # error_record 落库 2 条
        records = db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 2
        assert records[0]["classify_method"] == "auto_scan"
        assert records[0]["source"] == "auto_scan"
        assert records[0]["scan_upload_id"] == SCAN_ID

        # scan 记录 classify_status 更新为 success
        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == CLASSIFY_STATUS_SUCCESS

    @pytest.mark.asyncio
    async def test_high_confidence_persists_question_text(self, monkeypatch):
        """B1：Judge 输出 question_text → error_record 落库 + classify_result/出参透传"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            return {
                "items": [
                    {
                        "knowledge_point_name": "加法运算",
                        "error_type": "computation",
                        "confidence": 0.9,
                        "ocr_block_id": "blk_0001",
                        "question_text": "1. 计算 3+5=___",
                    },
                    {
                        "knowledge_point_name": "质数与合数",
                        "error_type": "concept",
                        "confidence": 0.85,
                        "ocr_block_id": "blk_0002",
                        "question_text": "2. 下列哪个是质数？",
                    },
                ]
            }

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert result["status"] == "success"
        # 出参透传 question_text
        assert [i["question_text"] for i in result["items"]] == [
            "1. 计算 3+5=___",
            "2. 下列哪个是质数？",
        ]
        # error_record 落库含 question_text（含直落 needs_review 中高置信项）
        records = db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 2
        assert all(r.get("question_text") for r in records)
        # scan.classify_result 同样落库 → 幂等返回可见题干锚点
        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert [i.get("question_text") for i in scans[0]["classify_result"]] == [
            "1. 计算 3+5=___",
            "2. 下列哪个是质数？",
        ]

    @pytest.mark.asyncio
    async def test_low_confidence_skips_error_record(self, monkeypatch):
        """低置信 / 知识点未匹配 → 不写 error_record + classify_status=needs_review"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            return _judge_result_low_confidence()

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert result["status"] == "needs_review"
        assert len(result["items"]) == 2
        # 第一道高置信 → 写了 error_record；第二道低置信 → 未写
        assert result["items"][0]["error_record_id"]
        assert not result["items"][1]["error_record_id"]

        # error_record 仅落库 1 条（高置信项）
        records = db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 1

        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == CLASSIFY_STATUS_NEEDS_REVIEW

    @pytest.mark.asyncio
    async def test_empty_items_results_in_needs_review(self, monkeypatch):
        """Judge 返回空 items → needs_review（无 error_record 写入）"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            return {"items": []}

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert result["status"] == "needs_review"
        assert result["items"] == []
        assert len(db.all(ERROR_RECORD_COLLECTION)) == 0

    @pytest.mark.asyncio
    async def test_high_confidence_out_of_candidate_creates_extra_ai(self, monkeypatch):
        """B1.5：未命中候选但高置信（单位换算）→ EXTRA_AI 节点新建 + 落库锚定"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            return {
                "items": [
                    {
                        "knowledge_point_name": "单位换算",
                        "error_type": "concept",
                        "confidence": 0.9,
                        "ocr_block_id": "blk_0001",
                        "question_text": "3.5 吨 = ___ 千克",
                        "candidate_hits": ["质数与合数"],
                    }
                ]
            }

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert result["status"] == "success"
        item = result["items"][0]
        assert item["error_record_id"]
        # 图谱外新建分支：new_kp_name 与 knowledge_point_name 同值（契约 §1.2b）
        assert item["new_kp_name"] == "单位换算"
        # candidate_hits：Judge 给的疑似正式教材候选结构化透出（仅响应，不持久化）
        assert item["candidate_hits"] == [
            {"textbook_id": "tb1", "grade": "3", "kp_name": "质数与合数"}
        ]

        # EXTRA_AI 虚拟教材 + 图谱外节点已幂等创建
        tbs = db.all("textbook_v2")
        assert any(t.get("textbook_id") == "EXTRA_AI" for t in tbs)
        extra_nodes = [
            n for n in db.all(CURRICULUM_NODE_COLLECTION)
            if n.get("textbook_id") == "EXTRA_AI"
        ]
        assert len(extra_nodes) == 1
        assert extra_nodes[0]["title"] == "单位换算"
        assert extra_nodes[0]["unit_title"] == "未分类"

        # error_record 锚定 EXTRA_AI（node_code 非空、textbook_id=EXTRA_AI、
        # 链锚点 + drill_stats/last_drill_result 初始化）
        records = db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 1
        rec = records[0]
        assert rec["textbook_id"] == "EXTRA_AI"
        assert rec["node_code"].startswith("xai_")
        assert rec["knowledge_point_name"] == "单位换算"
        assert rec["drill_stats"] == {}
        assert rec["last_drill_result"] == {}

    @pytest.mark.asyncio
    async def test_extra_ai_node_reused_across_scans(self, monkeypatch):
        """B1.5 幂等：不同 scan 同一图谱外知识点 → 节点复用不重复建"""
        db = FakeDB()
        _seed_scan(db)
        _seed_scan(db, scan_id="scan_test_002")
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            return {
                "items": [
                    {
                        # B1.5a 后「加法简便运算」会就近改名挂正式候选「加法运算」，
                        # 不建 EXTRA_AI；此处用与候选无交集/无操作符冲突的真正图谱外名测复用
                        "knowledge_point_name": "单位换算",
                        "error_type": "method",
                        "confidence": 0.85,
                        "ocr_block_id": "",
                        "question_text": "3.5 吨 = ___ 千克",
                    }
                ]
            }

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        r1 = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        r2 = await classify_scan_upload(db, scan_id="scan_test_002", actor="u1")
        assert r1["status"] == r2["status"] == "success"
        # 两个 scan 各 1 条 error_record，但 EXTRA_AI 节点只 1 个（title 幂等复用）
        assert len(db.all(ERROR_RECORD_COLLECTION)) == 2
        extra_nodes = [
            n for n in db.all(CURRICULUM_NODE_COLLECTION)
            if n.get("textbook_id") == "EXTRA_AI"
        ]
        assert len(extra_nodes) == 1
        assert extra_nodes[0]["title"] == "单位换算"
        # 两条记录共享同一 node_code
        codes = {r["node_code"] for r in db.all(ERROR_RECORD_COLLECTION)}
        assert len(codes) == 1

    @pytest.mark.asyncio
    async def test_low_confidence_out_of_candidate_stays_review(self, monkeypatch):
        """B1.5：未命中候选且低置信 → 仍 needs_review，不建 EXTRA_AI 节点"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            return {
                "items": [
                    {
                        "knowledge_point_name": "单位换算",
                        "error_type": "concept",
                        "confidence": 0.4,
                        "ocr_block_id": "",
                        "question_text": "3.5 吨 = ___ 千克",
                    }
                ]
            }

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert result["status"] == "needs_review"
        assert not result["items"][0]["error_record_id"]
        # 修正页新建预填名 + 疑似改挂建议透出
        assert result["items"][0]["new_kp_name"] == "单位换算"
        # 不落记录、不建 EXTRA_AI 节点
        assert len(db.all(ERROR_RECORD_COLLECTION)) == 0
        extra_nodes = [
            n for n in db.all(CURRICULUM_NODE_COLLECTION)
            if n.get("textbook_id") == "EXTRA_AI"
        ]
        assert extra_nodes == []

    def test_classify_prompt_confidence_rule_allows_extra_ai_high_conf(self):
        """B1.5 回归：classify prompt 不以候选内外限制置信度。

        曾写死「候选集无匹配或错因不明时 ≤0.5」，把候选外题干语义明确的题
        （如小数加法数位意义、单位换算）uniform 压到 0.4 → 全进 needs_review
        多走 6 次人工。B1.5 允许候选外明确题 ≥0.6 自动 EXTRA_AI 直落；
        此处保护 prompt 措辞，防旧规则回滚（LLM 端到端行为由真实 judge 调用覆盖）。
        """
        tmpl = error_scanner._CLASSIFY_USER_TEMPLATE
        assert "候选集无匹配或错因不明时 ≤0.5" not in tmpl
        assert "命中候选集且匹配明确 → ≥0.7" in tmpl
        assert "候选集外但题干完整" in tmpl and "≥0.6" in tmpl
        assert "EXTRA_AI" in tmpl

    def test_nearest_candidates_operator_conflict_filtered(self):
        """B1.5a 护栏单测：操作符互斥的近名不出现在就近结果。

        无护栏时「小数减法的实际应用」vs「小数乘除的实际应用」字符 2-gram
        Jaccard≈0.36 ≥0.25 会机械误挂（加法↔减法一字差覆辙）；护栏要求两边
        各含运算语义词但无交集 → 拒绝。共享运算词（乘法≈乘除）不受影响。
        """
        base = [
            {"kp_name": "加法运算"},
            {"kp_name": "小数乘除的实际应用"},
            {"kp_name": "小数加法的实际应用"},
        ]
        # 减法 vs 加/乘除：互斥 → 无就近候选（仍会走 EXTRA_AI 新建）
        assert error_scanner._nearest_candidates("小数减法的实际应用", base) == []
        # 乘法 vs 乘除：共享「乘」→ 允许就近（评审 manual_correct 0.8 修复目标）
        near = error_scanner._nearest_candidates("小数乘法的实际应用", base, limit=1)
        assert near and near[0]["kp_name"] == "小数乘除的实际应用"
        # 精确恒排最前；加法族共享「加」不受护栏影响（既有 correct 用例口径不回归）
        assert error_scanner._nearest_candidates("加法运算", base, limit=1)[0]["kp_name"] == "加法运算"
        assert (
            error_scanner._nearest_candidates("加法简便运算", base, limit=1)[0]["kp_name"]
            == "加法运算"
        )

    def test_nearest_candidates_template_suffix_stripped(self):
        """B1.5a 护栏 2：共享泛化模板后缀（的实际应用）不构成就近。

        真实候选集实测：无剥离时「小数加法的实际应用」会因共享「的实际应用」
        gram 误挂到「统计的实际应用」（j≈0.4，语义无关）；剥离公共尾后
        交集只反映主题词 → 统计被拒，仍走 EXTRA_AI。
        """
        base = [
            {"kp_name": "统计的实际应用"},
            {"kp_name": "小数乘除的实际应用"},
            {"kp_name": "小数减法的实际应用"},
        ]
        # 统计：仅共享「的实际应用」→ 剥离后交集空 → 不就近；
        # 乘除/减法：主题词操作符互斥 → 不就近
        assert error_scanner._nearest_candidates("小数加法的实际应用", base) == []
        # 乘除：剥后缀后主题「小数乘法/小数乘除」共享且同族 → 就近
        near = error_scanner._nearest_candidates("小数乘法的实际应用", base, limit=1)
        assert near and near[0]["kp_name"] == "小数乘除的实际应用"

    @pytest.mark.asyncio
    async def test_out_of_candidate_near_name_renames_to_formal(self, monkeypatch):
        """B1.5a：候选外高置信但语义就近（小数乘法的实际应用≈候选小数乘除的实际应用）
        → 自动修正为标准名挂 formal，不再一律 EXTRA_AI 新建（评审 manual_correct
        「图谱外恰当性」修复：应与 correct 人工分支口径一致）"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)
        db.add(
            CURRICULUM_NODE_COLLECTION,
            {
                "node_id": "n2",
                "code": "u2",
                "grade": "5",
                "textbook_id": "tb1",
                "ai_summary": {
                    "status": "success",
                    "knowledge_points": [{"name": "小数乘除的实际应用"}],
                },
            },
        )

        async def fake_judge(ocr_text, candidates):
            return {
                "items": [
                    {
                        "knowledge_point_name": "小数乘法的实际应用",
                        "error_type": "concept",
                        "confidence": 0.85,
                        "ocr_block_id": "blk_0001",
                        "question_text": "0.25×3.2 在购物总价中的实际应用",
                    }
                ]
            }

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert result["status"] == "success"
        item = result["items"][0]
        # 名修正为候选标准名、锚定正式候选；非图谱外新建（无 new_kp_name）
        assert item["knowledge_point_name"] == "小数乘除的实际应用"
        assert "new_kp_name" not in item
        # B1.5a 追踪：original_kp_name 保留 Judge 原判名（对称 EXTRA_AI 的 new_kp_name），
        # 供 eval/前端识别「自动就近改名」并核对改名目标
        assert item["original_kp_name"] == "小数乘法的实际应用"
        rec = db.all(ERROR_RECORD_COLLECTION)[0]
        assert rec["node_code"] == "u2"
        assert rec["textbook_id"] == "tb1"
        assert rec["knowledge_point_name"] == "小数乘除的实际应用"
        assert rec["original_kp_name"] == "小数乘法的实际应用"
        # 未误建 EXTRA_AI 节点
        extra_nodes = [
            n for n in db.all(CURRICULUM_NODE_COLLECTION)
            if n.get("textbook_id") == "EXTRA_AI"
        ]
        assert extra_nodes == []

    @pytest.mark.asyncio
    async def test_out_of_candidate_operator_conflict_stays_extra_ai(self, monkeypatch):
        """B1.5a 护栏行为：候选外 kp 与近候选操作符互斥（减 vs 加/乘除）
        → 不就近误挂，仍 EXTRA_AI 新建（防「加法↔减法一字差误挂」）"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)
        for nid, code, name in (
            ("n2", "u2", "小数乘除的实际应用"),
            ("n3", "u3", "小数加法的实际应用"),
        ):
            db.add(
                CURRICULUM_NODE_COLLECTION,
                {
                    "node_id": nid,
                    "code": code,
                    "grade": "5",
                    "textbook_id": "tb1",
                    "ai_summary": {
                        "status": "success",
                        "knowledge_points": [{"name": name}],
                    },
                },
            )

        async def fake_judge(ocr_text, candidates):
            return {
                "items": [
                    {
                        "knowledge_point_name": "小数减法的实际应用",
                        "error_type": "concept",
                        "confidence": 0.85,
                        "ocr_block_id": "blk_0001",
                        "question_text": "一个数减 0.5 的实际应用",
                    }
                ]
            }

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert result["status"] == "success"
        item = result["items"][0]
        # 图谱外新建分支：new_kp_name 与 knowledge_point_name 同值
        assert item["knowledge_point_name"] == "小数减法的实际应用"
        assert item["new_kp_name"] == "小数减法的实际应用"
        assert "original_kp_name" not in item  # 未就近改名 → 无 original_kp_name
        rec = db.all(ERROR_RECORD_COLLECTION)[0]
        assert rec["textbook_id"] == "EXTRA_AI"
        assert rec["node_code"].startswith("xai_")
        assert rec["knowledge_point_name"] == "小数减法的实际应用"
        assert "original_kp_name" not in rec
        # 只建了本 kp 的 EXTRA_AI 节点，未误挂任何正式候选
        extra_nodes = [
            n for n in db.all(CURRICULUM_NODE_COLLECTION)
            if n.get("textbook_id") == "EXTRA_AI"
        ]
        assert [n["title"] for n in extra_nodes] == ["小数减法的实际应用"]


# ---------------------------------------------------------------------------
# Judge 异常处理
# ---------------------------------------------------------------------------


class TestClassifyJudgeErrors:
    @pytest.mark.asyncio
    async def test_judge_not_configured_sets_failed(self, monkeypatch):
        """LLM_JUDGE_MODEL 未配置 → JudgeNotConfiguredError + classify_status=failed"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            raise JudgeNotConfiguredError("not configured")

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        with pytest.raises(JudgeNotConfiguredError):
            await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")

        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == CLASSIFY_STATUS_FAILED

        # 失败审计落库
        audit_logs = db.all(AUDIT_LOG_COLLECTION)
        assert any(
            log.get("action") == AUDIT_ACTION_SCAN_CLASSIFY
            and log.get("result") == "failed"
            for log in audit_logs
        )

    @pytest.mark.asyncio
    async def test_judge_response_error_sets_failed(self, monkeypatch):
        """Judge 响应解析失败 → JudgeResponseError + classify_status=failed"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            raise JudgeResponseError("bad json")

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        with pytest.raises(JudgeResponseError):
            await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")

        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == CLASSIFY_STATUS_FAILED


# ---------------------------------------------------------------------------
# 审计
# ---------------------------------------------------------------------------


class TestClassifyAudit:
    @pytest.mark.asyncio
    async def test_success_writes_audit(self, monkeypatch):
        """归类成功 → 写 scan_classify 审计（result=success）"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            return _judge_result_high_confidence()

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        await classify_scan_upload(db, scan_id=SCAN_ID, actor="actor_001")

        audit_logs = db.all(AUDIT_LOG_COLLECTION)
        classify_audits = [
            log for log in audit_logs if log.get("action") == AUDIT_ACTION_SCAN_CLASSIFY
        ]
        assert len(classify_audits) == 1
        assert classify_audits[0]["actor"] == "actor_001"
        assert classify_audits[0]["object_ref"] == SCAN_ID


# ===========================================================================
# F4.4 人工修正归类（correct_scan_classify）
# ===========================================================================


class TestCorrectPreChecks:
    @pytest.mark.asyncio
    async def test_scan_not_found_raises(self):
        """scan_id 不存在 → ScanNotFoundError"""
        db = FakeDB()
        with pytest.raises(ScanNotFoundError):
            await correct_scan_classify(
                db,
                scan_id="no_such_scan",
                items=[{"knowledge_point_name": "加法运算", "error_type": "computation"}],
                actor="u1",
            )

    @pytest.mark.asyncio
    async def test_empty_items_raises(self):
        """items 为空 → ScanCorrectValidationError"""
        db = FakeDB()
        _seed_scan(db)
        with pytest.raises(ScanCorrectValidationError):
            await correct_scan_classify(
                db, scan_id=SCAN_ID, items=[], actor="u1"
            )

    @pytest.mark.asyncio
    async def test_invalid_error_type_raises(self):
        """error_type 非四分类 → ScanCorrectValidationError"""
        db = FakeDB()
        _seed_scan(db)
        with pytest.raises(ScanCorrectValidationError):
            await correct_scan_classify(
                db,
                scan_id=SCAN_ID,
                items=[{"knowledge_point_name": "加法运算", "error_type": "unknown"}],
                actor="u1",
            )

    @pytest.mark.asyncio
    async def test_new_item_without_kp_or_error_raises(self):
        """无 error_record_id 且无 knowledge_point_name/error_type → 校验失败"""
        db = FakeDB()
        _seed_scan(db)
        with pytest.raises(ScanCorrectValidationError):
            await correct_scan_classify(
                db,
                scan_id=SCAN_ID,
                items=[{"raw_text_corrected": "some text"}],
                actor="u1",
            )


class TestCorrectUpdateExisting:
    @pytest.mark.asyncio
    async def test_update_existing_error_record(self):
        """已有 error_record_id → 更新归类（classify_method=manual_corrected）"""
        db = FakeDB()
        _seed_scan(db, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db)
        # 预置一条 auto_scan 归类的 error_record
        db.add(
            ERROR_RECORD_COLLECTION,
            {
                "record_id": "er_existing_001",
                "scholar_id": SCHOLAR_ID,
                "node_code": "",
                "primary_error": "computation",
                "classify_method": "auto_scan",
                "source": "auto_scan",
                "scan_upload_id": SCAN_ID,
                "confidence": 0.3,
            },
        )

        result = await correct_scan_classify(
            db,
            scan_id=SCAN_ID,
            items=[
                {
                    "error_record_id": "er_existing_001",
                    "knowledge_point_name": "加法运算",
                    "error_type": "concept",
                }
            ],
            actor="actor_001",
        )

        assert result["scan_id"] == SCAN_ID
        assert len(result["corrected"]) == 1
        assert result["corrected"][0]["error_record_id"] == "er_existing_001"
        assert result["corrected"][0]["knowledge_point_name"] == "加法运算"
        assert result["corrected"][0]["error_type"] == "concept"

        # error_record 已更新
        records = db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 1
        assert records[0]["classify_method"] == CLASSIFY_METHOD_MANUAL_CORRECTED
        assert records[0]["primary_error"] == "concept"
        assert records[0]["node_code"] == "u1"  # 匹配候选集 node_code
        assert "corrected_at" in records[0]

        # scan classify_status 回写 success
        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == CLASSIFY_STATUS_SUCCESS

    @pytest.mark.asyncio
    async def test_update_existing_persists_question_text(self):
        """B1：update 分支带 question_text → 同步落 question_text + raw_text_corrected"""
        db = FakeDB()
        _seed_scan(db, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db)
        db.add(
            ERROR_RECORD_COLLECTION,
            {
                "record_id": "er_existing_qt",
                "scholar_id": SCHOLAR_ID,
                "node_code": "",
                "primary_error": "computation",
                "classify_method": "auto_scan",
                "source": "auto_scan",
                "scan_upload_id": SCAN_ID,
            },
        )

        await correct_scan_classify(
            db,
            scan_id=SCAN_ID,
            items=[
                {
                    "error_record_id": "er_existing_qt",
                    "knowledge_point_name": "加法运算",
                    "error_type": "concept",
                    "question_text": "1. 计算 3+5=___（修正）",
                }
            ],
            actor="actor_001",
        )

        records = db.all(ERROR_RECORD_COLLECTION)
        assert records[0]["question_text"] == "1. 计算 3+5=___（修正）"
        assert records[0]["raw_text_corrected"] == "1. 计算 3+5=___（修正）"


class TestCorrectCreateNew:
    @pytest.mark.asyncio
    async def test_create_new_error_record(self):
        """无 error_record_id → 新建 error_record（source=manual_corrected）"""
        db = FakeDB()
        _seed_scan(db, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db)

        result = await correct_scan_classify(
            db,
            scan_id=SCAN_ID,
            items=[
                {
                    "knowledge_point_name": "质数与合数",
                    "error_type": "reading",
                    "raw_text_corrected": "修正后的题干",
                }
            ],
            actor="actor_001",
        )

        assert len(result["corrected"]) == 1
        new_id = result["corrected"][0]["error_record_id"]
        assert new_id  # 新生成 record_id
        assert result["corrected"][0]["knowledge_point_name"] == "质数与合数"
        assert result["corrected"][0]["error_type"] == "reading"

        # error_record 落库
        records = db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 1
        assert records[0]["record_id"] == new_id
        assert records[0]["classify_method"] == CLASSIFY_METHOD_MANUAL_CORRECTED
        assert records[0]["source"] == SOURCE_MANUAL_CORRECTED
        assert records[0]["scan_upload_id"] == SCAN_ID
        assert records[0]["primary_error"] == "reading"
        assert records[0]["node_code"] == "u1"
        assert records[0]["confidence"] == 1.0
        assert records[0]["raw_text_corrected"] == "修正后的题干"

        # scan classify_status 回写 success
        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == CLASSIFY_STATUS_SUCCESS

    @pytest.mark.asyncio
    async def test_create_new_with_question_text_merged(self):
        """B1：新建分支 question_text 优先；老客户端仅 raw_text_corrected 也回填 question_text"""
        db = FakeDB()
        _seed_scan(db, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db)

        result = await correct_scan_classify(
            db,
            scan_id=SCAN_ID,
            items=[
                {
                    "knowledge_point_name": "质数与合数",
                    "error_type": "reading",
                    "question_text": "2. 下列哪个是质数？（题干）",
                    "raw_text_corrected": "老字段题干",
                }
            ],
            actor="actor_001",
        )
        assert len(result["corrected"]) == 1
        records = db.all(ERROR_RECORD_COLLECTION)
        assert records[0]["question_text"] == "2. 下列哪个是质数？（题干）"
        assert records[0]["raw_text_corrected"] == "2. 下列哪个是质数？（题干）"

        # 老客户端只传 raw_text_corrected → question_text 回退合并（向上兼容）
        db2 = FakeDB()
        _seed_scan(db2, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db2)
        await correct_scan_classify(
            db2,
            scan_id=SCAN_ID,
            items=[
                {
                    "knowledge_point_name": "加法运算",
                    "error_type": "method",
                    "raw_text_corrected": "仅老字段题干",
                }
            ],
            actor="actor_001",
        )
        rec2 = db2.all(ERROR_RECORD_COLLECTION)[0]
        assert rec2["question_text"] == "仅老字段题干"
        assert rec2["raw_text_corrected"] == "仅老字段题干"

    @pytest.mark.asyncio
    async def test_create_new_new_kp_name_anchors_extra_ai(self):
        """B1.5：new_kp_name（图谱外专用）→ EXTRA_AI 节点新建并锚定落库"""
        db = FakeDB()
        _seed_scan(db, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db)

        result = await correct_scan_classify(
            db,
            scan_id=SCAN_ID,
            items=[
                {
                    "new_kp_name": "单位换算",
                    "knowledge_point_name": "单位换算",
                    "error_type": "concept",
                    "question_text": "3.5 吨 = ___ 千克",
                }
            ],
            actor="actor_001",
        )
        assert len(result["corrected"]) == 1
        assert result["corrected"][0]["knowledge_point_name"] == "单位换算"
        assert result["corrected"][0]["new_kp_name"] == "单位换算"

        # 图谱外节点已建
        extra_nodes = [
            n for n in db.all(CURRICULUM_NODE_COLLECTION)
            if n.get("textbook_id") == "EXTRA_AI"
        ]
        assert len(extra_nodes) == 1
        assert extra_nodes[0]["title"] == "单位换算"

        # 记录锚定 EXTRA_AI，链锚点 + drill 初始化
        rec = db.all(ERROR_RECORD_COLLECTION)[0]
        assert rec["textbook_id"] == "EXTRA_AI"
        assert rec["node_code"].startswith("xai_")
        assert rec["grade"] == ""
        assert rec["unit_title"] == "未分类"
        assert rec["drill_stats"] == {}
        assert rec["last_drill_result"] == {}

    @pytest.mark.asyncio
    async def test_create_new_nearest_candidate_renames_to_canonical(self):
        """B1.5：kp 未精确命中但子串命中候选（加法简便运算⊃加法运算）→ 就近修正为标准名，
        不再机械改挂候选首个（报告 manual_correct「强行归入少数节点」修复）"""
        db = FakeDB()
        _seed_scan(db, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db)

        result = await correct_scan_classify(
            db,
            scan_id=SCAN_ID,
            items=[
                {
                    "knowledge_point_name": "加法简便运算",
                    "error_type": "computation",
                    "question_text": "25×4+75×4 用简便方法计算",
                }
            ],
            actor="actor_001",
        )
        assert result["corrected"][0]["knowledge_point_name"] == "加法运算"
        rec = db.all(ERROR_RECORD_COLLECTION)[0]
        # 锚定正式候选节点（非 EXTRA_AI），名修正为候选标准名
        assert rec["node_code"] == "u1"
        assert rec["knowledge_point_name"] == "加法运算"
        assert rec["textbook_id"] == "tb1"
        assert rec["grade"] == "3"
        # 未误建 EXTRA_AI 节点
        extra_nodes = [
            n for n in db.all(CURRICULUM_NODE_COLLECTION)
            if n.get("textbook_id") == "EXTRA_AI"
        ]
        assert extra_nodes == []


class TestCorrectMixed:
    @pytest.mark.asyncio
    async def test_update_and_create_in_same_batch(self):
        """同一批修正：已有 record 更新 + 新建并存"""
        db = FakeDB()
        _seed_scan(db, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db)
        db.add(
            ERROR_RECORD_COLLECTION,
            {
                "record_id": "er_old_001",
                "scholar_id": SCHOLAR_ID,
                "node_code": "",
                "primary_error": "computation",
                "classify_method": "auto_scan",
                "source": "auto_scan",
                "scan_upload_id": SCAN_ID,
            },
        )

        result = await correct_scan_classify(
            db,
            scan_id=SCAN_ID,
            items=[
                {
                    "error_record_id": "er_old_001",
                    "knowledge_point_name": "加法运算",
                    "error_type": "concept",
                },
                {
                    "knowledge_point_name": "质数与合数",
                    "error_type": "method",
                },
            ],
            actor="actor_001",
        )

        assert len(result["corrected"]) == 2
        # 第一项更新
        assert result["corrected"][0]["error_record_id"] == "er_old_001"
        # 第二项新建
        assert result["corrected"][1]["error_record_id"] != "er_old_001"

        # error_record 共 2 条
        records = db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 2
        # 全部 classify_method=manual_corrected
        assert all(r["classify_method"] == CLASSIFY_METHOD_MANUAL_CORRECTED for r in records)


class TestCorrectAudit:
    @pytest.mark.asyncio
    async def test_correct_writes_audit(self):
        """修正成功 → 写 scan_correct 审计"""
        db = FakeDB()
        _seed_scan(db, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db)

        await correct_scan_classify(
            db,
            scan_id=SCAN_ID,
            items=[
                {"knowledge_point_name": "加法运算", "error_type": "computation"}
            ],
            actor="actor_001",
        )

        audit_logs = db.all(AUDIT_LOG_COLLECTION)
        correct_audits = [
            log for log in audit_logs if log.get("action") == AUDIT_ACTION_SCAN_CORRECT
        ]
        assert len(correct_audits) == 1
        assert correct_audits[0]["actor"] == "actor_001"
        assert correct_audits[0]["object_ref"] == SCAN_ID
        assert correct_audits[0]["context"]["items_count"] == 1
        assert correct_audits[0]["context"]["created_count"] == 1
