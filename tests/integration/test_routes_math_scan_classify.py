"""F4.3 扫描归类接口验收测试（契约 api-contract.md §3.10 POST /math/scan/classify）

验收场景：
- 400 缺参（scan_id 为空）
- 404 scan_id 不存在
- 404 OCR 未完成
- 200 高置信归类成功 → 写 error_record + status=success
- 200 低置信 → needs_review（不写 error_record）
- 200 幂等命中（已归类返回既有结果，Judge 不调用）
"""
from __future__ import annotations

import pytest

from services.database import (
    CURRICULUM_NODE_COLLECTION,
    ERROR_RECORD_COLLECTION,
    MATH_SCAN_UPLOAD_COLLECTION,
)
from services.routes_math import router as math_router

SCAN_ID = "scan_int_001"
SCHOLAR_ID = "scholar_int_001"


# ---------------------------------------------------------------------------
# 种子数据
# ---------------------------------------------------------------------------


def _seed_scan(
    fake_db,
    *,
    scan_id: str = SCAN_ID,
    ocr_status: str = "success",
    classify_status: str = "pending",
    classify_result: list | None = None,
) -> dict:
    doc = {
        "scan_id": scan_id,
        "scholar_id": SCHOLAR_ID,
        "image_url": "https://example.com/scan.jpg",
        "image_file_id": "cloud://test/scan.jpg",
        "image_hash": "abc123",
        "ocr_status": ocr_status,
        "ocr_text": "1. 计算 3+5=___\n2. 下列哪个是质数？",
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
    fake_db.add(MATH_SCAN_UPLOAD_COLLECTION, doc)
    return doc


def _seed_knowledge_nodes(fake_db) -> None:
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


def _seed_knowledge_nodes_with_units(fake_db) -> None:
    """多教材/多单元候选节点（M9 photo_context 优先锚点种子）

    - tb-multi / 第3单元 小数乘法（课时1）→ 小数乘整数、小数乘小数
    - tb-multi / 第2单元 位置（课时1）→ 用数对确定位置
    """
    fake_db.add(
        CURRICULUM_NODE_COLLECTION,
        {
            "node_id": "n-unit-a",
            "code": "ua1",
            "grade": "5",
            "semester": "up",
            "textbook_id": "tb-multi",
            "title": "小数乘整数",
            "unit_title": "第3单元 小数乘法",
            "lesson_title": "课时1 小数乘整数",
            "ai_summary": {
                "status": "success",
                "knowledge_points": [
                    {"name": "小数乘整数"},
                    {"name": "小数乘小数"},
                ],
            },
        },
    )
    fake_db.add(
        CURRICULUM_NODE_COLLECTION,
        {
            "node_id": "n-unit-b",
            "code": "ub1",
            "grade": "5",
            "semester": "up",
            "textbook_id": "tb-multi",
            "title": "用数对确定位置",
            "unit_title": "第2单元 位置",
            "lesson_title": "课时1 用数对确定位置",
            "ai_summary": {
                "status": "success",
                "knowledge_points": [{"name": "用数对确定位置"}],
            },
        },
    )


def _high_conf_judge_result() -> dict:
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


def _low_conf_judge_result() -> dict:
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
# 错误码验收
# ---------------------------------------------------------------------------


class TestScanClassifyErrorCodes:
    def test_400_when_scan_id_missing(self, make_client, fake_db):
        """缺参 scan_id 为空 → 400"""
        client = make_client(math_router)
        res = client.post("/math/scan/classify", json={"scan_id": ""})
        assert res.status_code == 400, res.text
        assert "scan_id" in res.json()["detail"]

    def test_404_when_scan_not_found(self, make_client, fake_db):
        """scan_id 不存在 → 404"""
        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify", json={"scan_id": "no_such_scan"}
        )
        assert res.status_code == 404, res.text

    def test_404_when_ocr_not_ready(self, make_client, fake_db):
        """OCR 未完成（pending）→ 404"""
        _seed_scan(fake_db, ocr_status="pending")
        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify", json={"scan_id": SCAN_ID}
        )
        assert res.status_code == 404, res.text


# ---------------------------------------------------------------------------
# 归类成功
# ---------------------------------------------------------------------------


class TestScanClassifySuccess:
    def test_high_confidence_returns_success(
        self, make_client, fake_db, monkeypatch
    ):
        """高置信 + 知识点匹配 → 200 status=success + error_record 落库"""
        _seed_scan(fake_db)
        _seed_knowledge_nodes(fake_db)

        async def fake_judge(ocr_text, candidates):
            return _high_conf_judge_result()

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify", json={"scan_id": SCAN_ID}
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["scan_id"] == SCAN_ID
        assert data["status"] == "success"
        assert len(data["items"]) == 2
        # 每道题都写了 error_record
        for item in data["items"]:
            assert item["error_record_id"]
            assert item["knowledge_point_name"]
            assert item["error_type"] in (
                "concept",
                "method",
                "computation",
                "reading",
            )

        # error_record 落库 2 条
        records = fake_db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 2
        assert records[0]["classify_method"] == "auto_scan"
        assert records[0]["source"] == "auto_scan"

    def test_low_confidence_returns_needs_review(
        self, make_client, fake_db, monkeypatch
    ):
        """低置信 / 知识点未匹配 → 200 status=needs_review（不写 error_record）"""
        _seed_scan(fake_db)
        _seed_knowledge_nodes(fake_db)

        async def fake_judge(ocr_text, candidates):
            return _low_conf_judge_result()

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify", json={"scan_id": SCAN_ID}
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["status"] == "needs_review"
        # 第一道高置信写了 error_record；第二道低置信未写
        assert data["items"][0]["error_record_id"]
        assert not data["items"][1]["error_record_id"]
        # error_record 仅落库 1 条
        assert len(fake_db.all(ERROR_RECORD_COLLECTION)) == 1


# ---------------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------------


class TestScanClassifyIdempotency:
    def test_already_classified_returns_existing(
        self, make_client, fake_db, monkeypatch
    ):
        """已归类（status=success）非 force → 返回既有结果，Judge 不调用"""
        existing_items = [
            {
                "error_record_id": "er_existing_001",
                "knowledge_point_name": "加法运算",
                "error_type": "computation",
                "confidence": 0.9,
                "ocr_block_id": "blk_0001",
            }
        ]
        _seed_scan(
            fake_db,
            classify_status="success",
            classify_result=existing_items,
        )

        call_count: list = []

        async def fake_judge(ocr_text, candidates):
            call_count.append(1)
            return _high_conf_judge_result()

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify", json={"scan_id": SCAN_ID}
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["status"] == "success"
        assert data["items"] == existing_items
        assert len(call_count) == 0  # Judge 未被调用

    def test_force_reclassify_re_invokes_judge(
        self, make_client, fake_db, monkeypatch
    ):
        """force_reclassify=true → 忽略已有结果，重新调用 Judge"""
        _seed_scan(
            fake_db,
            classify_status="success",
            classify_result=[],
        )
        _seed_knowledge_nodes(fake_db)

        call_count: list = []

        async def fake_judge(ocr_text, candidates):
            call_count.append(1)
            return _high_conf_judge_result()

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify",
            json={"scan_id": SCAN_ID, "force_reclassify": True},
        )
        assert res.status_code == 200, res.text
        assert res.json()["data"]["status"] == "success"
        assert len(call_count) == 1  # Judge 被重新调用


# ---------------------------------------------------------------------------
# M9：photo_context 优先锚点
# ---------------------------------------------------------------------------


class TestScanClassifyPhotoContext:
    def test_photo_context_anchor_biases_unit_and_writes_context(
        self, make_client, fake_db, monkeypatch
    ):
        """带 photo_context 命中教材链 → 候选收敛该单元（Judge 只收到锚点）
        且归类写回 error_record.photo_context（picked_from_unit:true）"""
        _seed_scan(fake_db)
        _seed_knowledge_nodes(fake_db)
        _seed_knowledge_nodes_with_units(fake_db)

        judge_calls: list = []

        async def fake_judge(ocr_text, candidates):
            judge_calls.append(candidates)
            names = [c["kp_name"] for c in candidates]
            return {
                "items": [
                    {
                        "knowledge_point_name": names[0],
                        "error_type": "computation",
                        "confidence": 0.9,
                        "ocr_block_id": "blk_0001",
                    }
                ]
            }

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify",
            json={
                "scan_id": SCAN_ID,
                "photo_context": (
                    '{"textbook_id": "tb-multi", '
                    '"unit_title": "第3单元 小数乘法"}'
                ),
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["status"] == "success"
        assert len(data["items"]) == 1
        assert data["items"][0]["error_record_id"]

        # ① Judge 仅收到第3单元的锚点候选（优先锚点收敛，未混入其他单元）
        assert len(judge_calls) == 1
        anchor_names = {c["kp_name"] for c in judge_calls[0]}
        assert anchor_names == {"小数乘整数", "小数乘小数"}
        assert "用数对确定位置" not in anchor_names
        assert "加法运算" not in anchor_names

        # ② error_record 落库 photo_context（含 picked_from_unit:true）
        records = fake_db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 1
        assert records[0]["photo_context"] == {
            "textbook_id": "tb-multi",
            "unit_title": "第3单元 小数乘法",
            "picked_from_unit": True,
        }
        # ③ 链锚点冗余随命中节点落库（单元/课时/教材均收敛到锚点链）
        assert records[0]["textbook_id"] == "tb-multi"
        assert records[0]["unit_title"] == "第3单元 小数乘法"
        assert records[0]["lesson_title"] == "课时1 小数乘整数"
        # ④ 审计沿用 scan_classify 动作并标记锚点
        audits = fake_db.all("audit_log")
        classify_audits = [
            a for a in audits if a.get("action") == "scan_classify"
        ]
        assert len(classify_audits) == 1
        assert classify_audits[0]["context"]["photo_context_anchored"] is True

    def test_photo_context_low_conf_falls_back_to_full_guess(
        self, make_client, fake_db, monkeypatch
    ):
        """锚点归类无高置信命中（题面与所选单元不符）→ 回退全量自动猜，
        结果与现状一致且不写 photo_context"""
        _seed_scan(fake_db)
        _seed_knowledge_nodes(fake_db)
        _seed_knowledge_nodes_with_units(fake_db)

        judge_calls: list = []

        async def fake_judge(ocr_text, candidates):
            judge_calls.append(candidates)
            if len(judge_calls) == 1:
                # 第一次：锚点候选集（第3单元），Judge 低置信判不出该单元题面
                return {
                    "items": [
                        {
                            "knowledge_point_name": "小数乘整数",
                            "error_type": "concept",
                            "confidence": 0.4,
                            "ocr_block_id": "blk_0001",
                        }
                    ]
                }
            # 第二次：全量候选自动猜（与现状同口径）
            return _high_conf_judge_result()

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify",
            json={
                "scan_id": SCAN_ID,
                "photo_context": (
                    '{"textbook_id": "tb-multi", '
                    '"unit_title": "第3单元 小数乘法"}'
                ),
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["status"] == "success"
        assert len(data["items"]) == 2

        # ① 锚点先判一次、回退全量再判一次
        assert len(judge_calls) == 2
        assert {c["kp_name"] for c in judge_calls[0]} == {
            "小数乘整数",
            "小数乘小数",
        }
        full_names = {c["kp_name"] for c in judge_calls[1]}
        assert {"加法运算", "质数与合数", "用数对确定位置"} <= full_names
        # ② 回退自动猜结果不写 photo_context（与现状一致）
        records = fake_db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 2
        for r in records:
            assert "photo_context" not in r
        audits = fake_db.all("audit_log")
        classify_audits = [
            a for a in audits if a.get("action") == "scan_classify"
        ]
        assert classify_audits[0]["context"]["photo_context_anchored"] is False

    def test_photo_context_chain_invalid_falls_back_default(
        self, make_client, fake_db, monkeypatch
    ):
        """链校验失败（单元不存在于该教材/教材不存在）→ 直接全量自动猜，
        行为与缺省一致（单次 Judge + 不写 photo_context）"""
        _seed_scan(fake_db)
        _seed_knowledge_nodes(fake_db)
        _seed_knowledge_nodes_with_units(fake_db)

        judge_calls: list = []

        async def fake_judge(ocr_text, candidates):
            judge_calls.append(candidates)
            return _high_conf_judge_result()

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify",
            json={
                "scan_id": SCAN_ID,
                "photo_context": (
                    '{"textbook_id": "tb-multi", '
                    '"unit_title": "不存在的单元"}'
                ),
            },
        )
        assert res.status_code == 200, res.text
        assert res.json()["data"]["status"] == "success"
        # 单次 Judge 全量候选（未触发锚点收敛）
        assert len(judge_calls) == 1
        full_names = {c["kp_name"] for c in judge_calls[0]}
        assert "加法运算" in full_names and "小数乘整数" in full_names
        for r in fake_db.all(ERROR_RECORD_COLLECTION):
            assert "photo_context" not in r

    def test_photo_context_malformed_json_ignored(
        self, make_client, fake_db, monkeypatch
    ):
        """photo_context 非法 JSON → 忽略并按现状自动归类（静默降级不报错）"""
        _seed_scan(fake_db)
        _seed_knowledge_nodes(fake_db)

        judge_calls: list = []

        async def fake_judge(ocr_text, candidates):
            judge_calls.append(candidates)
            return _high_conf_judge_result()

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify",
            json={"scan_id": SCAN_ID, "photo_context": "not-a-json{"},
        )
        assert res.status_code == 200, res.text
        assert res.json()["data"]["status"] == "success"
        assert len(judge_calls) == 1
        for r in fake_db.all(ERROR_RECORD_COLLECTION):
            assert "photo_context" not in r

    def test_force_reclassify_with_photo_context_reattach(
        self, make_client, fake_db, monkeypatch
    ):
        """force_reclassify=true + photo_context（M8 一键重归路径）→
        忽略既有结果重跑，并优先新单元锚点 + 写回 photo_context"""
        _seed_scan(
            fake_db,
            classify_status="success",
            classify_result=[
                {
                    "error_record_id": "er_old_001",
                    "knowledge_point_name": "加法运算",
                    "error_type": "computation",
                    "confidence": 0.9,
                    "ocr_block_id": "blk_0001",
                }
            ],
        )
        _seed_knowledge_nodes(fake_db)
        _seed_knowledge_nodes_with_units(fake_db)

        judge_calls: list = []

        async def fake_judge(ocr_text, candidates):
            judge_calls.append(candidates)
            names = [c["kp_name"] for c in candidates]
            return {
                "items": [
                    {
                        "knowledge_point_name": names[0],
                        "error_type": "computation",
                        "confidence": 0.95,
                        "ocr_block_id": "blk_0001",
                    }
                ]
            }

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify",
            json={
                "scan_id": SCAN_ID,
                "force_reclassify": True,
                "photo_context": (
                    '{"textbook_id": "tb-multi", '
                    '"unit_title": "第2单元 位置"}'
                ),
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["status"] == "success"
        assert len(judge_calls) == 1
        # 重归锚点 = 第2单元 位置（仅 用数对确定位置）
        assert {c["kp_name"] for c in judge_calls[0]} == {"用数对确定位置"}
        records = fake_db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 1
        assert records[0]["photo_context"] == {
            "textbook_id": "tb-multi",
            "unit_title": "第2单元 位置",
            "picked_from_unit": True,
        }
        assert records[0]["knowledge_point_name"] == "用数对确定位置"


# ---------------------------------------------------------------------------
# M13：双源知识锚 — 真题/图谱外 → exam_paper 题簇（不写正式教材链）
# ---------------------------------------------------------------------------


class TestScanClassifyExamPaperSource:
    """M13（Phase 2-b / O3=A）：真题归类落 EXTRA_AI（真题库）题簇且
    kp_source='exam_paper'；命中教材点写 exam_backlink_to 软回链；正式教材链零脏数据。"""

    def test_out_of_graph_lands_exam_paper_cluster_with_backlink(
        self, make_client, fake_db, monkeypatch
    ):
        """真题/图谱外（鸡兔同笼 + candidate_hits 小数乘整数）→ error_record
        锚 EXTRA_AI 真题题簇：kp_source='exam_paper' + exam_backlink_to 教材点；
        正式教材链无脏数据（不产生任何正式链锚定记录）"""
        _seed_scan(fake_db)
        _seed_knowledge_nodes(fake_db)
        _seed_knowledge_nodes_with_units(fake_db)

        async def fake_judge(ocr_text, candidates):
            return {
                "items": [
                    {
                        "knowledge_point_name": "鸡兔同笼",
                        "error_type": "method",
                        "confidence": 0.85,
                        "ocr_block_id": "blk_0001",
                        "candidate_hits": ["小数乘整数"],
                    }
                ]
            }

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post("/math/scan/classify", json={"scan_id": SCAN_ID})
        assert res.status_code == 200, res.text
        assert res.json()["data"]["status"] == "success"

        records = fake_db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 1
        rec = records[0]
        # 真题题簇锚 EXTRA_AI（不落正式教材链）
        assert rec["textbook_id"] == "EXTRA_AI"
        assert rec["kp_source"] == "exam_paper"
        # 教材点软回链（仅引用不改锚）
        assert rec["exam_backlink_to"] == {
            "node_code": "ua1",
            "kp_name": "小数乘整数",
            "textbook_id": "tb-multi",
            "unit_title": "第3单元 小数乘法",
            "lesson_title": "课时1 小数乘整数",
        }
        # 正式教材链零脏数据：tb-multi/tb1 上无任何新增锚定记录
        assert not any(
            r.get("textbook_id") not in ("EXTRA_AI", "")
            for r in records
        )

    def test_formal_hit_keeps_textbook_anchor_without_exam_fields(
        self, make_client, fake_db, monkeypatch
    ):
        """正式链命中 → error_record 锚教材节点，不写 kp_source/exam_backlink_to
        （读取层缺省 textbook；存量记录零差，无真题源标记混入）"""
        _seed_scan(fake_db)
        _seed_knowledge_nodes(fake_db)

        async def fake_judge(ocr_text, candidates):
            return _high_conf_judge_result()

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post("/math/scan/classify", json={"scan_id": SCAN_ID})
        assert res.status_code == 200, res.text

        records = fake_db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 2
        for rec in records:
            assert rec["textbook_id"] == "tb1"
            assert "kp_source" not in rec
            assert "exam_backlink_to" not in rec
