"""POST /math/practice-sheet 生成侧集成验收（离线 FakeDB + 假 LLM，不触网）

覆盖（超时根因修复后的回归基线）：
- ai_knowledge：按 name 匹配出题，每点 2 题，出参无 answer/hint_card；
- include_extended_points=true：每点追加 1 道奥数题（难度按 band 标注）；
- wrong_book（默认源）：按错题聚合出题（每点 2 题）+ primary_errors 回显；
- 幂等：10 分钟窗口内同参数复跑不重复调 LLM / 不重复落库；
- 参数/无选题错误：非法 source → 400、name 未匹配 → 400；
- LLM 出题失败 → 500（错误不吞）。
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from services.routes_math import router as math_router

SCHOLAR = "scholar_gen_001"
_COUNT_RE = re.compile(r"生成 (\d+) 道")


class FakeLLM:
    """client.chat.completions.create 替身：按 prompt 请求题数回合法 JSON，可注入失败"""

    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def _create(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("probe llm down")
        prompt = ""
        for msg in kwargs.get("messages") or []:
            if msg.get("role") == "user":
                prompt = msg.get("content") or ""
        m = _COUNT_RE.search(prompt)
        n = int(m.group(1)) if m else 2
        items = [
            {
                "question": f"q{i}",
                "answer": f"a{i}",
                "difficulty": 3,
                "hint_card": f"h{i}",
            }
            for i in range(n)
        ]
        content = json.dumps({"items": items}, ensure_ascii=False)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeOpenAI:
    def __init__(self, llm: FakeLLM):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=llm._create))


@pytest.fixture()
def stub_llm(monkeypatch):
    """把出题 LLM 替换为假实现（同 knowledge_summary 测试口径，不触真实火山）"""
    llm = FakeLLM()
    monkeypatch.setattr(
        "services.math.practice_sheet._get_llm_client", lambda: FakeOpenAI(llm)
    )
    monkeypatch.setattr("services.math.practice_sheet.LLM_SUMMARY_MODEL", "test-model")
    # 渲染后台任务（playwright）不在生成接口职责内：探针/集成一律跳过
    monkeypatch.setattr(
        "services.math.practice_sheet._schedule_render", lambda _db, _sid: None
    )
    return llm


def _seed_summary_node(fake_db, *, kp_name: str, code: str, node_id: str, with_ext=True) -> dict:
    doc = {
        "node_id": node_id,
        "code": code,
        "title": f"课时·{kp_name}",
        "grade": "五年级",
        "semester": "up",
        "unit_title": "分数",
        "lesson_title": f"课时·{kp_name}",
        "textbook_id": "TB_MATH_5",
        "description_version": 1,
        "ai_summary": {
            "status": "success",
            "knowledge_points": [
                {
                    "name": kp_name,
                    "summary": f"{kp_name} 总结",
                    "ability_dimensions": ["arithmetic"],
                    "source_node_id": node_id,
                    "source_lesson_id": "",
                }
            ],
            "extended_points": (
                [
                    {
                        "name": f"{kp_name}·奥数",
                        "summary": "扩展点说明",
                        "difficulty_band": "入门",
                        "related_knowledge_name": kp_name,
                        "source_lesson_id": "",
                    }
                ]
                if with_ext
                else []
            ),
        },
    }
    fake_db.add("curriculum_node", doc)
    return doc


def _seed_error_record(fake_db, *, scholar_id=SCHOLAR, code: str, occurrence: int = 3) -> dict:
    doc = {
        "scholar_id": scholar_id,
        "node_code": code,
        "occurrence": occurrence,
        "primary_error": "concept",
    }
    fake_db.add("error_record", doc)
    return doc


def _seed_exam_error_record(
    fake_db,
    *,
    scholar_id=SCHOLAR,
    kp_name: str,
    backlink_kp: str,
    code: str = "c1",
    occurrence: int = 3,
    primary_error: str = "concept",
) -> dict:
    """M13 双源落库形态的真题（exam_paper）错题：EXTRA_AI 锚 + exam_backlink_to 回链教材点"""
    doc = {
        "scholar_id": scholar_id,
        "node_code": code,
        "occurrence": occurrence,
        "primary_error": primary_error,
        "knowledge_point_name": kp_name,
        "textbook_id": "EXTRA_AI",
        "kp_source": "exam_paper",
        "exam_backlink_to": {
            "kp_name": backlink_kp,
            "node_code": code,
            "textbook_id": "TB_MATH_5",
            "unit_title": "分数",
            "lesson_title": f"课时·{backlink_kp}",
        },
    }
    fake_db.add("error_record", doc)
    return doc


class TestGenerateAiKnowledge:
    def test_ai_knowledge_generates_two_per_point(self, make_client, fake_db, stub_llm):
        _seed_summary_node(fake_db, kp_name="分数加减法", code="c1", node_id="n1")
        _seed_summary_node(fake_db, kp_name="分数比较", code="c2", node_id="n2")

        res = make_client(math_router).post(
            "/math/practice-sheet",
            json={
                "scholar_id": SCHOLAR,
                "source": "ai_knowledge",
                "knowledge_points": [{"name": "分数加减法"}, {"name": "分数比较"}],
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["source"] == "ai_knowledge"
        assert data["status"] == "generated"
        assert len(data["items"]) == 4          # 每点 2 题
        assert len(data["nodes"]) == 2
        assert stub_llm.calls == 2              # 每点 1 次 LLM
        for it in data["items"]:
            assert "answer" not in it           # 防背题：出参不含答案
            assert "hint_card" not in it
        # 落库 + 渲染任务入队（job 由 _schedule_render 置空，只验证 practice_sheet 落库）
        assert len(fake_db.all("practice_sheet")) == 1

    def test_include_extended_appends_one_olympiad_per_point(
        self, make_client, fake_db, stub_llm
    ):
        _seed_summary_node(fake_db, kp_name="分数加减法", code="c1", node_id="n1")
        _seed_summary_node(fake_db, kp_name="分数比较", code="c2", node_id="n2")

        res = make_client(math_router).post(
            "/math/practice-sheet",
            json={
                "scholar_id": SCHOLAR,
                "source": "ai_knowledge",
                "knowledge_points": [{"name": "分数加减法"}, {"name": "分数比较"}],
                "include_extended_points": True,
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        # 4 基础 + 2 奥数 = 6；奥数难度按 band 标注（入门 → difficulty 3）
        assert len(data["items"]) == 6
        assert stub_llm.calls == 4
        difficulties = {it["difficulty"] for it in data["items"]}
        assert difficulties <= {1, 2, 3, 4, 5}

    def test_idempotent_repeat_skips_llm_and_persist(self, make_client, fake_db, stub_llm):
        _seed_summary_node(fake_db, kp_name="分数加减法", code="c1", node_id="n1")
        payload = {
            "scholar_id": SCHOLAR,
            "source": "ai_knowledge",
            "knowledge_points": [{"name": "分数加减法"}],
        }
        client = make_client(math_router)
        first = client.post("/math/practice-sheet", json=payload)
        assert first.status_code == 200, first.text
        assert stub_llm.calls == 1

        second = client.post("/math/practice-sheet", json=payload)
        assert second.status_code == 200, second.text
        assert stub_llm.calls == 1                 # 幂等：不再调 LLM
        assert len(fake_db.all("practice_sheet")) == 1  # 不重复落库
        assert second.json()["data"]["sheet_id"] == first.json()["data"]["sheet_id"]

    def test_unmatched_knowledge_name_returns_400(self, make_client, fake_db, stub_llm):
        _seed_summary_node(fake_db, kp_name="分数加减法", code="c1", node_id="n1")
        res = make_client(math_router).post(
            "/math/practice-sheet",
            json={
                "scholar_id": SCHOLAR,
                "source": "ai_knowledge",
                "knowledge_points": [{"name": "不存在的知识点"}],
            },
        )
        assert res.status_code == 400, res.text
        assert stub_llm.calls == 0

    def test_llm_failure_returns_500(self, make_client, fake_db, monkeypatch):
        llm = FakeLLM(fail=True)
        monkeypatch.setattr(
            "services.math.practice_sheet._get_llm_client", lambda: FakeOpenAI(llm)
        )
        monkeypatch.setattr("services.math.practice_sheet.LLM_SUMMARY_MODEL", "test-model")
        monkeypatch.setattr(
            "services.math.practice_sheet._schedule_render", lambda _db, _sid: None
        )
        _seed_summary_node(fake_db, kp_name="分数加减法", code="c1", node_id="n1")
        res = make_client(math_router).post(
            "/math/practice-sheet",
            json={
                "scholar_id": SCHOLAR,
                "source": "ai_knowledge",
                "knowledge_points": [{"name": "分数加减法"}],
            },
        )
        assert res.status_code == 500, res.text


class TestGenerateWrongBook:
    def test_wrong_book_aggregates_top_nodes(self, make_client, fake_db, stub_llm):
        # 节点含 ai_summary（出题上下文取第一知识点总结）；错题 occurrence 高者在前
        _seed_summary_node(fake_db, kp_name="分数加减法", code="c1", node_id="n1")
        _seed_summary_node(fake_db, kp_name="分数比较", code="c2", node_id="n2")
        _seed_error_record(fake_db, code="c1", occurrence=5)
        _seed_error_record(fake_db, code="c2", occurrence=2)

        res = make_client(math_router).post(
            "/math/practice-sheet",
            json={"scholar_id": SCHOLAR},  # source 默认 wrong_book
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["source"] == "wrong_book"
        assert len(data["nodes"]) == 2
        assert len(data["items"]) == 4
        assert len(data["primary_errors"]) == 2
        # occurrence≥3 → band=挑战
        assert {b["band"] for b in data["difficulty_bands"]} == {"挑战", "巩固"}
        assert stub_llm.calls == 2

    def test_wrong_book_no_records_returns_400(self, make_client, fake_db, stub_llm):
        res = make_client(math_router).post(
            "/math/practice-sheet", json={"scholar_id": SCHOLAR}
        )
        assert res.status_code == 400, res.text
        assert stub_llm.calls == 0

    def test_invalid_source_returns_400(self, make_client, fake_db, stub_llm):
        res = make_client(math_router).post(
            "/math/practice-sheet",
            json={"scholar_id": SCHOLAR, "source": "not_a_source"},
        )
        assert res.status_code == 400, res.text
        assert stub_llm.calls == 0

    def test_missing_scholar_rejected_by_schema(self, make_client, fake_db, stub_llm):
        # scholar_id 为必填 → Pydantic 校验 422（不触达业务层）
        res = make_client(math_router).post("/math/practice-sheet", json={})
        assert res.status_code == 422, res.text


class TestGenerateDualSource:
    """M14 双源出题：skill_weakness 透传回显 + include_exam_variants 真题同款变式

    口径（设计 §4.4 路线 2-c）：
    - 教材母题 = source_type 'textbook'（默认，每选中教材点 2 道）；
    - 真题同款变式 = source_type 'exam_paper'：学者真题错题 exam_backlink_to 回链到本次选中
      教材点 → 同考点同错因各 1 道；无真题证据 / 回链不命中 → 零行为差；
    - 缺省请求（不带 M14 新字段）→ 与现状一致（兼容锁）。
    """

    def test_exam_variants_mix_into_textbook_mother_questions(
        self, make_client, fake_db, stub_llm
    ):
        _seed_summary_node(fake_db, kp_name="分数加减法", code="c1", node_id="n1")
        _seed_exam_error_record(
            fake_db,
            kp_name="分数加减法·真题A",
            backlink_kp="分数加减法",
            occurrence=3,
            primary_error="computation",
        )
        _seed_exam_error_record(
            fake_db,
            kp_name="分数加减法·真题B",
            backlink_kp="分数加减法",
            occurrence=5,
            primary_error="concept",
        )
        # 回链未命中的真题记录（回链选中点之外的教材点）→ 不入卷
        _seed_exam_error_record(
            fake_db,
            kp_name="分数比较·真题",
            backlink_kp="分数比较",
            occurrence=9,
            primary_error="reading",
        )

        res = make_client(math_router).post(
            "/math/practice-sheet",
            json={
                "scholar_id": SCHOLAR,
                "source": "ai_knowledge",
                "knowledge_points": [{"name": "分数加减法"}],
                "include_exam_variants": True,
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["include_exam_variants"] is True
        sources = [it["source_type"] for it in data["items"]]
        assert sources.count("textbook") == 2  # 母题：每点 2 道
        assert sources.count("exam_paper") == 2  # 命中回链 2 条真题 → 各 1 道
        assert stub_llm.calls == 3  # 1 次母题 + 2 次真题变式
        exam_items = [it for it in data["items"] if it["source_type"] == "exam_paper"]
        assert {it["target_error"] for it in exam_items} == {"computation", "concept"}
        for it in exam_items:
            assert it["node_code"] == "c1"  # 同考点（教材点）出题，非 EXTRA 挂靠
            assert it["source_kp"] == "分数加减法"
        # 落库 sheet 含双源字段（真题同款变式随 sheet 持久化）
        sheet = fake_db.all("practice_sheet")[0]
        assert sheet["include_exam_variants"] is True
        assert {it["source_type"] for it in sheet["items"]} == {"textbook", "exam_paper"}

    def test_exam_variants_require_flag_and_backlink_hit(
        self, make_client, fake_db, stub_llm
    ):
        _seed_summary_node(fake_db, kp_name="分数加减法", code="c1", node_id="n1")
        _seed_exam_error_record(
            fake_db, kp_name="分数加减法·真题", backlink_kp="分数加减法"
        )

        # 未开 include_exam_variants → 纯教材母题（双源零行为差）
        res = make_client(math_router).post(
            "/math/practice-sheet",
            json={
                "scholar_id": SCHOLAR,
                "source": "ai_knowledge",
                "knowledge_points": [{"name": "分数加减法"}],
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert len(data["items"]) == 2
        assert {it["source_type"] for it in data["items"]} == {"textbook"}
        assert stub_llm.calls == 1

        # 开了开关但回链未命中选中点 → 同样无真题变式
        _seed_summary_node(fake_db, kp_name="分数比较", code="c2", node_id="n2")
        stub_llm.calls = 0
        res = make_client(math_router).post(
            "/math/practice-sheet",
            json={
                "scholar_id": SCHOLAR,
                "source": "ai_knowledge",
                "knowledge_points": [{"name": "分数比较"}],
                "include_exam_variants": True,
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert {it["source_type"] for it in data["items"]} == {"textbook"}
        assert stub_llm.calls == 1

    def test_skill_weakness_normalized_echo_and_defaults(
        self, make_client, fake_db, stub_llm
    ):
        _seed_summary_node(fake_db, kp_name="分数加减法", code="c1", node_id="n1")
        payload = {
            "scholar_id": SCHOLAR,
            "source": "ai_knowledge",
            "knowledge_points": [{"name": "分数加减法"}],
            "skill_weakness": [
                {
                    "display_group": "分数计算",
                    "kp_ids": ["n1"],
                    "mastery_avg": 0.3,
                    "weakness_signal": True,
                },
                {"kp_ids": ["x"]},  # 无 display_group → 契约化剔除
            ],
        }
        res = make_client(math_router).post("/math/practice-sheet", json=payload)
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        # 契约化回显：仅 {display_group, kp_ids}
        assert data["skill_weakness"] == [{"display_group": "分数计算", "kp_ids": ["n1"]}]
        assert data["include_exam_variants"] is False
        assert fake_db.all("practice_sheet")[0]["skill_weakness"] == [
            {"display_group": "分数计算", "kp_ids": ["n1"]}
        ]

    def test_legacy_default_request_has_no_behavioral_change(
        self, make_client, fake_db, stub_llm
    ):
        # 缺省（不带 M14 新字段）→ 无真题变式、skill_weakness 空、开关 False，且幂等签名稳定
        _seed_summary_node(fake_db, kp_name="分数加减法", code="c1", node_id="n1")
        payload = {
            "scholar_id": SCHOLAR,
            "source": "ai_knowledge",
            "knowledge_points": [{"name": "分数加减法"}],
        }
        first = make_client(math_router).post("/math/practice-sheet", json=payload)
        assert first.status_code == 200, first.text
        data = first.json()["data"]
        assert data["skill_weakness"] == []
        assert data["include_exam_variants"] is False
        assert len(data["items"]) == 2
        assert all(it["source_type"] == "textbook" for it in data["items"])

        second = make_client(math_router).post("/math/practice-sheet", json=payload)
        assert second.status_code == 200
        assert second.json()["data"]["sheet_id"] == data["sheet_id"]  # 幂等命中
