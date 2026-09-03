#!/usr/bin/env python3
"""沉浸式学习页任务进度接口验证脚本（对应 docs_v1《沉浸式学习页任务进度接口测试用例》）

背景：
  沉浸式学习页缺陷 —— 一个任务（当前课组任务卡）结束后：
    1) 没有手动进入下一任务（下一课）的链接；
    2) 次日重新进入不自动推进到下一任务，仍停留已通关课的 allCleared 空态。
  已定位：整课通关时端侧从不调用 `PUT /scholar/{sid}/books/{tid}/position`
  推进断点锚点（current_chapter_id/current_lesson_id/current_group_id），
  故次日 GET books 仍返回旧课锚点。

本脚本用于证明后端接口侧能完整支撑该闭环（作为修复回归验收口径），
覆盖 docs_v1 用例文档 §6 的 K1~K10：

  K1  冒烟：读接口可用与数据可达
  K2  冷启动锚点写入（PUT position 建学者×教材关联 + GET books 回读）
  K3  group 级断点更新与幂等
  K4  锚点组 == 当前课第一个未掌握组（定位口径）
  K5  单组任务完成（三技能 mastered 上报）→ 课内下一组解锁
  K6  整课通关 → 手动进入下一课任务（推进锚点 → 重新定位）
  K7  次日重进自动定位下一任务（GET books → GET lessons → GET groups 同构读序列）
  K8  边界：最后一课通关、无下一课
  K9  异常与参数校验（400 / 404 / 空数组 / data:null）
  K10 组内技能步断点明细 immersive-progress CRUD（辅助链路，幂等清理）

口径（与小程序一致）：
  - 句掌握：组视图 sentence.status >= 3
  - 组掌握：组内全部句 status >= 3（木桶口径）
  - 下一任务：当前课按返回顺序第一个未掌握组

数据隔离：
  - 默认生成一次性隔离学者 imm_verify_{yyyyMMdd}_{HHmmss}_{rand}；
    全程只写该学者的 skill_state / scholar_book / immersive-progress，不碰内容。
  - immersive-progress 记录执行后幂等 DELETE 清理（K10）。

用法：
  # 服务已启动（python main.py，默认 127.0.0.1:8080）
  python scripts/immersive_task_progress_verify.py

  # 显式指定教材/学者
  python scripts/immersive_task_progress_verify.py --textbook-id tb_xxx --scholar-id imm_xxx

  # 服务未启动时自动拉起 uvicorn
  python scripts/immersive_task_progress_verify.py --port 8080

  # 远端环境
  python scripts/immersive_task_progress_verify.py --base-url https://xxx.example.com

退出码：全部通过(或 SKIP) → 0；任一用例 FAIL → 1。
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("immersive_task_progress_verify")

BASE_URL_DEFAULT_PORT = 8080
# 沉浸式页四步流程中三技能上报（ec/ce 均记 translation；shadowing→speaking；listening→listening）
SKILLS = ("translation", "speaking", "listening")
STATUS_MASTERED = "mastered"
MASTERED_INT = 3  # mastered=3 / review_due=4 均 >= 3 视为掌握

_results: list[dict] = []  # {case, title, status, detail}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _first_unmastered_group_id(groups: list[dict]) -> str | None:
    """当前课第一个未掌握组：按返回顺序，组内存在句 status<3 即为未掌握组。

    与小程序 buildNewSentenceCard 口径一致（组掌握 = 组内最弱句木桶）。
    """
    for g in groups or []:
        sents = g.get("sentences") or []
        if not sents:
            continue  # 无内容句的组不作为任务
        if any(int((s.get("status") or 0)) < MASTERED_INT for s in sents):
            return g.get("group_id") or ""
    return None


def _group_sentences(groups: list[dict], group_id: str) -> list[dict]:
    for g in groups or []:
        if g.get("group_id") == group_id:
            return g.get("sentences") or []
    return []


def _require_ok(resp: httpx.Response, case: str):
    if resp.status_code != 200:
        raise AssertionError(
            f"{case} 期望 200，实际 {resp.status_code}："
            f"{resp.text[:300]}"
        )
    body = resp.json()
    if body.get("success") is not True:
        raise AssertionError(f"{case} 响应 success != true：{body}")
    return body


# ---------------------------------------------------------------------------
# 用例注册
# ---------------------------------------------------------------------------


def case(case_id: str, title: str):
    """装饰器：注册用例为函数，独立 try/except，失败不中断后续用例。"""

    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                fn(*args, **kwargs)
                status, detail = "PASS", ""
            except AssertionError as e:
                status, detail = "FAIL", str(e)
            except Exception as e:  # noqa: BLE001 - 逐用例隔离，报错即 FAIL
                status, detail = "FAIL", f"{type(e).__name__}: {e}"
            _results.append({"case": case_id, "title": title, "status": status, "detail": detail})
            mark = "PASS" if status == "PASS" else "FAIL"
            print(f"[{mark}] {case_id} {title}" + (f"  -- {detail}" if detail else ""))

        return wrapper

    return decorator


def case_skip(case_id: str, title: str, detail: str):
    _results.append({"case": case_id, "title": title, "status": "SKIP", "detail": detail})
    print(f"[SKIP] {case_id} {title}  -- {detail}")


# ---------------------------------------------------------------------------
# HTTP 请求封装（依赖运行时上下文 ctx）
# ---------------------------------------------------------------------------


class Ctx:
    def __init__(self, client: httpx.Client, base_url: str, scholar_id: str,
                 textbook_id: str):
        self.client = client
        self.base_url = base_url
        self.scholar_id = scholar_id
        self.textbook_id = textbook_id
        # 由 setup 填充
        self.lessons: list[dict] = []          # GET lessons 原始行
        self.lessons_meta: list[dict] = []     # {lesson_id, groups, group_count}
        self.lesson_l1: str | None = None      # 主推进课（组数最多）
        self.lesson_l2: str | None = None      # L1 之后第一个可用课
        self.lesson_last: str | None = None    # 最后一个可用课
        self.g1: str | None = None             # L1 第一个未掌握组
        self.g1_has_next: bool = False         # L1 组数 >= 2

    # ---- 单接口封装 ----
    def get_books(self, subject_type: str = "english") -> dict:
        r = self.client.get(
            f"{self.base_url}/scholar/{self.scholar_id}/books",
            params={"subject_type": subject_type},
        )
        return _require_ok(r, "GET books")

    def get_lessons(self) -> dict:
        r = self.client.get(
            f"{self.base_url}/scholar/{self.scholar_id}/textbooks/{self.textbook_id}/lessons",
            params={"subject_type": "english"},
        )
        return _require_ok(r, "GET lessons")

    def get_groups(self, lesson_id: str) -> dict:
        r = self.client.get(
            f"{self.base_url}/tracking/textbooks/{self.textbook_id}/lessons/{lesson_id}/groups",
            params={"scholar_id": self.scholar_id},
        )
        return _require_ok(r, "GET groups")

    def put_position(self, body: dict) -> dict:
        r = self.client.put(
            f"{self.base_url}/scholar/{self.scholar_id}/books/{self.textbook_id}/position",
            json=body,
        )
        return _require_ok(r, "PUT position")

    def report_mastered(self, sentence_id: str, skill_code: str) -> None:
        r = self.client.post(f"{self.base_url}/tracking/state", json={
            "scholar_id": self.scholar_id,
            "sentence_id": sentence_id,
            "skill_code": skill_code,
            "status": STATUS_MASTERED,
            "mastery": 1.0,
            "score": 100,
            "attempt_type": "translate" if skill_code == "translation"
            else "speak" if skill_code == "speaking" else "listen",
            "attempt_status": "completed",
            "time_spent": 5,
        })
        body = _require_ok(r, f"POST tracking/state({sentence_id},{skill_code})")
        state = body.get("data", {}).get("state", {}) or {}
        if state.get("status") != STATUS_MASTERED:
            raise AssertionError(
                f"上报 mastered 后 state.status={state.get('status')!r}"
            )

    def master_group(self, group_id: str) -> None:
        """组内每句 × 三技能上报 mastered（模拟组任务四步流程全部通关）。"""
        sents = _group_sentences(self._groups_of(self.lesson_l1), group_id)
        for s in sents:
            sid = s.get("sentence_id")
            if not sid:
                continue
            for skill in SKILLS:
                self.report_mastered(sid, skill)

    def master_all_groups(self, lesson_id: str) -> None:
        """把某课全部组逐句三技能 mastered（幂等，重报只累加 attempt_count）。"""
        groups = self._groups_of(lesson_id)
        for g in groups or []:
            for s in (g.get("sentences") or []):
                sid = s.get("sentence_id")
                if not sid:
                    continue
                for skill in SKILLS:
                    self.report_mastered(sid, skill)

    def _groups_of(self, lesson_id: str | None) -> list[dict]:
        if not lesson_id:
            return []
        data = self.get_groups(lesson_id)
        return data.get("data", {}).get("groups") or []


# ---------------------------------------------------------------------------
# 环境准备（数据发现与选择，见用例文档 §4）
# ---------------------------------------------------------------------------


def _extract_textbook_records(payload: dict) -> list[dict]:
    """/textbook 返回形态不一（db.query 原始响应），多形态兜底提取。"""
    for cand in (
        payload.get("records"),
        (payload.get("data") or {}).get("records"),
        payload.get("data") if isinstance(payload.get("data"), list) else None,
    ):
        if isinstance(cand, list):
            return cand
    return []


def discover_textbook(client: httpx.Client, base_url: str) -> str:
    r = client.get(f"{base_url}/textbook", params={"subject_type": "english"})
    if r.status_code != 200:
        raise RuntimeError(f"GET /textbook 失败: {r.status_code} {r.text[:200]}")
    records = _extract_textbook_records(r.json())
    english = [
        tb for tb in records
        if (tb.get("subject_type") or "english") == "english"
        and (tb.get("textbook_id") or tb.get("_id"))
    ]
    if not english:
        raise RuntimeError(
            "自动发现不到英语教材（GET /textbook 无 english 记录），请用 --textbook-id 显式指定"
        )
    tb = english[0]
    return tb.get("textbook_id") or tb.get("_id")


def setup_lessons(ctx: Ctx) -> None:
    """按用例文档 §4.2 选择 L1/L2/L_last 并缓存组视图。"""
    body = ctx.get_lessons()
    lessons = body.get("data", {}).get("lessons") or []
    if len(lessons) < 1:
        raise RuntimeError("教材无可用课（GET lessons 为空），无法执行用例")

    metas = []
    for lsn in lessons[:20]:
        lid = lsn.get("lesson_id")
        if not lid:
            continue
        try:
            gdata = ctx.get_groups(lid)
        except AssertionError:
            continue  # 跳过组视图异常的课（如无句子的课）
        groups = gdata.get("data", {}).get("groups") or []
        metas.append({"lesson_id": lid, "groups": groups, "group_count": len(groups)})

    usable = [m for m in metas if m["group_count"] >= 1]
    if not usable:
        raise RuntimeError("教材所有可用课均无组/句，无法执行用例")

    # L1 = 组数最多的一课；L2 = L1 之后第一个可用课；L_last = 最后一个可用课
    l1 = max(usable, key=lambda m: m["group_count"])
    l1_idx = metas.index(l1)
    l2 = next((m for m in metas[l1_idx + 1:] if m["group_count"] >= 1), None)
    l_last = max(usable, key=lambda m: metas.index(m))

    ctx.lessons = lessons
    ctx.lessons_meta = metas
    ctx.lesson_l1 = l1["lesson_id"]
    ctx.lesson_l2 = l2["lesson_id"] if l2 else None
    ctx.lesson_last = l_last["lesson_id"]

    groups_l1 = l1["groups"]
    ctx.g1 = _first_unmastered_group_id(groups_l1)
    ctx.g1_has_next = len(groups_l1) >= 2

    print("\n===== 数据准备 =====")
    print(f"  scholar_id    = {ctx.scholar_id}")
    print(f"  textbook_id   = {ctx.textbook_id}")
    print(f"  L1(主推进课)   = {ctx.lesson_l1}  组数={l1['group_count']}")
    print(f"  L2(下一课)     = {ctx.lesson_l2 or '无'}")
    print(f"  L_last(最后课) = {ctx.lesson_last}")
    print(f"  L1 首任务组 G1 = {ctx.g1}")


# ---------------------------------------------------------------------------
# K1~K10 用例实现
# ---------------------------------------------------------------------------


@case("K1", "冒烟：读接口可用与数据可达")
def k1(ctx: Ctx):
    assert ctx.lesson_l1, "无可用课 L1"
    b = ctx.get_groups(ctx.lesson_l1)
    groups = b.get("data", {}).get("groups") or []
    assert groups, "L1 组视图为空"
    assert all((g.get("sentences") or []) for g in groups), "L1 存在无句子的组"
    if ctx.lesson_l2:
        b2 = ctx.get_groups(ctx.lesson_l2)
        assert b2.get("data", {}).get("groups"), f"L2({ctx.lesson_l2}) 组视图为空"
    else:
        print("       → 环境无下一课 L2，冒烟仅验证 L1（K6/K7 将 SKIP）")


@case("K2", "冷启动锚点写入：PUT position 建关联 + GET books 回读")
def k2(ctx: Ctx):
    assert ctx.lesson_l1
    doc = ctx.put_position({
        "current_chapter_id": ctx.lesson_l1,
        "current_lesson_id": ctx.lesson_l1,
        "subject_type": "english",
    })["data"]
    assert doc.get("status") == "learning", f"status={doc.get('status')!r}"
    assert doc.get("current_chapter_id") == ctx.lesson_l1
    assert doc.get("current_lesson_id") == ctx.lesson_l1

    books = ctx.get_books()["data"]["books"]
    book = next((x for x in books if x.get("textbook_id") == ctx.textbook_id), None)
    assert book, "GET books 未返回该教材"
    assert book.get("current_chapter_id") == ctx.lesson_l1
    assert book.get("current_lesson_id") == ctx.lesson_l1
    assert isinstance(book.get("last_studied_at"), int), "last_studied_at 应为毫秒整数"


@case("K3", "group 级断点更新与幂等（部分字段更新，其余锚点保留）")
def k3(ctx: Ctx):
    assert ctx.g1, "L1 无未掌握组，无法取 G1"
    doc = ctx.put_position({"current_group_id": ctx.g1})["data"]
    assert doc.get("current_group_id") == ctx.g1

    doc2 = ctx.put_position({"current_group_id": ctx.g1})["data"]  # 幂等重放
    assert doc2.get("current_group_id") == ctx.g1

    book = next(x for x in ctx.get_books()["data"]["books"]
                if x.get("textbook_id") == ctx.textbook_id)
    assert book.get("current_chapter_id") == ctx.lesson_l1, "chapter 锚点被覆盖"
    assert book.get("current_lesson_id") == ctx.lesson_l1, "lesson 锚点被覆盖"
    assert book.get("current_group_id") == ctx.g1
    assert book.get("total_time_spent") == 0, "position 更新不应累加学习时长"


@case("K4", "定位口径：锚点组 == 当前课第一个未掌握组")
def k4(ctx: Ctx):
    assert ctx.g1
    book = next(x for x in ctx.get_books()["data"]["books"]
                if x.get("textbook_id") == ctx.textbook_id)
    assert book.get("current_group_id") == ctx.g1, "books 锚点组异常"
    first = _first_unmastered_group_id(ctx.get_groups(ctx.lesson_l1)["data"]["groups"])
    assert first == ctx.g1, f"组视图第一个未掌握组({first}) != 锚点组({ctx.g1})"


@case("K5", "单组任务完成 → 课内下一组解锁（组级推进）")
def k5(ctx: Ctx):
    assert ctx.g1
    ctx.master_group(ctx.g1)
    groups = ctx.get_groups(ctx.lesson_l1)["data"]["groups"]
    # 组内全部句已掌握
    assert all(
        int((s.get("status") or 0)) >= MASTERED_INT
        for s in _group_sentences(groups, ctx.g1)
    ), f"G1={ctx.g1} 仍有句未掌握"
    first = _first_unmastered_group_id(groups)
    if ctx.g1_has_next:
        assert first is not None and first != ctx.g1, (
            "G1 已通关但下一未掌握组未出现"
        )
        print(f"       → G1 通关后，课内下一任务组 = {first}")
    else:
        assert first is None, "L1 仅一组，通关后应无未掌握组"
        print("       → L1 仅 1 组：单组通关即整课通关（语义并入 K6）")


@case("K6", "整课通关 → 手动进入下一课任务（推进锚点 + 重新定位）")
def k6(ctx: Ctx):
    if not ctx.lesson_l2:
        case_skip("K6", "整课通关 → 手动进入下一课任务（推进锚点 + 重新定位）",
                  "无可用下一课 L2，跳过")
        return
    # 1) L1 剩余全部组通关
    ctx.master_all_groups(ctx.lesson_l1)
    groups_l1 = ctx.get_groups(ctx.lesson_l1)["data"]["groups"]
    assert _first_unmastered_group_id(groups_l1) is None, "L1 仍存在未掌握组"

    # 2) 取 L2 第一个未掌握组 N2（下一任务）
    groups_l2 = ctx.get_groups(ctx.lesson_l2)["data"]["groups"]
    n2 = _first_unmastered_group_id(groups_l2)
    assert n2, f"L2({ctx.lesson_l2}) 无未掌握组，无法作为下一任务"

    # 3) 手动进入下一课：推进锚点到 L2 / N2
    doc = ctx.put_position({
        "current_chapter_id": ctx.lesson_l2,
        "current_lesson_id": ctx.lesson_l2,
        "current_group_id": n2,
    })["data"]
    assert doc.get("current_chapter_id") == ctx.lesson_l2
    assert doc.get("current_group_id") == n2

    # 4) GET books 回读 + GET groups(L2) 定位
    book = next(x for x in ctx.get_books()["data"]["books"]
                if x.get("textbook_id") == ctx.textbook_id)
    assert book.get("current_chapter_id") == ctx.lesson_l2
    assert book.get("current_group_id") == n2
    groups_l2_after = ctx.get_groups(ctx.lesson_l2)["data"]["groups"]
    assert _first_unmastered_group_id(groups_l2_after) == n2
    print(f"       → L1 通关完成；锚点已推进 L2={ctx.lesson_l2}, N2={n2}；重进任务卡将落 N2")


@case("K7", "次日重进自动定位下一任务（books→lessons→groups 同构读序列）")
def k7(ctx: Ctx):
    if not ctx.lesson_l2:
        case_skip("K7", "次日重进自动定位下一任务（books→lessons→groups 同构读序列）",
                  "无可用下一课 L2，跳过")
        return
    book = next(x for x in ctx.get_books()["data"]["books"]
                if x.get("textbook_id") == ctx.textbook_id)
    a_lesson = book.get("current_lesson_id") or book.get("current_chapter_id")
    a_group = book.get("current_group_id")
    assert a_lesson == ctx.lesson_l2, f"锚点课 {a_lesson} != L2({ctx.lesson_l2})（锚点被服务端改动？）"
    assert a_group, "锚点缺 current_group_id"

    # 客户端 buildColdStartTarget：锚点课须能在课列表命中
    lessons_body = ctx.get_lessons()
    lessons = lessons_body["data"]["lessons"]
    found = next((lsn for lsn in lessons if lsn.get("lesson_id") == a_lesson), None)
    assert found, f"锚点课 {a_lesson} 不在课列表（冷启动无法命中）"
    pct = (found.get("progress") or {}).get("overall_percent") or 0
    assert pct < 100, f"L2 overall_percent={pct} 已达 100，次日将 again allCleared"

    first = _first_unmastered_group_id(ctx.get_groups(a_lesson)["data"]["groups"])
    assert first == a_group, f"次日定位组 {first} != 锚点组 {a_group}"
    print(f"       → 次日重读：锚点课 {a_lesson} / 锚点组 {a_group} 即下一任务（overall={pct}%）")


@case("K8", "边界：最后一课通关、无下一课（接口稳定、无脏锚点）")
def k8(ctx: Ctx):
    assert ctx.lesson_last
    ctx.put_position({
        "current_chapter_id": ctx.lesson_last,
        "current_lesson_id": ctx.lesson_last,
    })
    ctx.master_all_groups(ctx.lesson_last)
    groups = ctx.get_groups(ctx.lesson_last)["data"]["groups"]
    assert _first_unmastered_group_id(groups) is None, "最后一课仍存在未掌握组"

    lessons = ctx.get_lessons()["data"]["lessons"]
    last_row = next((lsn for lsn in lessons if lsn.get("lesson_id") == ctx.lesson_last), None)
    assert last_row is not None
    pct = (last_row.get("progress") or {}).get("overall_percent") or 0
    assert pct == 100, f"最后一课 overall_percent={pct} != 100"

    # 无下一课场景下 GET books 仍正常返回，锚点停在最后一课（不产生越界锚点）
    books = ctx.get_books()["data"]["books"]
    book = next(x for x in books if x.get("textbook_id") == ctx.textbook_id)
    assert book.get("current_lesson_id") == ctx.lesson_last
    print(f"       → 最后一课 {ctx.lesson_last} overall=100，无下一课；books 锚点正常停驻")


@case("K9", "异常与参数校验（400 / 404 / 空数组 / data:null）")
def k9(ctx: Ctx):
    # 9.1 groups 缺 scholar_id → 400
    r = ctx.client.get(
        f"{ctx.base_url}/tracking/textbooks/{ctx.textbook_id}/lessons/{ctx.lesson_l1}/groups"
    )
    assert r.status_code == 400, f"缺 scholar_id 期望 400，实际 {r.status_code}"

    # 9.2 lesson 不存在 → 404
    r = ctx.client.get(
        f"{ctx.base_url}/tracking/textbooks/{ctx.textbook_id}/lessons/not_exist_lesson/groups",
        params={"scholar_id": ctx.scholar_id},
    )
    assert r.status_code == 404, f"不存在 lesson 期望 404，实际 {r.status_code}"

    # 9.3 PUT position 空 body → 400
    r = ctx.client.put(
        f"{ctx.base_url}/scholar/{ctx.scholar_id}/books/{ctx.textbook_id}/position",
        json={},
    )
    assert r.status_code == 400, f"空 body 期望 400，实际 {r.status_code}"

    # 9.4 无关联学者 GET books → 200 空数组
    nobody = f"imm_nobody_{int(time.time() * 1000)}"
    r = ctx.client.get(
        f"{ctx.base_url}/scholar/{nobody}/books", params={"subject_type": "english"}
    )
    body = _require_ok(r, "GET books(无关联学者)")
    assert body["data"]["books"] == [], "无关联学者 books 应为空数组"

    # 9.5 immersive-progress GET 无记录 → 200 data:null
    r = ctx.client.get(
        f"{ctx.base_url}/scholar/{ctx.scholar_id}/textbooks/{ctx.textbook_id}/groups/{ctx.g1 or 'g_none'}/immersive-progress"
    )
    body = _require_ok(r, "GET immersive-progress(无记录)")
    assert body["data"] is None, "无记录应返回 data:null"


@case("K10", "immersive-progress 明细 CRUD（写入/覆盖/幂等删除清理）")
def k10(ctx: Ctx):
    gid = ctx.g1
    assert gid
    sents = _group_sentences(ctx._groups_of(ctx.lesson_l1), gid)
    s1 = (sents[0] if sents else {}).get("sentence_id")
    s2 = (sents[1] if len(sents) > 1 else sents[0] or {}).get("sentence_id")
    assert s1, "G1 无句子，无法构造断点明细"
    s2 = s2 or s1

    base = f"{ctx.base_url}/scholar/{ctx.scholar_id}/textbooks/{ctx.textbook_id}/groups/{gid}/immersive-progress"

    def _get():
        r = ctx.client.get(base)
        return _require_ok(r, "GET immersive-progress")

    # 初始无记录 → data:null
    assert _get()["data"] is None, "初始应无断点记录"

    def _put(sentence_id: str, step: str):
        r = ctx.client.put(base, json={
            "version": 2,
            "scholar_id": ctx.scholar_id,
            "textbook_id": ctx.textbook_id,
            "group_id": gid,
            "sentence_id": sentence_id,
            "challenge_active": True,
            "saved_at": _now_ms(),
            "payload": {"step": step, "done": True},
        })
        return _require_ok(r, "PUT immersive-progress")

    doc = _put(s1, "ec_translation")["data"]
    assert doc.get("sentence_id") == s1 and doc.get("version") == 2
    assert doc.get("challenge_active") is True

    got = _get()["data"]
    assert got.get("sentence_id") == s1
    assert got.get("payload", {}).get("step") == "ec_translation"

    # last-write-wins 覆盖
    _put(s2, "listening")
    got = _get()["data"]
    assert got.get("sentence_id") == s2, "覆盖式 upsert 未生效"
    assert got.get("payload", {}).get("step") == "listening"

    # 删除 + 幂等
    r = ctx.client.delete(base)
    body = _require_ok(r, "DELETE immersive-progress")
    assert body["data"].get("deleted") is True
    r = ctx.client.delete(base)
    body = _require_ok(r, "DELETE immersive-progress(幂等)")
    assert body["data"].get("deleted") is False
    assert _get()["data"] is None, "删除后应回到 data:null"

    # 异常：version 缺失 → 400；body 主键与路径不一致 → 400
    r = ctx.client.put(base, json={
        "scholar_id": ctx.scholar_id, "textbook_id": ctx.textbook_id,
        "group_id": gid, "sentence_id": s1,
        "challenge_active": True, "saved_at": _now_ms(),
        "payload": {"a": 1},
    })
    assert r.status_code == 400, f"缺 version 期望 400，实际 {r.status_code}"
    r = ctx.client.put(base, json={
        "version": 2, "scholar_id": "someone_else",
        "textbook_id": ctx.textbook_id, "group_id": gid,
        "sentence_id": s1, "challenge_active": True,
        "saved_at": _now_ms(), "payload": {"a": 1},
    })
    assert r.status_code == 400, f"主键不一致期望 400，实际 {r.status_code}"
    print(f"       → 断点明细 CRUD + 幂等清理完成（G1={gid}）")


# ---------------------------------------------------------------------------
# 服务拉起与汇总
# ---------------------------------------------------------------------------


def ensure_server(base_url: str, port: int, autostart: bool) -> subprocess.Popen | None:
    try:
        r = httpx.get(f"{base_url}/health", timeout=3)
        if r.status_code == 200:
            logger.info("服务已就绪: %s/health", base_url)
            return None
    except Exception:
        pass
    if not autostart:
        sys.exit(
            f"[ERROR] {base_url} 不可达。请先运行 `python main.py`，"
            f"或加 --port 让脚本自动拉起服务。"
        )
    logger.info("服务未启动，自动拉起: python -m uvicorn main:app --port %s", port)
    log_path = Path("/tmp/scholar_immersive_verify_server.log")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "127.0.0.1", "--port", str(port),
        ],
        cwd=str(HERE),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        if proc.poll() is not None:
            sys.exit(f"[ERROR] uvicorn 启动失败（退出码 {proc.returncode}），日志见 {log_path}")
        try:
            r = httpx.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                logger.info("服务已拉起: %s（日志: %s）", base_url, log_path)
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.kill()
    sys.exit(f"[ERROR] uvicorn 30s 内未就绪，日志见 {log_path}")


def print_summary() -> None:
    print("\n===== 汇总 =====")
    passed = sum(1 for x in _results if x["status"] == "PASS")
    failed = sum(1 for x in _results if x["status"] == "FAIL")
    skipped = sum(1 for x in _results if x["status"] == "SKIP")
    for x in _results:
        if x["status"] != "PASS":
            print(f"  [{x['status']}] {x['case']} {x['title']} -- {x['detail']}")
    print(f"  PASS={passed}  FAIL={failed}  SKIP={skipped}  (共 {len(_results)} 条)")
    if failed:
        print("  结论：存在失败用例（exit 1）")
    else:
        print("  结论：接口链路通过（exit 0）")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="沉浸式学习页任务进度接口验证（用例文档 docs_v1《沉浸式学习页任务进度接口测试用例》K1~K10）"
    )
    parser.add_argument("--base-url", default=None, help="服务地址（默认 http://127.0.0.1:{port}）")
    parser.add_argument("--port", type=int, default=BASE_URL_DEFAULT_PORT,
                        help="本地服务端口（服务未启动时可自动拉起，默认 8080）")
    parser.add_argument("--no-autostart", action="store_true", help="服务不可达时不自动拉起")
    parser.add_argument("--keep-server", action="store_true", help="自拉起的服务测试完不关闭")
    parser.add_argument("--scholar-id", default="",
                        help="隔离学者 ID（缺省自动生成一次性 imm_verify_*）")
    parser.add_argument("--textbook-id", default="",
                        help="英语教材 ID（缺省自动发现 GET /textbook?subject_type=english 第一本）")
    args = parser.parse_args()

    port = args.port
    base_url = args.base_url or f"http://127.0.0.1:{port}"
    proc = ensure_server(base_url, port, autostart=not args.no_autostart)

    # 隔离学者：一次性随机（保证无历史、可重复跑）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = f"{random.randint(0, 9999):04d}"
    scholar_id = args.scholar_id or f"imm_verify_{ts}_{rand}"

    try:
        with httpx.Client(timeout=30.0) as client:
            textbook_id = args.textbook_id or discover_textbook(client, base_url)
            ctx = Ctx(client, base_url, scholar_id, textbook_id)
            setup_lessons(ctx)

            # 数据准备期错误（无教材/无课）→ 不进入用例，直接失败退出
            if not ctx.lesson_l1:
                sys.exit("[ERROR] 环境数据不足：无可用课/组，无法执行用例")

            # K1 冒烟需 L2；若缺 L2 仍先跑单课用例
            k1(ctx)
            k2(ctx)
            k3(ctx)
            k4(ctx)
            k5(ctx)
            k6(ctx)
            k7(ctx)
            k8(ctx)
            k9(ctx)
            k10(ctx)

        print_summary()
    finally:
        if proc is not None and not args.keep_server:
            proc.terminate()
            logger.info("已关闭自动拉起的本地服务")

    if any(x["status"] == "FAIL" for x in _results):
        sys.exit(1)


if __name__ == "__main__":
    main()
