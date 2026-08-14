"""管理接口：教材批量删除 + 管理页面"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from services.dependencies import get_db

logger = logging.getLogger("scholar-admin.routes.admin")
router = APIRouter(tags=["管理"])

# ==================== 删除接口 ====================

COLLECTIONS_TO_CLEANUP = [
    "textbook",
    "unit",
    "paragraph",
    "sentence",
    "learning_mastery_tracking",
]


@router.post("/admin/textbook/cleanup")
async def cleanup_textbooks(data: dict):
    """根据 text_book_id 列表批量删除教材及其关联数据

    请求体：
    {
      "text_book_ids": ["tb_xxxxx", "tb_yyyyy"]
    }

    会依次从 textbook / unit / paragraph / sentence / learning_mastery_tracking
    5 张表中删除所有关联记录。
    """
    text_book_ids: list[str] = data.get("text_book_ids", [])
    if not text_book_ids:
        raise HTTPException(status_code=400, detail="缺少 text_book_ids 参数")

    db = get_db()
    results: dict[str, int] = {}
    total = 0

    # ------------------- 1. 先查 sentence 的 ID（用于 learning_mastery_tracking 间接关联） -------------------
    sentence_ids: list[str] = []
    try:
        for i in range(0, len(text_book_ids), 100):
            batch = text_book_ids[i : i + 100]
            resp = await db.query(
                collection="sentence",
                where={"text_book_id": {"$in": batch}},
                limit=5000,
                select={"sentence_id": 1},
            )
            for rec in resp.get("records", []):
                if rec.get("sentence_id"):
                    sentence_ids.append(rec["sentence_id"])
    except Exception as e:
        logger.warning(f"[cleanup] 查询 sentence_id 失败: {e}")

    # ------------------- 2. 删除 learning_mastery_tracking（通过 sentence_id 批量删除） -------------------
    try:
        if sentence_ids:
            tm_cnt = 0
            for i in range(0, len(sentence_ids), 100):
                batch = sentence_ids[i : i + 100]
                resp = await db.delete(
                    collection="learning_mastery_tracking",
                    where={"sentence_id": {"$in": batch}},
                    multi=True,
                )
                tm_cnt += resp.get("deleted_count", 0)
            results["learning_mastery_tracking"] = tm_cnt
            total += tm_cnt
            logger.info(f"[cleanup] learning_mastery_tracking: 删除 {tm_cnt} 条")
        else:
            results["learning_mastery_tracking"] = 0
    except Exception as e:
        logger.warning(f"[cleanup] learning_mastery_tracking 删除异常: {e}")
        results["learning_mastery_tracking"] = -1

    # ------------------- 3. 删除 textbook / unit / paragraph / sentence -------------------
    for coll in ["textbook", "unit", "paragraph", "sentence"]:
        try:
            if coll == "textbook":
                resp = await db.delete(
                    collection=coll,
                    where={"_id": {"$in": text_book_ids}},
                    multi=True,
                )
            else:
                resp = await db.delete(
                    collection=coll,
                    where={"text_book_id": {"$in": text_book_ids}},
                    multi=True,
                )
            cnt = resp.get("deleted_count", 0)
            results[coll] = cnt
            total += cnt
            logger.info(f"[cleanup] {coll}: 删除 {cnt} 条")
        except Exception as e:
            logger.warning(f"[cleanup] {coll} 删除异常: {e}")
            results[coll] = -1

    return {
        "success": True,
        "deleted_total": total,
        "detail": results,
        "text_book_ids": text_book_ids,
    }


# ==================== 合并接口 ====================


@router.post("/admin/textbook/merge")
async def merge_textbooks(data: dict):
    """将多个教材合并为一个 — 所选教材的 unit/sentence 全部转移到第一个教材下

    请求体：
    {
      "text_book_ids": ["tb_xxxxx", "tb_yyyyy", "tb_zzzzz"]
    }

    合并规则：
    - 取第一个 ID 作为保留教材（survivor），其余教材将被删除
    - 被合并教材的所有 unit / paragraph / sentence 的 text_book_id 更新为保留教材的 ID
    - learning_mastery_tracking 通过 sentence_id 间接关联，无需修改

    返回：
    {
      "success": true,
      "survivor_id": "tb_xxxxx",
      "merged_ids": ["tb_yyyyy", "tb_zzzzz"],
      "detail": { "unit": 10, "paragraph": 12, "sentence": 80, "deleted_textbook": 2 }
    }
    """
    text_book_ids: list[str] = data.get("text_book_ids", [])
    if len(text_book_ids) < 2:
        raise HTTPException(status_code=400, detail="至少需要选择 2 本教材才能合并")

    survivor_id = text_book_ids[0]
    merge_ids = text_book_ids[1:]  # 将被合并后删除的教材
    now = int(time.time())

    db = get_db()
    results: dict[str, int] = {}

    # 1. 验证保留教材是否存在
    survivor_resp = await db.query(
        collection="textbook",
        where={"_id": survivor_id},
        limit=1,
    )
    if not survivor_resp.get("records"):
        raise HTTPException(status_code=404, detail=f"教材不存在: {survivor_id}")

    try:
        # 2. 迁移 unit / paragraph / sentence 的 text_book_id
        for coll in ["unit", "paragraph", "sentence"]:
            batch_cnt = 0
            # CloudBase NoSQL 批量更新：遍历每个被合并的 textbook，将其下的记录迁移
            for mid in merge_ids:
                # 查询该 merge_id 下当前集合的所有记录
                query_resp = await db.query(
                    collection=coll,
                    where={"text_book_id": mid},
                    limit=5000,
                )
                records_to_update = query_resp.get("records", [])
                logger.info(
                    f"[merge] {coll} mid={mid} "
                    f"query_resp_keys={list(query_resp.keys())} "
                    f"records_type={type(records_to_update).__name__} "
                    f"records_len={getattr(records_to_update, '__len__', lambda: 0)()}"
                )
                # 防御：CloudBase 返回的 records 可能是 dict / str / list
                if isinstance(records_to_update, dict):
                    logger.info(
                        f"[merge] {coll} mid={mid} records is dict, "
                        f"keys={list(records_to_update.keys())[:10]} "
                        f"sample_v={repr({k: type(v).__name__ for k, v in list(records_to_update.items())[:3]})}"
                    )
                    records_to_update = [records_to_update]
                elif isinstance(records_to_update, str):
                    logger.info(
                        f"[merge] {coll} mid={mid} records is str, "
                        f"len={len(records_to_update)} preview={repr(records_to_update[:200])}"
                    )
                    try:
                        records_to_update = json.loads(records_to_update) if records_to_update else []
                        logger.info(
                            f"[merge] {coll} mid={mid} after str decode → "
                            f"type={type(records_to_update).__name__}"
                        )
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.error(f"[merge] {coll} mid={mid} JSON decode fail: {e}")
                        records_to_update = []
                    if isinstance(records_to_update, dict):
                        records_to_update = [records_to_update]
                if not isinstance(records_to_update, list):
                    logger.error(
                        f"[merge] {coll} mid={mid} records is unexpected "
                        f"type={type(records_to_update).__name__}, "
                        f"value={repr(records_to_update)[:300]}"
                    )
                    records_to_update = []
                for rec in records_to_update:
                    # 单个元素也可能是 JSON 字符串
                    if isinstance(rec, str):
                        try:
                            rec = json.loads(rec)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                f"[merge] {coll} mid={mid} elem str not JSON: "
                                f"{repr(rec[:100])}"
                            )
                            continue
                    if not isinstance(rec, dict):
                        logger.warning(
                            f"[merge] {coll} mid={mid} elem not dict: "
                            f"type={type(rec).__name__} val={repr(rec)[:100]}"
                        )
                        continue
                    rec_id = rec.get("_id")
                    if rec_id:
                        await db.update(
                            collection=coll,
                            where={"_id": rec_id},
                            data={"$set": {"text_book_id": survivor_id}},
                            multi=False,
                        )
                        batch_cnt += 1
            results[coll] = batch_cnt
            logger.info(
                f"[merge] {coll}: 迁移 {batch_cnt} 条 → {survivor_id}"
            )

        # 3. 删除被合并的 textbook 记录
        delete_resp = await db.delete(
            collection="textbook",
            where={"_id": {"$in": merge_ids}},
            multi=True,
        )
        deleted_cnt = delete_resp.get("deleted_count", 0)
        results["deleted_textbook"] = deleted_cnt
        logger.info(f"[merge] textbook: 删除 {deleted_cnt} 条合并源教材")

        # 4. 更新保留教材的 updated_at
        await db.update(
            collection="textbook",
            where={"_id": survivor_id},
            data={"$set": {"updated_at": now}},
            multi=False,
        )

        return {
            "success": True,
            "survivor_id": survivor_id,
            "merged_ids": merge_ids,
            "detail": results,
        }

    except Exception as e:
        logger.error(f"[merge] 合并失败: {e}")
        raise HTTPException(status_code=500, detail=f"合并失败: {str(e)}")


# ==================== 管理页面 ====================


MANAGE_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>教材管理</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f7fa; color: #333; padding: 24px;
  }
  .container { max-width: 900px; margin: 0 auto; }
  h1 { font-size: 22px; margin-bottom: 8px; }
  .sub { color: #888; font-size: 13px; margin-bottom: 20px; }
  .toolbar {
    display: flex; align-items: center; gap: 12px; margin-bottom: 16px;
    flex-wrap: wrap;
  }
  .btn {
    border: none; border-radius: 6px; padding: 8px 18px; font-size: 14px;
    cursor: pointer; font-weight: 500;
  }
  .btn-danger { background: #e74c3c; color: #fff; }
  .btn-danger:hover { background: #c0392b; }
  .btn-danger:disabled { background: #e0e0e0; color: #999; cursor: not-allowed; }
  .btn-outline { background: #fff; color: #555; border: 1px solid #d0d5dd; }
  .btn-outline:hover { background: #f0f2f5; }
  .btn-primary { background: #4a6cf7; color: #fff; }
  .btn-primary:hover { background: #3b5de7; }
  .card {
    background: #fff; border-radius: 10px; border: 1px solid #e8ecf1;
    margin-bottom: 16px; overflow: hidden;
  }
  .card-header {
    display: flex; align-items: center; gap: 10px; padding: 14px 18px;
    border-bottom: 1px solid #f0f0f0;
  }
  .card-header label { font-size: 15px; cursor: pointer; }
  .card-body {
    padding: 14px 18px; display: flex; gap: 24px; flex-wrap: wrap;
  }
  .card-body span { font-size: 13px; color: #666; }
  .card-body .tag {
    display: inline-block; background: #eef2ff; color: #4a6cf7;
    padding: 2px 8px; border-radius: 4px; font-size: 12px;
  }
  .status-msg {
    margin-top: 12px; padding: 10px 16px; border-radius: 6px; font-size: 14px;
    display: none;
  }
  .status-msg.success { background: #e7f7ed; color: #1e7e34; display: block; }
  .status-msg.error { background: #fdecea; color: #c0392b; display: block; }
  .loading { text-align:center; padding:40px; color:#999; }
  .empty { text-align:center; padding:40px; color:#999; font-size:14px; }
  input[type=checkbox] { width:18px; height:18px; cursor:pointer; accent-color:#4a6cf7; }
  .count { font-size:13px; color:#888; margin-left:auto; }
</style>
</head>
<body>
<div class="container">
  <h1>📚 教材管理</h1>
  <p class="sub">管理已生成的教材数据，选中教材后可一键删除该教材及其关联的所有 unit / sentence / 学习追踪记录。</p>

  <div class="toolbar">
    <button class="btn btn-outline" onclick="selectAll()">全选</button>
    <button class="btn btn-outline" onclick="selectNone()">取消全选</button>
    <button class="btn btn-danger" id="deleteBtn" onclick="doDelete()" disabled>
      删除选中教材
    </button>
    <button class="btn btn-primary" id="mergeBtn" onclick="doMerge()" disabled>
      合并选中教材
    </button>
    <button class="btn btn-outline" onclick="loadList()">刷新列表</button>
    <span class="count" id="countInfo"></span>
  </div>

  <div class="status-msg" id="msg"></div>

  <div id="list">
    <div class="loading">加载中...</div>
  </div>
</div>

<script>
const API = '/textbook';

let textbooks = [];
let selected = new Set();

async function loadList() {
  const list = document.getElementById('list');
  list.innerHTML = '<div class="loading">加载中...</div>';
  try {
    const resp = await fetch(API);
    const data = await resp.json();
    textbooks = data.records || [];
    selected.clear();
    render();
    updateToolbar();
  } catch (e) {
    list.innerHTML = '<div class="empty">加载失败: ' + e.message + '</div>';
  }
}

function render() {
  const list = document.getElementById('list');
  if (textbooks.length === 0) {
    list.innerHTML = '<div class="empty">暂未创建任何教材</div>';
    return;
  }
  list.innerHTML = textbooks.map((tb, i) => {
    const bid = tb._id || '';
    const title = tb.title || '未命名';
    const grade = tb.grade || '';
    const edition = tb.edition || '';
    const createdAt = tb.created_at
      ? new Date(tb.created_at * 1000).toLocaleDateString('zh-CN')
      : '';
    return `<div class="card">
      <div class="card-header">
        <input type="checkbox" id="cb${i}" value="${bid}"
               onchange="onCheck('${bid}', this.checked)">
        <label for="cb${i}">${title}</label>
      </div>
      <div class="card-body">
        <span>🆔 <code>${bid}</code></span>
        ${grade ? `<span class="tag">${grade}</span>` : ''}
        ${edition ? `<span>📖 ${edition}</span>` : ''}
        ${createdAt ? `<span>📅 ${createdAt}</span>` : ''}
      </div>
    </div>`;
  }).join('');
}

function onCheck(id, checked) {
  if (checked) selected.add(id);
  else selected.delete(id);
  updateToolbar();
}

function selectAll() {
  textbooks.forEach(t => selected.add(t._id || ''));
  render();
  textbooks.forEach((t, i) => document.getElementById('cb'+i).checked = true);
  updateToolbar();
}

function selectNone() {
  selected.clear();
  textbooks.forEach((t, i) => {
    const cb = document.getElementById('cb'+i);
    if (cb) cb.checked = false;
  });
  updateToolbar();
}

function updateToolbar() {
  const delBtn = document.getElementById('deleteBtn');
  delBtn.disabled = selected.size === 0;
  delBtn.textContent = selected.size === 0
    ? '删除选中教材'
    : `删除选中教材 (${selected.size})`;

  const mergeBtn = document.getElementById('mergeBtn');
  mergeBtn.disabled = selected.size < 2;
  mergeBtn.textContent = selected.size < 2
    ? '合并选中教材'
    : `合并选中教材 (${selected.size})`;

  document.getElementById('countInfo').textContent =
    `共 ${textbooks.length} 本教材，已选 ${selected.size} 本`;
}

async function doDelete() {
  if (selected.size === 0) return;
  const ids = [...selected];
  const ok = confirm(
    `确认删除以下 ${ids.length} 本教材及其关联的所有数据？\\n\\n` +
    `删除范围：textbook / unit / paragraph / sentence / learning_mastery_tracking\\n\\n` +
    `此操作不可撤销！`
  );
  if (!ok) return;

  const btn = document.getElementById('deleteBtn');
  btn.disabled = true;
  btn.textContent = '删除中...';
  const msg = document.getElementById('msg');
  msg.className = 'status-msg';

  try {
    const resp = await fetch('/admin/textbook/cleanup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text_book_ids: ids }),
    });
    const data = await resp.json();
    if (data.success) {
      msg.className = 'status-msg success';
      msg.innerHTML =
        `✅ 删除成功！共删除 ${data.deleted_total} 条记录<br>` +
        Object.entries(data.detail)
          .map(([k,v]) => `${k}: ${v >= 0 ? v + ' 条' : '失败'}`)
          .join(' &nbsp;|&nbsp; ');
      loadList();
    } else {
      msg.className = 'status-msg error';
      msg.textContent = '删除失败: ' + (data.detail || '未知错误');
    }
  } catch (e) {
    msg.className = 'status-msg error';
    msg.textContent = '请求失败: ' + e.message;
  }
  btn.disabled = false;
  btn.textContent = `删除选中教材 (${selected.size})`;
}

async function doMerge() {
  if (selected.size < 2) return;
  const ids = [...selected];
  const titleList = ids.map(id => {
    const tb = textbooks.find(t => t._id === id);
    return tb ? tb.title : id;
  });
  const ok = confirm(
    `确认将以下 ${ids.length} 本教材合并为一本？\\n\\n` +
    `保留教材（取第一个）：${titleList[0]}\\n` +
    `被合并教材（将被删除）：\\n` +
    titleList.slice(1).map(t => `  - ${t}`).join('\\n') +
    `\\n\\n合并后，所有 unit / sentence 将归属到保留教材下，被合并教材本身将被删除。`
  );
  if (!ok) return;

  const mergeBtn = document.getElementById('mergeBtn');
  const origText = mergeBtn.textContent;
  mergeBtn.disabled = true;
  mergeBtn.textContent = '合并中...';
  const msg = document.getElementById('msg');
  msg.className = 'status-msg';

  try {
    const resp = await fetch('/admin/textbook/merge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text_book_ids: ids }),
    });
    const data = await resp.json();
    if (data.success) {
      msg.className = 'status-msg success';
      msg.innerHTML =
        `✅ 合并成功！教材已合并至 <code>${data.survivor_id}</code><br>` +
        Object.entries(data.detail)
          .map(([k,v]) => `${k}: ${v} 条`)
          .join(' &nbsp;|&nbsp; ');
      loadList();
    } else {
      msg.className = 'status-msg error';
      msg.textContent = '合并失败: ' + (data.detail || '未知错误');
    }
  } catch (e) {
    msg.className = 'status-msg error';
    msg.textContent = '请求失败: ' + e.message;
  }
  mergeBtn.disabled = false;
  mergeBtn.textContent = origText;
}

// 初始化
loadList();
</script>
</body>
</html>"""


@router.get("/admin/textbook", response_class=HTMLResponse)
async def admin_textbook_page():
    """教材管理页面 — 展示教材列表，支持选中并批量删除"""
    return HTMLResponse(content=MANAGE_PAGE)
