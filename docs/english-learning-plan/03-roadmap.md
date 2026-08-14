# 第三步:落地路线 —— 与项目数据模型衔接

> 说明:本项目为 CloudBase 文档数据库 + FastAPI 后端。以下按 **P0 → P1** 的实施顺序,给出数据模型、接口设计与实现要点,与现有代码(services/tracking_stats.py、services/routes_tracking.py)直接衔接。

---

## 3.1 现有数据模型回顾

| 集合 | 关键字段 | 现状 |
|------|---------|------|
| `sentence` | `sentence_id, unit_id, paragraph_id, text, translation, level, keywords, index` | 已有 |
| `unit` | `unit_id, title, total_sentences` | 已有 |
| `learning_mastery_tracking` | `scholar_id, text_book_id, sentence_id, time_spent, status, score, mastery` | 已有 |

## 3.2 P0 实施计划(本轮)

### 3.2.1 学习时间 ✅ 已完成

- **接口**:`POST /tracking/stats`(已实现)
- **返回**:`summary.total_time_spent`(秒)、`total_time_spent_display`(可读格式)
- **后续增强**:按日/周/月聚合 → 新增 `GET /tracking/stats/time`(时间维度统计),供图表使用

### 3.2.2 学习进度 ✅ 已完成

- **接口**:`POST /tracking/stats`(已实现)
- **返回**:`summary.textbook_progress`、`units[]`(unit 级进度)、`sentences[]`(sentence 级进度)
- **前端展示建议**:环形进度(教材) + 列表(单元) + 展开(句子),已学/未学用颜色区分

### 3.2.3 学习次数 —— 待实现(本轮补上)

**数据层**:`learning_mastery_tracking` 增加字段(插入时维护):

```json
{
  "study_count": 1,            // 该句子累计学习次数
  "last_study_time": 1718000000 // 上次学习时间戳
}
```

> 说明:客户端每次上报学习记录时,若同一 `(scholar_id, sentence_id)` 记录已存在,则 `study_count+1`、更新 `last_study_time`;若不存在则新建。可复用现有 CRUD 的 upsert 能力。

**接口**:扩展 `POST /tracking/stats` 返回:

```json
{
  "summary": {
    "total_study_count": 37,
    "avg_study_count_per_sentence": 3.08
  },
  "sentences": [
    { "sentence_id": "s1", "study_count": 5, "last_study_time": 1718000000, "...": "..." }
  ]
}
```

**实现要点**:在 `services/tracking_stats.py` 的 `merge_records` 中累计 `study_count`(同句多条记录则相加),并取 `last_study_time` 最大值。

### 3.2.4 单元测试

参照现有 `tests/test_tracking_stats.py` 模式,为「学习次数」新增用例:
- 同一句子多次记录 → `study_count` 累加
- `last_study_time` 取最新
- 未知句子不计入

## 3.3 P1 实施计划(下一迭代)

### 3.3.1 滚动学习调度器 ⭐(差异化核心)

**目标**:根据遗忘曲线,生成「今日复习队列」——把易忘的句子优先推给用户。

**调度算法(简版,可在后端实现)**:

```
优先级分 = w1 × (1 - mastery)          # 掌握度越低越优先
         + w2 × (days_since_last / 期望间隔)  # 距上次学习越久越优先
         + w3 × (is_learned ? 0 : 1)    # 未学句子优先
         + w4 × (study_count 折扣)        # 学习次数越多,间隔可以越长
```

- 期望间隔按间隔重复经典策略:第 1 次学后 1 天、第 2 次后 3 天、第 3 次后 7 天、之后 14/30 天…
- 掌握度来源:`score/100` 或 `mastery`,优先 `score`

**接口设计**:

```
POST /tracking/review-plan
{
  "scholar_id": "s1",
  "text_book_id": "tb1",
  "max_count": 20,          // 今日队列数量,默认 20
  "include_unlearned": true  // 是否混入未学新句子
}
→ { "data": { "review_list": [ {"sentence_id","text","translation","mastery","days_since_last"} ] } }
```

**数据依赖**:需要 `last_study_time`(3.2.3 新增)与 `updated_at` 时间戳。

### 3.3.2 薄弱点推荐

- 聚合 `score < 60` 或 `mastery < 0.6` 的句子,按分数升序生成补漏清单
- 可作为 `review-plan` 的可选参数:`include_weak=true`

### 3.3.3 连续打卡/日历热力图

- 按 `last_study_time` 聚合成天粒度,统计连续学习天数
- 接口:`GET /tracking/streak?scholar_id=xxx`

## 3.4 实施顺序与工期估算

| 阶段 | 内容 | 工期 |
|------|------|------|
| 1. P0-学习时间 | 已完成 | — |
| 2. P0-学习进度 | 已完成 | — |
| 3. P0-学习次数 | 扩展 tracking_stats + 测试 | 0.5 天 |
| 4. P1-滚动学习调度器 | review-plan 接口 + 调度算法 + 测试 | 2-3 天 |
| 5. P1-薄弱点/打卡 | 聚合接口 | 1 天 |

## 3.5 风险与注意事项

1. **数据完整性问题**:`study_count` 对存量数据需要做一次回填(按现有记录计数)。
2. **时间戳统一**:确认客户端上报是否携带 `time` 字段,否则用服务端 `updated_at`。
3. **调度算法冷启动**:新用户无历史记录,`review-plan` 应默认返回未学句子 + 最新学习的句子。
4. **分页拉取成本**:统计需要拉取整个教材的 sentence 集合,现有实现已分页,教材很大时需考虑缓存。
