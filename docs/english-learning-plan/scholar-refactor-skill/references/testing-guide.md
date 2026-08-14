# 测试指南(单元测试 + 集成测试)

> 本文件是 scholar-refactor-skill 的**测试体系规范**。每个 Phase 的改动都必须伴随测试用例,
> 并同步更新本文档的「测试矩阵」,做到「增加功能的同时维护测试用例」。

## 1. 测试分层

| 层级 | 目录 | 测什么 | 依赖 | 典型对象 |
|------|------|--------|------|----------|
| 单元测试 | `tests/unit/` | 纯函数 / 算法 / 字段映射 / 差距脚本 | 无(不触网、不连库、不实例化客户端) | `_normalize_types`、滚动调度公式、聚合计算、状态枚举映射、`check_schema` 集合名识别 |
| 集成测试 | `tests/integration/` | 接口全链路(路由参数校验 → DB 交互 → 响应结构) | FakeDB + FastAPI TestClient | `POST /tracking/stats`、各新接口 |
| 测试替身 | `tests/fakes/` | 外部依赖的内存实现,接口签名与真实客户端对齐 | 无 | `FakeDB`(模拟 CloudBaseNoSQLClient) |

> 既有基线 `tests/test_tracking_stats.py` 自包含(自带 FakeDB + 单测 + 接口测试),保留不动;
> 后续新增测试一律使用 `tests/fakes/fake_db.py` + `tests/conftest.py` 的共享设施。

## 2. 目录结构

```
tests/
├── conftest.py              # 共享 fixtures:fake_db(种子数据)/ client(TestClient + FakeDB)
├── fakes/
│   ├── __init__.py
│   └── fake_db.py           # 通用内存 FakeDB:query/insert/update/delete/count,多集合、where 操作符、排序、分页、投影
├── unit/
│   ├── __init__.py
│   └── test_*.py            # 每个被测纯函数一个文件(如 test_database_normalize.py / test_check_schema.py)
├── integration/
│   ├── __init__.py
│   └── test_*.py            # 每个接口一组(如 test_tracking_endpoints.py)
└── test_tracking_stats.py   # 既有基线,保留(可选逐步并入 unit/integration)
```

## 3. 命名与编写约定

- 文件:`test_*.py`;类:`TestXxx`(对应被测功能/接口);方法:`test_xxx`(对应具体场景)。
- 单元测试:**不实例化** `CloudBaseNoSQLClient`(构造函数校验密钥,需环境变量);只调静态方法/纯函数。
- 集成测试:一律请求 `client` fixture(已把 `services.routes_tracking.get_db` 注入为 FakeDB);
  其他路由模块需在测试中追加 `monkeypatch.setattr("services.<module>.get_db", lambda: fake_db)`。
- 断言响应结构的关键字段,不要逐字相等;浮点用 `pytest.approx`。
- 需要某集合的种子数据时,在测试内用 `fake_db.add(collection, doc)` 补充,不污染其他用例(fixture 按函数级重建)。

## 4. 运行方式

```bash
python3 -m pytest tests/ -q            # 全量
python3 -m pytest tests/unit -q        # 单元测试
python3 -m pytest tests/integration -q # 集成测试
```

## 5. 测试矩阵(各 Phase 必须配套的测试)

> 执行到对应 Phase 时,新增下列测试;同时把「已覆盖」勾选,并把新接口/新函数追加到对应行。

| Phase | 功能改动 | 必须新增的测试 | 层级 |
|-------|----------|----------------|------|
| 0 | 基线确认 | 既有 `tests/test_tracking_stats.py` 全绿;新增的 unit/integration 模板通过 | 全部 |
| 1 | `textbook_v2`/`chapter`/`lesson`/`sentence_v2` 内容分层 | `models_content` 查询辅助与文档构建纯函数单测(`test_models_content.py` 14);幂等迁移脚本单测(`test_migrate_content_v2.py` 8);构建双写、视觉识别双写、`sentence_v2` 层级引用完整性、旧表未被修改 的集成测试(`test_content_v2_build.py` 5);差距脚本集合名识别(`test_check_schema.py` 9) | unit + integration |
| 2 | `skill`/`skill_state` + 状态上报接口 | 状态归一化(中文→英文)、复合键、滚动调度公式、upsert 幂等/attempt_count 累加/复合键冲突 的纯函数单测(`test_models_learning.py` 31);迁移幂等 + 旧表原样保留 + 字段映射(`test_migrate_learning.py` 8);`POST /tracking/state` 参数校验与落库、`GET /tracking/{scholar_id}` 优先 skill_state 回退旧表 的集成测试(`test_state_endpoints.py` 7) | unit + integration |
| 3 | `study_attempt`/`study_session` 事件上报 | 事件类型推断/状态归一/文档构建/只增不改/会话结算/会话隔离 单测(`test_events.py` 12);会话 start/end 参数校验、结算回填、重复会话互不干扰、无会话事件不计入 的集成测试(`test_session_endpoints.py` 7);`POST /tracking/state` 返回结构含 attempt 且事件追加落库(`test_state_endpoints.py` 新增 6) | unit + integration |
| 4 | `progress`/`mastery` 聚合 + 重构 tracking/stats | 聚合纯函数单测(句子/课/章/书逐级+分布/时长/复现/skill 过滤, `test_progress.py` 19);stats 接口集成测试(服务端聚合/兼容入口/校验/复现, `test_stats_aggregation.py` 12);旧 `test_tracking_stats.py` 保持通过(兼容层) | unit + integration |
| 5 | `scholar_book` 断点续学 | 复合键/文档构建/upsert 幂等/时长增量累加/结算回写/无教材不落库 的纯函数单测(`test_scholar_book.py` 12);首次加入/断点更新/重复幂等/参数校验/列表含进度/断点取回/多教材隔离/会话结算回写与累加 的集成测试(`test_scholar_book_endpoints.py` 12) | unit + integration |
| 6 | 迁移下线旧表 | 迁移核对逻辑单测;确认 `rg` 无旧字段引用;全量回归 + `check_schema.py` 差距归零。**清理顺序固定:先迁移核对 → 再全量回归通过 → 最后才删除/下线旧表(仅 Phase 6 末尾)** | 全部 |

## 6. 测试维护规则(强制)

1. **新增功能必须带测试**:新增集合 / 接口 / 纯函数时,必须同时新增对应单元或集成测试,
   并更新第 5 节矩阵与 `references/execution-guide.md` 对应 Phase 的验收标准。禁止只改功能不碰测试。
2. **行为变更先动测试**:修改已有行为时,先更新/新增测试再改实现(或同一提交内完成)。
3. **纯逻辑搬迁测试随行**:重构把逻辑从一个函数移到新模块时,测试随代码迁移,保证覆盖不降。
4. **FakeDB 接口扩展要自测**:FakeDB 新增能力(操作符/upsert 等)时,同步补充对 FakeDB 自身的单元测试
   (放 `tests/unit/test_fake_db.py`),并在本文件第 1 节注明。
5. **每次 Phase 结束全量回归**:`python3 -m pytest tests/ -q` 必须 0 失败(允许记录既有 warning),
   与 `scripts/check_schema.py` 的输出一起作为该 Phase 验收依据。
6. **测试也是代码**:遵守与业务代码相同的质量要求(lint 无错、命名清晰、可读)。
