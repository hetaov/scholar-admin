# e2e 测试基础设施重构 — T1.1 现状盘点清单

> 对应执行计划任务 1：搭建 e2e 测试基础设施（conftest 公共 fixtures：FakeDB + 外部调用 mock）
> 本文件为 T1.1 交付物，供 T1.2~T1.7 迁移时对照。

## 1. 基线结果

- 命令：`python -m pytest tests/ -q`
- 结果：**402 passed（0.78s），全绿** ✓
- 范围：`tests/integration/`（18 个文件 181 个测试）+ `tests/unit/`（其余）

## 2. 集成测试文件全量盘点（19 个文件，181 tests，4106 行）

| 文件 | tests | 行数 | client 构建 | get_db 注入方式 | seed/helper | 外部 mock / 假 Provider |
|---|---|---|---|---|---|---|
| test_tracking_endpoints.py | 2 | 37 | **conftest client**（模板，无自建） | conftest 注入 | — | — |
| test_main_routes.py | 1 | 41 | 直接 `main.app`（冒烟） | — | — | — |
| test_query_split_endpoints.py | 11 | 350 | `_client` | A: setattr `routes_tracking` | `_seed_content` `_seed_states` | — |
| test_calendar.py | 8 | 141 | `_client` | A: setattr `routes_tracking` | — | — |
| test_review_plan.py | 8 | 211 | `_client` | A: setattr `routes_tracking` | `_seed_content` | — |
| test_weakness_plan.py | 10 | 202 | `_client` | A: setattr `routes_tracking` | `_seed_content` | — |
| test_session_endpoints.py | 8 | 138 | `_client` | A: setattr `routes_state` | — | — |
| test_state_endpoints.py | 10 | 150 | `_client` | A: setattr `routes_state`+`routes_tracking` | — | — |
| test_scholar_book_endpoints.py | 12 | 344 | `_tracking_client`+`_state_client` 双 client | A: setattr ×2 | `_seed_content` | — |
| test_dialogue_task_routes.py | 15 | 326 | `_client` | A: setattr `routes_dialogue`+`dialogue_task` | `_seed_task` | `run_dialogue_task` `match_dialogue` |
| test_p2_features.py | 22 | 444 | `_tracking_client`+`_dialogue_client` 双 client | A: setattr ×3 | `_seed_badges`(类内) `_skill_state` | `run_dialogue_task` |
| test_routes_evaluation.py | 9 | 165 | `_client`+`_patch_db` | B: overrides(get_db) + A: setattr(`_call_judge`) | `_seed_attempt` `_seed_speech` | `_call_judge` |
| test_routes_training.py | 6 | 182 | `_client` | B: overrides(get_db) | `_seed_state` | `_call_judge` |
| test_routes_conversation.py | 10 | 164 | `_client` | B: overrides(get_db) | — | `_generate_reply` `_call_judge` |
| test_routes_eval.py | 11 | 191 | `_client(asr, model_output)` | B: overrides(get_asr_service) | — | `_call_volcano` + `FakeASR` |
| test_speech_eval.py | 14 | 271 | `_client(provider)` | B: overrides(get_db, get_speech_provider) | `_payload` | `FakeSpeechProvider`+变体 |
| test_tts.py | 20 | 336 | `_client(provider)` | B: overrides(get_db, get_tts_provider) | `_payload` | `FakeTtsProvider`+变体 |
| test_content_v2_build.py | 4 | 112 | 无 client（直接调函数） | A: setattr `routes_build` ×2 内联 | — | — |

## 3. 关键发现：get_db 注入方式差异（T1.3 核心依据）

存在两种不兼容的注入方式，必须分别处理：

- **方式 A（`monkeypatch.setattr("services.X.get_db", ...)`，模块属性替换）**：11 个文件。
  适用模块：`routes_tracking`、`routes_state`、`routes_dialogue`、`dialogue_task`、`routes_build`。
  原因：这些模块在调用时才从模块全局查 `get_db`，替换属性即生效。
- **方式 B（`app.dependency_overrides`，FastAPI 官方）**：6 个文件。
  适用模块：`routes_evaluation`、`routes_training`、`routes_conversation`、`routes_eval`、`routes_tts`、`routes_speech_eval`。
  原因：`Depends()` 在路由定义期已捕获原函数对象，运行时不再从模块查找，setattr 无效（代码注释已注明）。
- **混用文件**：`test_routes_evaluation.py` 同时用两种（get_db 走 B，_call_judge 走 A）。

## 4. 重复模式汇总

### 4.1 重复的 seed / helper（T1.5 收敛对象）
- `_seed_content`：4 处（query_split / review_plan / weakness_plan / scholar_book），内容为 chapter→lesson→sentence_v2 层级，彼此略有差异（需统一为可配置参数）
- `_seed_attempt` / `_seed_speech`：routes_evaluation（唯一）
- `_seed_states`：query_split（唯一）
- `_seed_state`：routes_training（唯一）
- `_seed_task`：dialogue_task_routes（唯一）
- `_seed_badges`：p2_features（类内方法，唯一）
- `_payload`：speech_eval / tts 各自定义（2 处，结构相同）

### 4.2 重复的假 Provider（T1.4 收敛对象）
- `FakeTtsProvider` + `UnavailableTtsProvider` / `FailTtsProvider`（test_tts.py）
- `FakeSpeechProvider` + Unavailable/Fail 变体（test_speech_eval.py）
- `FakeASR` + Fail 变体（test_routes_eval.py）
- 结构一致：`available` 标志 + 对应业务方法，可统一为带配置参数的工厂

### 4.3 外部调用 mock（触网风险点，T1.4 `no_external_calls` 覆盖对象）
- `services.evaluation_engine._call_judge`（LLM Judge）：3 处
- `services.evaluator._call_volcano`（大模型）：1 处
- `services.dialogue_task.match_dialogue`：1 处；`routes_dialogue.run_dialogue_task`：2 处
- `services.routes_conversation._generate_reply`：1 处
- TTS / Speech / ASR Provider：3 处

> 注意：`dialogue_task_routes` 中 `run_dialogue_task` 的测试是刻意验证真实执行器流程，只 mock 其内部 `match_dialogue`，因此 `no_external_calls` 应 mock 底层火山调用点而非 `run_dialogue_task` 本身。

## 5. 迁移工作量评估（T1.6 分批依据）

- **低**（无 seed、单模块、模板化）：tracking_endpoints、main_routes、calendar、session、state、content_v2_build
- **中**（有 seed 或需 provider 参数）：query_split、review_plan、weakness_plan、scholar_book、routes_conversation、routes_evaluation、routes_training、routes_eval、speech_eval、tts
- **高**（双 client / 类内测试 / 复杂 mock）：dialogue_task_routes、p2_features

## 6. 对后续小任务的启示

- **T1.2**：公共 fixture 需支持"路由模块列表 + 注入方式自动选择"；`client` 默认保持 tracking 单模块（不破坏 test_tracking_endpoints 模板文件）
- **T1.3**：统一注入需分两层——方式 A 模块走 autouse setattr 扫描；方式 B 模块在 app 构建时统一 dependency_overrides
- **T1.4**：`no_external_calls` autouse 默认 mock `_call_judge`/`_call_volcano`/火山底层，且允许测试显式覆盖
- **T1.5**：seed 工厂按集合命名，`_seed_content` 需参数化以兼容 4 处差异

## 7. T1.6 存量迁移完成记录（2026-08-18）

**结果：全量 `402 passed`（与基线一致，无丢失），lint 0，行为断言零改动。**

按工作量分批迁移，全部本地 `_client`/`_seed_*`/Fake Provider 收敛到公共设施：

- **低**：calendar、session、state → `make_client`（方式 A 模块，autouse 已兜底 get_db）
- **中**：
  - review_plan / weakness_plan → `make_client` + 公共 `seed_content`（weakness_plan 双教材用两次调用薄包装）
  - scholar_book → `make_client(tracking/state 双 router)` + `seed_content(1 课 2 句, include_text=False)`
  - routes_conversation → `make_client` + 保留 `_generate_reply` 确定性 mock
  - routes_training → `make_client`（`_call_judge` 由 no_external_calls 兜底）
  - tts → 本地三个 Fake Provider 类删，改用公共 `FakeTtsProvider`（DEFAULT_RAW 与本地 RAW_TTS_OK 值一致）
- **高**：
  - dialogue_task_routes → `make_client` + 公共 `seed_task`（字段含 TTL 完全等价）；真实使用 monkeypatch 的 6 个方法保留参数
  - p2_features → `make_client(tracking/dialogue 双 router)`，业务特有 seed（_seed_badges 等）保留

**收尾（T1.2/T1.5 遗留尾巴）**：query_split `_client` → `make_client`；routes_evaluation `_seed_attempt/_seed_speech` → 公共 `seed_attempt/seed_speech`（字段一致）；speech_eval 本地 Provider 类 → 公共 `FakeSpeechProvider`（DEFAULT_RAW 与本地 RAW_SOE_RESULT 值一致）。

**保留的业务差异（非重复代码）**：weakness_plan 双教材 `_seed_content`、training `_seed_state`（特定英文文本）、p2_features `_seed_badges`（`_id` 显式）、speech_eval/tts/conversation/routes_eval 的 `_client` 薄包装（注入 provider/模型输出）。

## 8. T1.7 约定固化（2026-08-18）

**T1.7 为系列收尾：将 T1.2~T1.6 落地的测试设施固化为一组强制约定，写入本文档作为长期规范。**

此后所有新增/修改测试（integration、e2e 为主；unit 若需造数同样走 `tests/fakes/`）一律遵循以下三大约定；测试数量、断言行为与基线（402 passed）保持一致是回归底线。

### 8.1 约定一：新测试造数统一走 `tests/fakes/`

| 场景 | 约定入口 | 说明 |
|---|---|---|
| 教材内容层级（章→课→句） | `seed_factory.seed_content` | 参数化：`textbook_id/chapter_id/lesson_ids/sentence_ids/include_text`；默认 tb_1 / 1 章 2 课 4 句 |
| dialogue_task 任务 | `seed_factory.seed_task` | `**overrides` 覆盖任意字段；默认 pending、TTL 24h |
| learning_attempt | `seed_factory.seed_attempt` | 返回 `_id` |
| speech_evaluation | `seed_factory.seed_speech` | 含 SOE-N parsed，返回 `_id` |
| skill_state 批量 | `seed_factory.seed_skill_states` | 传完整 dict 列表 |
| 语音评测 payload | `seed_factory.speech_payload` | `**overrides` 覆盖字段 |
| TTS 替身 | `fake_providers.FakeTtsProvider` | `DEFAULT_RAW`、`unavailable()`、`failing()`、`result=` |
| 语音评测替身 | `fake_providers.FakeSpeechProvider` | `DEFAULT_RAW`、`unavailable()`、`failing()` |
| ASR 替身 | `fake_providers.FakeAsrService` | `DEFAULT_RESULT`、`unavailable()`、`failing()` |

**强制规则**：
1. 不得在测试文件内再定义本地 `_seed_*` / `_payload` / Fake Provider 类；差异用工厂参数或 `overrides` 表达。
2. 仅当业务差异无法用参数表达（如 weakness_plan 双教材、training `_seed_state` 特定英文文本、p2_features `_seed_badges` 显式 `_id`）时，允许保留薄包装，但薄包装内部必须复用公共工厂。
3. 造数与断言解耦：seed 只负责预置数据，断言仍针对接口响应与落库文档，不做行为改动。

### 8.2 约定二：客户端统一 `make_client`

`tests/conftest.py` 的 `make_client(router..., patch_modules=(), overrides=None)` 是唯一客户端构建入口，自动完成 get_db 双通道注入：

- **方式 A 模块**（函数体内调用 `get_db()`：routes_tracking / routes_state / routes_dialogue / dialogue_task / routes_build）：直接 `make_client(router)`，get_db 由 `fake_db_auto_inject` autouse 兜底，无需手动 monkeypatch。
- **方式 B 模块**（`Depends(get_db)` 定义期捕获：routes_evaluation / routes_training / routes_conversation / routes_eval / routes_tts / routes_speech_eval）：`make_client(router, overrides={get_tts_provider: lambda: provider})` 注入 Provider；get_db 由 dependency_overrides 兜底。
- 多 router 组合：`make_client(state_router, tracking_router)`。

**强制规则**：
1. 不得再手写 `_client` / `_patch_db` / 内联 `monkeypatch.setattr("services.X.get_db", ...)`。
2. 需要注入 provider / 模拟模型输出时，用薄包装 `def _client(make_client, provider=None): return make_client(router, overrides=...)`，业务 mock 保留在薄包装内。
3. 真实使用 `monkeypatch` 的测试方法签名必须保留 `monkeypatch` 参数，不得因迁移误删。

### 8.3 约定三：外部调用默认屏蔽（`no_external_calls`）

`tests/integration/conftest.py` 与 `tests/e2e/conftest.py` 的 autouse fixture 默认屏蔽真实外部调用：

- `services.evaluation_engine._call_judge`（LLM Judge，评估回落 L1 规则）
- `services.evaluator._call_volcano`（火山方舟大模型）

**强制规则**：
1. 测试无需（也不应）再显式 mock 上述两个调用点；需要模拟模型返回时在测试体内 `monkeypatch.setattr` 覆盖（执行晚于 fixture，优先级更高）。
2. TTS / SOE-N / ASR 等真实 Provider 一律由测试显式注入 `tests/fakes/fake_providers.py` 替身，不依赖 `no_external_calls`。
3. 业务执行器本身的流程验证（如 `run_dialogue_task`）应 mock 其内部底层调用点而非执行器整体，保留真实流程覆盖。

### 8.4 新测试编写速查（Checklist）

```python
# 方式 A 模块（如 routes_tracking）
def test_x(self, make_client, fake_db):
    seed_content(fake_db)                    # 约定一：公共工厂造数
    client = make_client(tracking_router)    # 约定二：make_client
    resp = client.get("/tracking/...")
    assert resp.status_code == 200           # 外部调用已由 no_external_calls 屏蔽

# 方式 B 模块（如 routes_tts）
def test_x(self, make_client, fake_db):
    provider = FakeTtsProvider()             # 约定一：公共 Provider
    client = make_client(tts_router, overrides={get_tts_provider: lambda: provider})
    resp = client.post("/tts", json=...)
```

**完成标准**：新增测试后全量 `python -m pytest tests/ -q` 通过，且 lint（ruff/flake8 按项目配置）零错误；若发现新的重复 seed/provider 模式，先公共化到 `tests/fakes/` 再使用。
