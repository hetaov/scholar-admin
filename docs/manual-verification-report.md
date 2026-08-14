# Scholar Admin API 人工验证报告

- 验证日期：2026-08-13
- 验证方式：Postman Collection 已创建于 `docs/scholar-admin.postman_collection.json`（8 分组 / 29 请求），本次通过 curl 逐接口人工验证
- 验证环境：本地 FakeDB 内存库（`python local_dev.py`，端口 8081）

## 验证结论

**核心业务接口全部通过**（系统 / CRUD / 学习追踪 / 学者×教材 / 教材管理 / 管理页），发现 **1 处真实 bug（vision 三接口调用不存在的方法）已修复**，另有若干"需外部密钥"类接口属预期失败。

## 通过接口（23 项）

| # | 接口 | 验证结果 |
|---|------|---------|
| 1 | `GET /` 服务信息 | ✅ 正常返回服务名/版本 |
| 2 | `GET /health` 健康检查 | ✅ 正常 |
| 3 | `GET /collections` 集合清单 | ✅ 15 集合，含 `scholar_book` |
| 4 | `GET /collections/{name}` 结构 | ✅ 返回集合文档结构 |
| 5-9 | `POST /collections/{name}/count/insert/query/update/delete` | ✅ 全流程通过（含多文档数组插入、`$set` 更新） |
| 10 | `POST /tracking/state` 上报学习状态 | ✅ 写 `skill_state` + `study_attempt`（mastery 0.9 落库） |
| 11 | `POST /tracking/session/start` 创建会话 | ✅ 返回 `session_id` |
| 12 | `POST /tracking/session/end` 结算 | ✅ 回写 `scholar_book`（last_studied_at 更新） |
| 13 | `GET /tracking/{id}` 学习追踪 | ✅ 分页结构正常 |
| 14 | `POST /tracking/stats` 聚合统计 | ✅ progress 0.45、1/2 句已学、时长 2分0秒 |
| 15 | `GET /scholar/{id}/books` 断点取回 | ✅ 含断点 + 进度 summary，按最近学习倒序 |
| 16 | `PUT /scholar/{id}/books/{tb}/position` 记录断点 | ✅ 复合键 `scholar_demo_001_tb_01`，幂等 upsert |
| 17 | `GET /textbook` 教材列表 | ✅ |
| 18 | `POST /textbook` 添加教材 | ✅ |
| 19 | `GET /admin/textbook` 管理页 | ✅ 返回 HTML 页面 |
| 20 | `POST /admin/textbook/merge` | ✅ 参数校验正常（`text_book_ids`） |
| 21 | `POST /match/dialogue` | ✅ 参数校验正常（`scholarId` + `sentence`） |
| 22 | `GET /docs` Swagger | ✅ |
| 23 | `GET /admin/textbook/cleanup` | ✅ 参数校验正常 |

## 发现并修复的 Bug

**`services/routes_vision.py` 三个识别接口调用不存在的方法**（火山服务重构后遗留）：

- `POST /vision/recognize` → 原调用 `service.recognize_upload(...)` ❌
- `POST /vision/recognize-url` → 原调用 `service.recognize_url(...)` ❌
- `POST /vision/recognize-base64` → 原调用 `service.recognize_upload(...)` ❌

`VolcanoVisionService` 实际只有统一入口 `recognize(image_bytes=..., image_url=...)`（同时支持 bytes 与 URL），且路由层还对已解析结果二次调用 `_parse_response`（会把 dict 当原始文本再次解析而报错）。

**修复**（`routes_vision.py`）：

```python
# 文件上传
result = service.recognize(image_bytes=contents)
# URL
result = service.recognize(image_url=body.url)
# Base64
result = service.recognize(image_bytes=buf)
# 三处返回均直接使用 result（recognize 已返回解析后的 dict）
```

**回归验证**：全量 `python -m pytest tests/ -q` → **203 passed**，lint 干净。

## 未完整验证项（环境限制）

| 接口 | 原因 |
|------|------|
| 真实腾讯云数据库（8080 的 `main.py`） | `.env` 中密钥签名失败 `AuthFailure.SignatureFailure`（密钥疑似过期）。更新 `.env` 中 `TENCENTCLOUD_SECRETID/SECRETKEY` 后可用 Postman 直接验证真实环境 |
| `POST /vision/*` 实际识别 | 需要 `VOLCANO_API_KEY` / `VOLCANO_VISION_MODEL` 环境变量（接口链路已修复，缺密钥报错属预期） |
| `POST /build/sentence` / `build/nce` 等构建接口 | 大批量写入，FakeDB 下执行超时，属预期（真实环境建议后台执行） |
| `POST /admin/textbook/cleanup` | 破坏性操作，仅验证参数校验，未实际删除数据 |

## 使用 Postman 人工验证

1. **导入**：Postman → Import → 选择 `docs/scholar-admin.postman_collection.json`
2. **联调环境（无密钥）**：`python local_dev.py`（8081），collection 中 `baseUrl` 改为 `http://localhost:8081`
3. **真实环境**：修复 `.env` 密钥后 `python main.py`（8080），`baseUrl` 保持 `http://localhost:8080`
4. **推荐顺序**：01 系统 → 02 CRUD → 03 学习追踪（state → session/start → session/end）→ 04 学者×教材（position → books）→ 05 教材管理 → 06/07/08（需密钥，可选）
5. 「创建学习会话」请求已内置脚本，会自动把 `session_id` 写入 collection 变量供「结算」使用
