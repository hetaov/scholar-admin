"""外部依赖的内存替身(fakes)。

- fake_db: 模拟 services.database.CloudBaseNoSQLClient 接口的内存数据库
"""

# A02：干跑零写库断言引用的敏感集合清单
# 供 B10 引用，避免在各测试文件中硬编码漂移
SENSITIVE_COLLECTIONS = [
    "math_scan_upload",
    "error_record",
    "curriculum_node",
    "textbook",
    "audit_log",
]
