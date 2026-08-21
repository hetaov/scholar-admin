FROM python:3.11-slim

WORKDIR /app

# ---------------------------------------------------------------------------
# 系统依赖（一次性 apt-get 安装 + 清理，保持 image 最小）
# - git:                 拉取 SOE-N 纯源码 SDK（GitHub release zip 为备用）
# - ca-certificates:     HTTPS 证书链（pip/requests/ark/volcano 必需）
# - curl:                health check / 调试（备用）
# - fonts-noto-cjk:      F3.2 A4 练习纸渲染：中文 + 数学符号字体
# - libpango/libcairo/libgdkpixbuf: WeasyPrint 系统依赖（HTML→PDF 渲染）
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing \
        git \
        ca-certificates \
        curl \
        fonts-noto-cjk \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        shared-mime-info \
    ; \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple/

# 复制项目文件
COPY . .

# 官方 SOE-N SDK（tencentcloud-speech-sdk-python）纯源码分发（PyPI 无包，见 scripts/soe_n_verify.py）。
# 若 COPY 后 vendor/ 已随上下文带入（本地含 vendor 的场景）→ 跳过；否则 GitHub 拉取。
# 注意：git 已在第 1 层 apt-get 安装，此处不再触网 apt，避免 sources.list 过期。
RUN set -eux; \
    if [ ! -d vendor/tencentcloud-speech-sdk-python/common ]; then \
        ( \
            # 优先：GitHub release tar.gz（带宽更稳、不依赖 git 网络层握手细节）
            curl -fsSL -o /tmp/sdk.tgz \
                https://codeload.github.com/TencentCloud/tencentcloud-speech-sdk-python/tar.gz/refs/heads/master \
            && mkdir -p vendor/tencentcloud-speech-sdk-python \
            && tar -xzf /tmp/sdk.tgz -C vendor/tencentcloud-speech-sdk-python --strip-components=1 \
            && rm -f /tmp/sdk.tgz \
        ) || ( \
            # 备用：git 浅克隆（HTTP/1.1 规避 HTTP/2 framing layer 偶发断连）
            git -c http.version=HTTP/1.1 clone --depth 1 \
                https://github.com/TencentCloud/tencentcloud-speech-sdk-python.git \
                vendor/tencentcloud-speech-sdk-python \
        ); \
    fi; \
    # 最后清理：无论用哪种方式拉到 SDK，purge git 都不影响（文件已落盘）
    if command -v git >/dev/null 2>&1; then \
        apt-get purge -y git >/dev/null 2>&1 || true; \
        apt-get autoremove -y >/dev/null 2>&1 || true; \
    fi

# 暴露端口
EXPOSE 8080

# 启动服务
CMD ["python", "main.py"]

