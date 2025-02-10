# -------------------------
# 阶段1：构建 .NET 应用
# -------------------------
FROM --platform=$BUILDPLATFORM mcr.microsoft.com/dotnet/sdk:9.0-bookworm-slim AS build-env
WORKDIR /root/build
ARG TARGETARCH

# 将 .NET 项目（现位于 c 目录）复制进来
COPY c /root/build/c

# 执行发布，注意调整项目路径（这里假定项目仍名为 Lagrange.OneBot）
RUN dotnet publish -p:DebugType="none" -a $TARGETARCH -f "net9.0" -o "/root/out" /root/build/c/Lagrange.OneBot

# -------------------------
# 阶段2：构建最终镜像（包含 .NET runtime 和 Python 环境）
# -------------------------
FROM mcr.microsoft.com/dotnet/runtime:9.0-bookworm-slim
WORKDIR /app

# 复制 .NET 应用的发布输出和 docker-entrypoint 脚本
COPY --from=build-env /root/out /app/dotnet/bin
COPY c/Lagrange.OneBot/Resources/docker-entrypoint.sh /app/dotnet/bin/docker-entrypoint.sh
RUN chmod +x /app/dotnet/bin/docker-entrypoint.sh

# 安装系统依赖：gosu、gcc、build-essential 以及 Python 环境和相关依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    gcc \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    libffi-dev \
    libssl-dev && \
    # 建立 python 的软链接，方便使用 python 命令
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 将 Python 项目复制到镜像中（假定目录为 python）
COPY python /app/python

# 安装 Python 包，同时添加 --break-system-packages 以避免 “externally-managed-environment” 错误
RUN python -m pip install --upgrade pip --break-system-packages && \
    python -m pip install -r /app/python/requirements.txt --no-cache-dir --break-system-packages && \
    python -m pip install socksio wechatpy cryptography --no-cache-dir --break-system-packages

# 暴露 Python 服务使用的端口（根据原 Dockerfile 设置）
EXPOSE 6185 6186

# 构造启动脚本，同时启动 .NET 服务和 Python 服务
RUN printf '#!/bin/sh\n\n'\
'echo "Starting .NET service..."\n'\
'/app/dotnet/bin/docker-entrypoint.sh &\n\n'\
'echo "Starting Python service..."\n'\
'cd /app/python\n'\
'exec python main.py\n' > /app/start.sh && chmod +x /app/start.sh

# 设置容器启动时执行该启动脚本
ENTRYPOINT ["/app/start.sh"]
