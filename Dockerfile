# 1. 使用 .NET SDK 构建阶段（注意：BUILDPLATFORM 与 TARGETARCH 可根据需要传入）
FROM --platform=$BUILDPLATFORM mcr.microsoft.com/dotnet/sdk:9.0-bookworm-slim AS build-env
WORKDIR /root/build
ARG TARGETARCH

# 复制 .NET 项目相关文件（假设项目已放在目录 c 下，且项目文件位于 c/Lagrange.OneBot 中）
COPY c /root/build/c

# 执行 dotnet publish（注意调整项目路径，这里假定项目仍名为 Lagrange.OneBot）
RUN dotnet publish -p:DebugType="none" -a $TARGETARCH -f "net9.0" -o "/root/out" /root/build/c/Lagrange.OneBot

# 2. 最终阶段：基于 .NET Runtime 镜像，同时安装 Python 环境并复制 Python 代码
FROM mcr.microsoft.com/dotnet/runtime:9.0-bookworm-slim
WORKDIR /app

# 将 .NET 应用的发布文件复制到镜像中（此处放在 /app/dotnet/bin 目录）
COPY --from=build-env /root/out /app/dotnet/bin
# 同时复制 .NET 项目的 docker-entrypoint 脚本（注意：原来在 Lagrange.OneBot/Resources 下，现假定位于 c/Lagrange.OneBot/Resources）
COPY c/Lagrange.OneBot/Resources/docker-entrypoint.sh /app/dotnet/bin/docker-entrypoint.sh
RUN chmod +x /app/dotnet/bin/docker-entrypoint.sh

# 安装所需的系统软件：
# - gosu（用于权限切换，与原 .NET 脚本保持一致）
# - gcc、build-essential、python3、python3-pip、python3-dev、libffi-dev、libssl-dev（Python 编译和运行所需）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gosu \
        gcc \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        libffi-dev \
        libssl-dev && \
    # 为方便起见建立 python 的软链接，使“python”命令指向 python3
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 复制 Python 项目（位于 python 目录下）到镜像内 /app/python 目录
COPY python /app/python

# 安装 Python 项目所需依赖
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/python/requirements.txt --no-cache-dir && \
    python -m pip install socksio wechatpy cryptography --no-cache-dir

# 暴露 Python 服务可能用到的端口（原 dockerfile 中 EXPOSE 了 6185 和 6186）
EXPOSE 6185 6186

# 构造一个启动脚本，同时启动 .NET 和 Python 服务
# 此脚本将先后台启动 .NET 的 docker-entrypoint.sh 脚本，再进入 /app/python 目录启动 Python 主程序
RUN printf '#!/bin/sh\n\n'\
'echo "启动 .NET 服务..."\n'\
'/app/dotnet/bin/docker-entrypoint.sh &\n\n'\
'echo "启动 Python 服务..."\n'\
'cd /app/python\n'\
'exec python main.py\n' > /app/start.sh && chmod +x /app/start.sh

# 设置容器启动时执行该脚本
ENTRYPOINT ["/app/start.sh"]
