# 第一阶段：构建 .NET 项目
FROM --platform=$BUILDPLATFORM mcr.microsoft.com/dotnet/sdk:9.0-bookworm-slim AS dotnet-build
ARG TARGETARCH
WORKDIR /build

# 只复制 C# 项目目录
COPY ./c ./c
RUN dotnet publish -p:DebugType="none" -a $TARGETARCH -f "net9.0" -o "/out" "./c/Lagrange.OneBot"

# 第二阶段：构建 Python 基础镜像
FROM python:3.10-slim AS python-base
WORKDIR /AstrBot

# 安装 Python 项目依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制 Python 项目文件
COPY ./python/requirements.txt .
COPY ./python/main.py .
RUN pip install --no-cache-dir -r requirements.txt socksio wechatpy cryptography

# 第三阶段：最终运行时镜像
FROM mcr.microsoft.com/dotnet/runtime:9.0-bookworm-slim

# 安装 Python 运行时和依赖
RUN apt-get update && \
    apt-get install -y \
    python3.10 \
    python3-pip \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录结构
WORKDIR /app
RUN mkdir -p /app/c/bin /app/python

# 从构建阶段复制文件
COPY --from=dotnet-build /out /app/c/bin
COPY --from=python-base /AstrBot /app/python
COPY ./c/Lagrange.OneBot/Resources/docker-entrypoint.sh /app/c/bin/

# 设置入口点脚本和权限
RUN chmod +x /app/c/bin/docker-entrypoint.sh

# 安装进程管理工具
RUN apt-get update && \
    apt-get install -y supervisor && \
    rm -rf /var/lib/apt/lists/*

# 配置 Supervisor
RUN mkdir -p /var/log/supervisor
COPY <<EOF /etc/supervisor/conf.d/supervisord.conf
[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log

[program:dotnet-app]
command=/app/c/bin/docker-entrypoint.sh
directory=/app/c/bin
autostart=true
autorestart=true

[program:python-app]
command=python3 /app/python/main.py
directory=/app/python
autostart=true
autorestart=true
EOF

# 设置非特权用户
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app /var/log/supervisor

# 暴露端口
EXPOSE 6185 6186

# 启动服务
USER appuser
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
