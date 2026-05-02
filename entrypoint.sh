#!/bin/sh
set -e

# 确保 chroma_db 目录存在且权限正确
mkdir -p /app/chroma_db
chown -R appuser:appuser /app/chroma_db

# 切换到 appuser 执行传入的命令
exec runuser -u appuser -- "$@"