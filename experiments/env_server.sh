#!/bin/bash
# 服务器环境配置（使用 HuggingFace 镜像）
# 用法: source env_server.sh

# 使用 HuggingFace 镜像（国内可访问）
#export HF_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT='https://mirrors.tuna.tsinghua.edu.cn/'

# 可选：设置缓存目录（如果需要指定位置，取消下面注释并修改路径）
# export HF_HOME="$HOME/.cache/huggingface"

echo "✓ 已配置为服务器模式（使用镜像: $HF_ENDPOINT）"

