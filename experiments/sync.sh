#!/bin/bash
# 同步 generated_kernels 到服务器
# 用法: ./sync.sh

cd "$(dirname "$0")/.."

echo "=== 同步 experiments/generated_kernels 到服务器 ==="
tar czf - experiments/generated_kernels 2>/dev/null | ssh A800-didi "cd ~/projects/kernelbench-openevolve && tar xzf -"

echo "=== 同步完成 ==="

