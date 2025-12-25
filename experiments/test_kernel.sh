#!/bin/bash
# 通过 Git 同步并测试 kernel
# 用法: ./test_kernel.sh <kernel_file> <level> <problem_id>
# 示例: ./test_kernel.sh level3_problem43_v1.py 3 43

KERNEL_FILE=$1
LEVEL=$2
PROBLEM_ID=$3

if [ -z "$KERNEL_FILE" ] || [ -z "$LEVEL" ] || [ -z "$PROBLEM_ID" ]; then
    echo "用法: ./test_kernel.sh <kernel_file> <level> <problem_id>"
    echo "示例: ./test_kernel.sh level3_problem43_v1.py 3 43"
    exit 1
fi

cd "$(dirname "$0")/.."

echo "=== 步骤 1: Git commit & push ==="
git add experiments/generated_kernels/
git commit -m "Test kernel: ${KERNEL_FILE}" 2>/dev/null || echo "(无新更改需要提交)"
git push

echo ""
echo "=== 步骤 2: 服务器 git pull + 运行测试 ==="
ssh A800-didi "cd ~/projects/kernelbench-openevolve && \
    git pull && \
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate kernelbench-evolve && \
    export HF_ENDPOINT='https://hf-mirror.com' && \
    cd KernelBench && \
    PYTHONPATH=. python3 scripts/run_and_check.py \
    ref_origin=kernelbench \
    level=${LEVEL} \
    problem_id=${PROBLEM_ID} \
    kernel_src_path=../experiments/generated_kernels/${KERNEL_FILE} \
    eval_mode=local \
    gpu_arch='[\"Ampere\"]'"
