#!/bin/bash

# PCPO VisDA17 实验脚本
# 运行VisDA17域适应任务 (Synthetic -> Real)

# 设置参数
DATA_ROOT="data"
OUTPUT_ROOT="output/pcpo/visda17"
DATASET="visda17"
SOURCE="synthetic"
TARGET="real"
SEED=1

echo "PCPO VisDA17 实验"
echo "================="
echo "数据集: ${DATASET}"
echo "源域: ${SOURCE}"
echo "目标域: ${TARGET}"
echo "输出目录: ${OUTPUT_ROOT}"
echo "种子: ${SEED}"
echo ""

# 创建输出目录
mkdir -p ${OUTPUT_ROOT}

# 实验计数器
current=1
total=1

# 函数：运行单个实验
run_experiment() {
    local source=$1
    local target=$2
    local exp_name="${source}2${target}_seed${SEED}"
    
    echo "=== 实验 $current/$total: $exp_name ==="
    
    bash scripts/pcpo/run_single.sh ${DATASET} ${source} ${target}
    
    echo "完成实验 $current/$total: $exp_name"
    echo "----------------------------------------"
    ((current++))
}

# VisDA17 1个任务 (Synthetic -> Real)
run_experiment "${SOURCE}" "${TARGET}"  # Sy2Re

echo "=== VisDA17实验完成! ==="
echo ""
echo "结果汇总:"
exp="${SOURCE}2${TARGET}"
result_file="${OUTPUT_ROOT}/${exp}_seed${SEED}/final_results.txt"
if [ -f "$result_file" ]; then
    final_acc=$(grep 'Final Accuracy' "$result_file" | cut -d':' -f2 | tr -d ' ')
    echo "  $exp: $final_acc"
else
    echo "  $exp: 结果文件不存在"
fi

echo ""
echo "查看详细结果:"
echo "  ls ${OUTPUT_ROOT}/"
echo ""
echo "结果分析:"
echo "  python scripts/pcpo/summarize_results.py --dataset visda17"