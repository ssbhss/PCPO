#!/bin/bash

# PCPO Office31 完整实验脚本
# 运行所有Office31域适应任务 (源域到所有不同目标域)

# 设置参数
DATA_ROOT="data"
OUTPUT_ROOT="output/pcpo/office31"
DATASET="office31"
SEED=1

echo "PCPO Office31 完整实验"
echo "======================"
echo "数据集: ${DATASET}"
echo "输出目录: ${OUTPUT_ROOT}"
echo "种子: ${SEED}"
echo ""

# 创建输出目录
mkdir -p ${OUTPUT_ROOT}

# 实验计数器
current=1
total=6

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

# Office31 6个任务 (每个源域到其他两个目标域)
run_experiment "amazon" "dslr"      # A2D
run_experiment "amazon" "webcam"    # A2W
run_experiment "dslr" "amazon"      # D2A
run_experiment "dslr" "webcam"      # D2W
run_experiment "webcam" "amazon"    # W2A
run_experiment "webcam" "dslr"      # W2D

echo "=== 所有Office31实验完成! ==="
echo ""
echo "结果汇总:"
for exp in amazon2dslr amazon2webcam dslr2amazon dslr2webcam webcam2amazon webcam2dslr; do
    result_file="${OUTPUT_ROOT}/${exp}_seed${SEED}/final_results.txt"
    if [ -f "$result_file" ]; then
        final_acc=$(grep 'Final Accuracy' "$result_file" | cut -d':' -f2 | tr -d ' ')
        echo "  $exp: $final_acc"
    else
        echo "  $exp: 结果文件不存在"
    fi
done

echo ""
echo "查看详细结果:"
echo "  ls ${OUTPUT_ROOT}/"
echo ""
echo "平均准确率计算:"
echo "  python scripts/pcpo/summarize_results.py --dataset office31"