#!/bin/bash

# PCPO OfficeHome 完整实验脚本
# 运行所有OfficeHome域适应任务 (源域到所有不同目标域)

# 设置参数
DATA_ROOT="data"
OUTPUT_ROOT="output/pcpo/office_home"
DATASET="office_home"
SEED=1

echo "PCPO OfficeHome 完整实验"
echo "========================"
echo "数据集: ${DATASET}"
echo "输出目录: ${OUTPUT_ROOT}"
echo "种子: ${SEED}"
echo ""

# 创建输出目录
mkdir -p ${OUTPUT_ROOT}

# 实验计数器
current=1
total=12

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

# OfficeHome 12个任务 (每个源域到其他三个目标域)
# Art -> others
run_experiment "art" "clipart"      # Ar2Cl
run_experiment "art" "product"      # Ar2Pr
run_experiment "art" "real_world"   # Ar2Rw

# Clipart -> others
run_experiment "clipart" "art"      # Cl2Ar
run_experiment "clipart" "product"  # Cl2Pr
run_experiment "clipart" "real_world" # Cl2Rw

# Product -> others
run_experiment "product" "art"      # Pr2Ar
run_experiment "product" "clipart"  # Pr2Cl
run_experiment "product" "real_world" # Pr2Rw

# Real World -> others
run_experiment "real_world" "art"   # Rw2Ar
run_experiment "real_world" "clipart" # Rw2Cl
run_experiment "real_world" "product" # Rw2Pr

echo "=== 所有OfficeHome实验完成! ==="
echo ""
echo "结果汇总:"
for exp in art2clipart art2product art2real_world clipart2art clipart2product clipart2real_world product2art product2clipart product2real_world real_world2art real_world2clipart real_world2product; do
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
echo "  python scripts/pcpo/summarize_results.py --dataset office_home"