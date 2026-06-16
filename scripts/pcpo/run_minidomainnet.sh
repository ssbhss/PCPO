#!/bin/bash

# PCPO miniDomainNet 完整实验脚本
# 运行所有miniDomainNet域适应任务 (源域到所有不同目标域)

# 设置参数
DATA_ROOT="data"
OUTPUT_ROOT="output/pcpo/miniDomainNet"
DATASET="miniDomainNet"
SEED=1

echo "PCPO miniDomainNet 完整实验"
echo "=========================="
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

# miniDomainNet 12个任务 (4个域，每个源域到其他三个目标域)
# Clipart -> others
run_experiment "clipart" "painting"     # Cl2Pa
run_experiment "clipart" "real"         # Cl2Re
run_experiment "clipart" "sketch"       # Cl2Sk

# Painting -> others
run_experiment "painting" "clipart"     # Pa2Cl
run_experiment "painting" "real"        # Pa2Re
run_experiment "painting" "sketch"      # Pa2Sk

# Real -> others
run_experiment "real" "clipart"         # Re2Cl
run_experiment "real" "painting"        # Re2Pa
run_experiment "real" "sketch"          # Re2Sk

# Sketch -> others
run_experiment "sketch" "clipart"       # Sk2Cl
run_experiment "sketch" "painting"      # Sk2Pa
run_experiment "sketch" "real"          # Sk2Re

echo "=== 所有miniDomainNet实验完成! ==="
echo ""
echo "结果汇总:"
for source in clipart painting real sketch; do
    for target in clipart painting real sketch; do
        if [ "$source" != "$target" ]; then
            exp="${source}2${target}"
            result_file="${OUTPUT_ROOT}/${exp}_seed${SEED}/final_results.txt"
            if [ -f "$result_file" ]; then
                final_acc=$(grep 'Final Accuracy' "$result_file" | cut -d':' -f2 | tr -d ' ')
                echo "  $exp: $final_acc"
            else
                echo "  $exp: 结果文件不存在"
            fi
        fi
    done
done

echo ""
echo "查看详细结果:"
echo "  ls ${OUTPUT_ROOT}/"
echo ""
echo "平均准确率计算:"
echo "  python scripts/pcpo/summarize_results.py --dataset minidomainnet"