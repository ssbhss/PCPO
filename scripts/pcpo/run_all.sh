#!/bin/bash

# PCPO 所有数据集完整实验脚本
# 运行所有支持的数据集的域适应任务

echo "PCPO 所有数据集完整实验"
echo "======================"
echo "开始时间: $(date)"
echo ""

# 创建主输出目录
mkdir -p output/pcpo

# 实验计数器
current_dataset=1
total_datasets=4

# 函数：运行数据集实验
run_dataset() {
    local dataset=$1
    local script=$2
    
    echo "=== 数据集 $current_dataset/$total_datasets: $dataset ==="
    echo "开始时间: $(date)"
    
    if [ -f "$script" ]; then
        bash "$script"
        echo "完成数据集: $dataset"
    else
        echo "错误: 脚本不存在: $script"
    fi
    
    echo "结束时间: $(date)"
    echo "================================================"
    ((current_dataset++))
}

# 运行所有数据集实验
run_dataset "Office31" "scripts/pcpo/run_office31.sh"
run_dataset "OfficeHome" "scripts/pcpo/run_officehome.sh"
run_dataset "VisDA17" "scripts/pcpo/run_visda17.sh"
run_dataset "miniDomainNet" "scripts/pcpo/run_minidomainnet.sh"

echo "=== 所有PCPO实验完成! ==="
echo "结束时间: $(date)"
echo ""

# 生成总体结果汇总
echo "生成总体结果汇总..."
python scripts/pcpo/summarize_results.py --all-datasets

echo ""
echo "查看所有结果:"
echo "  ls output/pcpo/"
echo ""
echo "详细结果文件:"
echo "  output/pcpo/all_results_summary.txt"