#!/bin/bash

# PCPO 一键训练所有源域模型脚本
# 为所有支持的数据集训练源域模型

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 默认参数
DATA_ROOT="data"
OUTPUT_DIR="pretrained_models"
MAX_ITERATIONS=20000
BATCH_SIZE=64
LR=0.01
SEED=1

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "PCPO 一键训练所有源域模型脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --data-root <path>     数据根目录 (默认: data)"
            echo "  --output-dir <path>    输出目录 (默认: pretrained_models)"
            echo "  --max-iterations <num> 最大迭代次数 (默认: 20000)"
            echo "  --batch-size <num>     批大小 (默认: 64)"
            echo "  --lr <float>           学习率 (默认: 0.001)"
            echo "  --seed <num>           随机种子 (默认: 1)"
            echo "  -h, --help             显示此帮助信息"
            echo ""
            echo "支持的数据集和源域:"
            echo "  Office31: amazon, dslr, webcam"
            echo "  OfficeHome: art, clipart, product, real_world"
            echo "  VisDA17: train"
            echo "  miniDomainNet: clipart, painting, real, sketch"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            exit 1
            ;;
    esac
done

print_info "PCPO 一键训练所有源域模型"
print_info "=========================="
print_info "数据根目录: $DATA_ROOT"
print_info "输出目录: $OUTPUT_DIR"
print_info "最大迭代次数: $MAX_ITERATIONS"
print_info "批大小: $BATCH_SIZE"
print_info "学习率: $LR"
print_info "随机种子: $SEED"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 训练计数器
current=1
total=0

# 计算总任务数
datasets=("office31" "office_home" "visda17" "minidomainnet")
declare -A domain_counts
domain_counts["office31"]=3
domain_counts["office_home"]=4
domain_counts["visda17"]=1
domain_counts["minidomainnet"]=4

for dataset in "${datasets[@]}"; do
    # 对于minidomainnet，检查domainnet目录
    if [ "$dataset" = "minidomainnet" ]; then
        if [ -d "$DATA_ROOT/domainnet" ]; then
            total=$((total + domain_counts[$dataset]))
        fi
    else
        if [ -d "$DATA_ROOT/$dataset" ]; then
            total=$((total + domain_counts[$dataset]))
        fi
    fi
done

print_info "预计训练 $total 个源域模型"
echo ""

# 训练函数
train_source_model() {
    local dataset=$1
    local domain=$2
    local num_classes=$3
    local data_dir="$DATA_ROOT/$dataset/$domain"
    
    print_info "[$current/$total] 训练源域模型: $dataset - $domain"
    
    # 检查数据目录是否存在
    # 对于minidomainnet，数据实际存储在domainnet目录下
    if [ "$dataset" = "minidomainnet" ]; then
        actual_data_dir="$DATA_ROOT/domainnet/$domain"
    else
        actual_data_dir="$data_dir"
    fi
    
#    if [ ! -d "$actual_data_dir" ]; then
#        print_warning "数据目录不存在，跳过: $actual_data_dir"
#        ((current++))
#        return
#    fi
    
    # 检查是否已存在训练好的模型
    model_path="$OUTPUT_DIR/best_model_${dataset}_${domain}.pth"
    if [ -f "$model_path" ]; then
        print_warning "模型已存在，跳过: $model_path"
        ((current++))
        return
    fi
    
    # 训练模型
    python tools/pcpo/train_source_simple.py \
        --dataset "$dataset" \
        --source-domain "$domain" \
        --source-data-dir "$actual_data_dir" \
        --num-classes "$num_classes" \
        --output-dir "$OUTPUT_DIR" \
        --use-auto-config \
        --seed "$SEED"
    
    if [ $? -eq 0 ]; then
        print_success "完成: $dataset - $domain"
    else
        print_error "失败: $dataset - $domain"
    fi
    
    ((current++))
    echo ""
}

# Office31 源域模型训练
if [ -d "$DATA_ROOT/office31" ]; then
    print_info "=== Office31 源域模型训练 ==="
    train_source_model "office31" "amazon" 31
    train_source_model "office31" "dslr" 31
    train_source_model "office31" "webcam" 31
fi

# OfficeHome 源域模型训练
if [ -d "$DATA_ROOT/office_home" ]; then
    print_info "=== OfficeHome 源域模型训练 ==="
    train_source_model "office_home" "art" 65
    train_source_model "office_home" "clipart" 65
    train_source_model "office_home" "product" 65
    train_source_model "office_home" "real_world" 65
fi

# VisDA17 源域模型训练
if [ -d "$DATA_ROOT/visda17" ]; then
    print_info "=== VisDA17 源域模型训练 ==="
    train_source_model "visda17" "synthetic" 12  # train为源域
fi

# miniDomainNet 源域模型训练 (4个主要域)
if [ -d "$DATA_ROOT/domainnet" ]; then
    print_info "=== miniDomainNet 源域模型训练 ==="
    train_source_model "minidomainnet" "clipart" 126
    train_source_model "minidomainnet" "painting" 126
    train_source_model "minidomainnet" "real" 126
    train_source_model "minidomainnet" "sketch" 126
fi

print_success "=== 所有源域模型训练完成! ==="
echo ""

# 汇总训练结果
print_info "训练结果汇总:"
trained_models=0
total_models=0

for dataset in "${datasets[@]}"; do
    if [ ! -d "$DATA_ROOT/$dataset" ]; then
        continue
    fi
    
    case $dataset in
        "office31")
            domains=("amazon" "dslr" "webcam")
            ;;
        "office_home")
            domains=("art" "clipart" "product" "real_world")
            ;;
        "visda17")
            domains=("synthetic")
            ;;
        "minidomainnet")
            domains=("clipart" "painting" "real" "sketch")
            ;;
    esac
    
    echo "  $dataset:"
    for domain in "${domains[@]}"; do
        if [ "$dataset" = "minidomainnet" ]; then
            if [ -d "$DATA_ROOT/domainnet" ]; then
                total=$((total + domain_counts[$dataset]))
            fi
        fi
        model_path="$OUTPUT_DIR/best_model_${dataset}_${domain}.pth"
        total_models=$((total_models + 1))
        if [ -f "$model_path" ]; then
            trained_models=$((trained_models + 1))
            print_success "    ✓ $domain"
        else
            print_error "    ✗ $domain"
        fi
    done
done

echo ""
print_info "训练完成统计: $trained_models/$total_models 个模型"

if [ $trained_models -eq $total_models ]; then
    print_success "所有源域模型训练成功!"
    print_info "现在可以运行PCPO实验:"
    echo "  bash scripts/pcpo/run_single.sh <dataset> <source> <target>"
    echo "  bash scripts/pcpo/run_all.sh"
else
    print_warning "部分源域模型训练失败，请检查错误信息"
fi

echo ""
print_info "模型保存位置: $OUTPUT_DIR"
print_info "查看训练日志: ls $OUTPUT_DIR/*.txt"