#!/bin/bash

# PCPO 主运行脚本
# 提供简单的命令行接口来运行PCPO实验

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

# 显示帮助信息
show_help() {
    echo "PCPO (Prototype-based Cross-domain Prompt Optimization) 运行脚本"
    echo ""
    echo "用法:"
    echo "  $0 [选项] <命令> [参数...]"
    echo ""
    echo "命令:"
    echo "  train-source <dataset> <domain>     训练源域模型"
    echo "  train-all-source                    训练所有源域模型"
    echo "  list-models                         列出所有已训练的源域模型"
    echo "  single <dataset> <source> <target>  运行单个域适应实验"
    echo "  dataset <dataset>                   运行完整数据集实验"
    echo "  all                                 运行所有数据集实验"
    echo "  summarize [dataset]                 汇总结果"
    echo ""
    echo "支持的数据集:"
    echo "  office31, office_home, visda17, minidomainnet"
    echo ""
    echo "选项:"
    echo "  -h, --help                          显示此帮助信息"
    echo "  --data-root <path>                  数据根目录 (默认: data)"
    echo "  --output-dir <path>                 输出目录 (默认: output/pcpo)"
    echo "  --seed <num>                        随机种子 (默认: 1)"
    echo "  --gpu <ids>                         GPU设备ID (默认: 0)"
    echo ""
    echo "示例:"
    echo "  # 训练Office31的Amazon源域模型"
    echo "  $0 train-source office31 amazon"
    echo ""
    echo "  # 训练所有源域模型"
    echo "  $0 train-all-source"
    echo ""
    echo "  # 运行Office31的Amazon到DSLR适应"
    echo "  $0 single office31 amazon dslr"
    echo ""
    echo "  # 运行Office31所有实验"
    echo "  $0 dataset office31"
    echo ""
    echo "  # 运行所有数据集实验"
    echo "  $0 all"
    echo ""
    echo "  # 汇总Office31结果"
    echo "  $0 summarize office31"
    echo ""
    echo "  # 汇总所有结果"
    echo "  $0 summarize"
}

# 检查数据集是否支持
check_dataset() {
    local dataset=$1
    case $dataset in
        office31|office_home|visda17|minidomainnet)
            return 0
            ;;
        *)
            print_error "不支持的数据集: $dataset"
            print_info "支持的数据集: office31, office_home, visda17, minidomainnet"
            exit 1
            ;;
    esac
}

# 检查数据目录是否存在
check_data_dir() {
    local dataset=$1
    local domain=$2
    
    # 对于minidomainnet，数据实际存储在domainnet目录下
    if [ "$dataset" = "minidomainnet" ]; then
        local data_path="$DATA_ROOT/domainnet/$domain"
    else
        local data_path="$DATA_ROOT/$dataset/$domain"
    fi
    
    if [ ! -d "$data_path" ]; then
        print_error "数据目录不存在: $data_path"
        print_info "请确保数据集已下载并放置在正确位置"
        exit 1
    fi
}

# 获取数据集信息
get_dataset_info() {
    local dataset=$1
    case $dataset in
        office31)
            echo "31 resnet50"
            ;;
        office_home)
            echo "65 resnet50"
            ;;
        visda17)
            echo "12 resnet101"
            ;;
        minidomainnet)
            echo "126 resnet50"
            ;;
    esac
}

# 训练源域模型
train_source() {
    local dataset=$1
    local domain=$2
    
    check_dataset $dataset
    check_data_dir $dataset $domain
    
    local info=$(get_dataset_info $dataset)
    local num_classes=$(echo $info | cut -d' ' -f1)
    
    print_info "训练源域模型: $dataset - $domain"
    print_info "类别数: $num_classes"
    
    # 对于minidomainnet，数据实际存储在domainnet目录下
    if [ "$dataset" = "minidomainnet" ]; then
        local source_data_dir="$DATA_ROOT/domainnet/$domain"
    else
        local source_data_dir="$DATA_ROOT/$dataset/$domain"
    fi
    
    python tools/pcpo/train_source_simple.py \
        --dataset $dataset \
        --source-domain $domain \
        --source-data-dir "$source_data_dir" \
        --num-classes $num_classes \
        --output-dir "pretrained_models" \
        --use-auto-config \
        --seed $SEED
    
    print_success "源域模型训练完成"
}

# 运行单个实验
run_single() {
    local dataset=$1
    local source=$2
    local target=$3
    
    check_dataset $dataset
    check_data_dir $dataset $source
    check_data_dir $dataset $target
    
    print_info "运行单个实验: $dataset $source -> $target"
    
    bash scripts/pcpo/run_single.sh $dataset $source $target
    
    print_success "实验完成: $dataset $source -> $target"
}

# 运行数据集实验
run_dataset() {
    local dataset=$1
    
    check_dataset $dataset
    
    print_info "运行数据集实验: $dataset"
    
    case $dataset in
        office31)
            bash scripts/pcpo/run_office31.sh
            ;;
        office_home)
            bash scripts/pcpo/run_officehome.sh
            ;;
        visda17)
            bash scripts/pcpo/run_visda17.sh
            ;;
        minidomainnet)
            bash scripts/pcpo/run_minidomainnet.sh
            ;;
    esac
    
    print_success "数据集实验完成: $dataset"
}

# 运行所有实验
run_all() {
    print_info "运行所有数据集实验"
    
    bash scripts/pcpo/run_all.sh
    
    print_success "所有实验完成"
}

# 汇总结果
summarize_results() {
    local dataset=$1
    
    print_info "汇总结果"
    
    if [ -n "$dataset" ]; then
        check_dataset $dataset
        python scripts/pcpo/summarize_results.py --dataset $dataset
    else
        python scripts/pcpo/summarize_results.py --all-datasets
    fi
    
    print_success "结果汇总完成"
}

# 默认参数
DATA_ROOT="data"
OUTPUT_DIR="output/pcpo"
SEED=1
GPU_IDS="0"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --gpu)
            GPU_IDS="$2"
            shift 2
            ;;
        train-source)
            if [ $# -lt 3 ]; then
                print_error "train-source 需要2个参数: <dataset> <domain>"
                exit 1
            fi
            export CUDA_VISIBLE_DEVICES=$GPU_IDS
            train_source "$2" "$3"
            exit 0
            ;;
        train-all-source)
            export CUDA_VISIBLE_DEVICES=$GPU_IDS
            bash scripts/pcpo/run_train_all_source.sh --data-root "$DATA_ROOT" --output-dir "pretrained_models" --seed "$SEED"
            exit 0
            ;;
        list-models)
            bash scripts/pcpo/list_source_models.sh
            exit 0
            ;;
        single)
            if [ $# -lt 4 ]; then
                print_error "single 需要3个参数: <dataset> <source> <target>"
                exit 1
            fi
            export CUDA_VISIBLE_DEVICES=$GPU_IDS
            run_single "$2" "$3" "$4"
            exit 0
            ;;
        dataset)
            if [ $# -lt 2 ]; then
                print_error "dataset 需要1个参数: <dataset>"
                exit 1
            fi
            export CUDA_VISIBLE_DEVICES=$GPU_IDS
            run_dataset "$2"
            exit 0
            ;;
        all)
            export CUDA_VISIBLE_DEVICES=$GPU_IDS
            run_all
            exit 0
            ;;
        summarize)
            summarize_results "$2"
            exit 0
            ;;
        *)
            print_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
done

# 如果没有提供命令，显示帮助
show_help