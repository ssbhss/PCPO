#!/bin/bash

# PCPO 单个实验脚本
# 用法: bash scripts/pcpo/run_single.sh <dataset> <source> <target> [output_suffix]

# 检查参数
if [ $# -lt 3 ]; then
    echo "用法: $0 <dataset> <source> <target> [output_suffix]"
    echo "示例: $0 Office31 amazon dslr"
    echo "示例: $0 OfficeHome art clipart custom_exp"
    echo "示例: $0 VisDA17 synthetic real"
    echo "示例: $0 miniDomainNet real sketch"
    exit 1
fi

DATASET=$1
SOURCE=$2
TARGET=$3
OUTPUT_SUFFIX=${4:-""}

# 设置基本参数
DATA_ROOT="data"
CONFIG_FILE="configs/trainers/PCPO/pcpo.yaml"
SEED=1

# 数据集名称映射 (脚本使用小写，DASSL使用首字母大写)
case $DATASET in
    "office31" | "Office31")
        DATASET_LOWER="office31"
        DATASET_DASSL="Office31"
        DATASET_CONFIG="configs/datasets/office31.yaml"
        NUM_CLASSES=31
        SOURCE_ARCH="resnet50"
        ;;
    "office_home" | "OfficeHome")
        DATASET_LOWER="office_home"
        DATASET_DASSL="OfficeHome"
        DATASET_CONFIG="configs/datasets/office_home.yaml"
        NUM_CLASSES=65
        SOURCE_ARCH="resnet50"
        ;;
    "visda17" | "VisDA17")
        DATASET_LOWER="visda17"
        DATASET_DASSL="VisDA17"
        DATASET_CONFIG="configs/datasets/visda17.yaml"
        NUM_CLASSES=12
        SOURCE_ARCH="resnet101"
        ;;
    "minidomainnet" | "miniDomainNet")
        DATASET_LOWER="minidomainnet"
        DATASET_DASSL="miniDomainNet"
        DATASET_CONFIG="configs/datasets/mini_domainnet.yaml"
        NUM_CLASSES=126
        SOURCE_ARCH="resnet50"
        ;;
    *)
        echo "不支持的数据集: $DATASET"
        echo "支持的数据集: office31/Office31, office_home/OfficeHome, visda17/VisDA17, minidomainnet/miniDomainNet"
        exit 1
        ;;
esac

# 生成输出目录名 (使用小写)
if [ -n "$OUTPUT_SUFFIX" ]; then
    OUTPUT_DIR="output/pcpo/${DATASET_LOWER}/${SOURCE}2${TARGET}_${OUTPUT_SUFFIX}"
else
    OUTPUT_DIR="output/pcpo/${DATASET_LOWER}/${SOURCE}2${TARGET}_seed${SEED}"
fi

# 设置源模型路径 (使用小写)
SOURCE_MODEL_PATH="pretrained_models/best_model_${DATASET_LOWER}_${SOURCE}.pth"

# 设置目标域数据路径 (使用小写)
if [ "$DATASET_LOWER" = "minidomainnet" ]; then
    TARGET_DATA_DIR="${DATA_ROOT}/domainnet/${TARGET}"
else
    TARGET_DATA_DIR="${DATA_ROOT}/${DATASET_LOWER}/${TARGET}"
fi

echo "PCPO 单个实验"
echo "=============="
echo "数据集 (输入): ${DATASET}"
echo "数据集 (小写): ${DATASET_LOWER}"
echo "数据集 (DASSL): ${DATASET_DASSL}"
echo "源域: ${SOURCE}"
echo "目标域: ${TARGET}"
echo "类别数: ${NUM_CLASSES}"
echo "源模型架构: ${SOURCE_ARCH}"
echo "源模型路径: ${SOURCE_MODEL_PATH}"
echo "目标数据路径: ${TARGET_DATA_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "种子: ${SEED}"
echo ""

# 检查源模型是否存在
if [ ! -f "$SOURCE_MODEL_PATH" ]; then
    echo "错误: 源模型不存在: $SOURCE_MODEL_PATH"
    echo "请先训练源模型:"
    echo "  ./run_pcpo.sh train-source $DATASET $SOURCE"
    echo "  或者运行: ./run_pcpo.sh train-all-source"
    if [ "$DATASET" = "minidomainnet" ]; then
        echo "  或者直接运行: python tools/pcpo/train_source_simple.py --dataset $DATASET --source-domain $SOURCE --source-data-dir ${DATA_ROOT}/domainnet/${SOURCE} --num-classes $NUM_CLASSES"
    else
        echo "  或者直接运行: python tools/pcpo/train_source_simple.py --dataset $DATASET --source-domain $SOURCE --source-data-dir ${DATA_ROOT}/${DATASET}/${SOURCE} --num-classes $NUM_CLASSES"
    fi
    echo "  模型将保存为: pretrained_models/best_model_${DATASET}_${SOURCE}.pth"
    exit 1
fi

# 检查目标数据是否存在
#if [ ! -d "$TARGET_DATA_DIR" ]; then
#    echo "错误: 目标数据目录不存在: $TARGET_DATA_DIR"
#    exit 1
#fi

# 调试信息
echo "调试信息:"
echo "  DATASET_CONFIG: ${DATASET_CONFIG}"
echo "  CONFIG_FILE: ${CONFIG_FILE}"
echo "  DATASET_DASSL: ${DATASET_DASSL}"
echo "  NUM_CLASSES: ${NUM_CLASSES}"
echo "  SOURCE_ARCH: ${SOURCE_ARCH}"
echo ""

# 运行PCPO实验
python tools/pcpo/train.py \
    --root ${DATA_ROOT} \
    --trainer PCPO \
    --dataset-config-file ${DATASET_CONFIG} \
    --config-file ${CONFIG_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --source-domains ${SOURCE} \
    --target-domains ${TARGET} \
    --source-model-path ${SOURCE_MODEL_PATH} \
    --target-data-dir ${TARGET_DATA_DIR} \
    --seed ${SEED} \
    --opts \
    TRAINER.PCPO.NUM_CLASSES ${NUM_CLASSES} \
    TRAINER.PCPO.SOURCE_ARCH ${SOURCE_ARCH} \
    DATASET.NAME ${DATASET_DASSL}

echo ""
echo "实验完成!"
echo "结果保存在: ${OUTPUT_DIR}"
echo ""
echo "查看结果:"
echo "  cat ${OUTPUT_DIR}/final_results.txt"