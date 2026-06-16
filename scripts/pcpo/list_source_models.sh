#!/bin/bash

# 列出所有已训练的源域模型

MODEL_DIR="pretrained_models"

echo "已训练的源域模型列表"
echo "===================="

if [ ! -d "$MODEL_DIR" ]; then
    echo "模型目录不存在: $MODEL_DIR"
    exit 1
fi

# 统计模型数量
total_models=0
office31_models=0
office_home_models=0
visda17_models=0
minidomainnet_models=0

echo ""
echo "Office31 模型:"
for model in "$MODEL_DIR"/best_model_office31_*.pth; do
    if [ -f "$model" ]; then
        basename_model=$(basename "$model")
        domain=$(echo "$basename_model" | sed 's/best_model_office31_\(.*\)\.pth/\1/')
        size=$(du -h "$model" | cut -f1)
        echo "  ✓ $domain ($size)"
        ((office31_models++))
        ((total_models++))
    fi
done
if [ $office31_models -eq 0 ]; then
    echo "  (无模型)"
fi

echo ""
echo "OfficeHome 模型:"
for model in "$MODEL_DIR"/best_model_office_home_*.pth; do
    if [ -f "$model" ]; then
        basename_model=$(basename "$model")
        domain=$(echo "$basename_model" | sed 's/best_model_office_home_\(.*\)\.pth/\1/')
        size=$(du -h "$model" | cut -f1)
        echo "  ✓ $domain ($size)"
        ((office_home_models++))
        ((total_models++))
    fi
done
if [ $office_home_models -eq 0 ]; then
    echo "  (无模型)"
fi

echo ""
echo "VisDA17 模型:"
for model in "$MODEL_DIR"/best_model_visda17_*.pth; do
    if [ -f "$model" ]; then
        basename_model=$(basename "$model")
        domain=$(echo "$basename_model" | sed 's/best_model_visda17_\(.*\)\.pth/\1/')
        size=$(du -h "$model" | cut -f1)
        echo "  ✓ $domain ($size)"
        ((visda17_models++))
        ((total_models++))
    fi
done
if [ $visda17_models -eq 0 ]; then
    echo "  (无模型)"
fi

echo ""
echo "miniDomainNet 模型:"
for model in "$MODEL_DIR"/best_model_minidomainnet_*.pth; do
    if [ -f "$model" ]; then
        basename_model=$(basename "$model")
        domain=$(echo "$basename_model" | sed 's/best_model_minidomainnet_\(.*\)\.pth/\1/')
        size=$(du -h "$model" | cut -f1)
        echo "  ✓ $domain ($size)"
        ((minidomainnet_models++))
        ((total_models++))
    fi
done
if [ $minidomainnet_models -eq 0 ]; then
    echo "  (无模型)"
fi

echo ""
echo "统计信息:"
echo "=========="
echo "Office31:      $office31_models/3 个模型"
echo "OfficeHome:    $office_home_models/4 个模型"
echo "VisDA17:       $visda17_models/1 个模型"
echo "miniDomainNet: $minidomainnet_models/4 个模型"
echo "总计:          $total_models 个模型"

# 计算总大小
if [ $total_models -gt 0 ]; then
    total_size=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
    echo "总大小:        $total_size"
fi

echo ""
echo "模型目录: $MODEL_DIR"