# PCPO (Prototype-based Cross-domain Prompt Optimization) 实验脚本

本目录包含运行PCPO方法的所有实验脚本和工具。

## 文件结构

```
scripts/pcpo/
├── README.md                    # 本文件
├── run_single.sh               # 单个实验脚本
├── run_office31.sh             # Office31完整实验
├── run_officehome.sh           # OfficeHome完整实验  
├── run_visda17.sh              # VisDA17实验
├── run_minidomainnet.sh        # miniDomainNet完整实验
├── run_all.sh                  # 所有数据集实验
└── summarize_results.py        # 结果汇总工具
```

## 快速开始

### 1. 准备数据

确保数据集已下载并放置在正确位置：

```bash
data/
├── office31/
│   ├── amazon/
│   ├── dslr/
│   └── webcam/
├── office_home/
│   ├── art/
│   ├── clipart/
│   ├── product/
│   └── real_world/
├── visda17/
│   ├── train/      # 合成数据 (源域)
│   └── validation/ # 真实数据 (目标域)
└── domainnet/          # miniDomainNet数据存储在这里
    ├── clipart/
    ├── painting/
    ├── real/
    └── sketch/
```

### 2. 训练源域模型

在运行PCPO之前，需要先训练源域模型：

```bash
# Office31 - Amazon源域
python tools/pcpo/train_source.py \
    --dataset office31 \
    --source-domain amazon \
    --source-data-dir data/office31/amazon \
    --num-classes 31

# OfficeHome - Art源域  
python tools/pcpo/train_source.py \
    --dataset office_home \
    --source-domain art \
    --source-data-dir data/office_home/art \
    --num-classes 65

# VisDA17 - Synthetic源域
python tools/pcpo/train_source.py \
    --dataset visda17 \
    --source-domain synthetic \
    --source-data-dir data/visda17/train \
    --num-classes 12

# miniDomainNet - Real源域
python tools/pcpo/train_source.py \
    --dataset minidomainnet \
    --source-domain real \
    --source-data-dir data/minidomainnet/real \
    --num-classes 126
```

### 3. 运行PCPO实验

#### 单个实验

```bash
# 运行单个域适应任务
bash scripts/pcpo/run_single.sh <dataset> <source> <target> [output_suffix]

# 示例
bash scripts/pcpo/run_single.sh office31 amazon dslr
bash scripts/pcpo/run_single.sh office_home art clipart  
bash scripts/pcpo/run_single.sh visda17 synthetic real
bash scripts/pcpo/run_single.sh minidomainnet real sketch
```

#### 完整数据集实验

```bash
# Office31 (6个任务)
bash scripts/pcpo/run_office31.sh

# OfficeHome (12个任务)  
bash scripts/pcpo/run_officehome.sh

# VisDA17 (1个任务)
bash scripts/pcpo/run_visda17.sh

# miniDomainNet (30个任务)
bash scripts/pcpo/run_minidomainnet.sh

# 所有数据集 (49个任务)
bash scripts/pcpo/run_all.sh
```

## 实验配置

### 支持的数据集

| 数据集 | 类别数 | 域数 | 任务数 | 源模型架构 |
|--------|--------|------|--------|------------|
| Office31 | 31 | 3 | 6 | ResNet50 |
| OfficeHome | 65 | 4 | 12 | ResNet50 |
| VisDA17 | 12 | 2 | 1 | ResNet101 |
| miniDomainNet | 126 | 6 | 30 | ResNet50 |

### 默认参数

- 学习率: 0.01
- 批大小: 64  
- 优化器: SGD
- 动量: 0.9
- 权重衰减: 0.0005
- 原型训练轮数: 10
- CLIP架构: ViT-B/16

## 结果分析

### 查看单个实验结果

```bash
# 查看特定实验结果
cat output/pcpo/office31/amazon2dslr_seed1/final_results.txt
```

### 汇总结果

```bash
# 汇总单个数据集结果
python scripts/pcpo/summarize_results.py --dataset office31

# 汇总所有数据集结果  
python scripts/pcpo/summarize_results.py --all-datasets

# 查看汇总文件
cat output/pcpo/all_results_summary.txt
```

### 结果格式

结果文件包含以下信息：
- 每个任务的准确率
- 平均准确率和标准差
- 最小/最大准确率
- 任务数量统计

## 自定义实验

### 修改配置

编辑配置文件来调整实验参数：

```bash
# PCPO主配置 (包含数据集特定配置)
configs/trainers/PCPO/pcpo.yaml

# 标准数据集配置
configs/datasets/office31.yaml
configs/datasets/office_home.yaml  
configs/datasets/visda17.yaml
configs/datasets/mini_domainnet.yaml
```

数据集特定的配置（如类别数、源模型架构等）现在直接集成在PCPO的trainer配置中，无需单独的数据集配置文件。

### 添加新数据集

1. 创建数据集配置文件
2. 在`run_single.sh`中添加数据集支持
3. 创建专用的实验脚本

## 故障排除

### 常见问题

1. **源模型不存在**
   ```bash
   错误: 源模型不存在: output/pcpo/source_models/best_model_amazon.pth
   ```
   解决：先运行源域模型训练

2. **目标数据目录不存在**
   ```bash
   错误: 目标数据目录不存在: data/office31/dslr
   ```
   解决：检查数据集路径和目录结构

3. **CUDA内存不足**
   - 减小批大小
   - 使用更小的模型架构

4. **导入错误**
   - 检查Python路径设置
   - 确保所有依赖已安装

### 调试模式

添加调试选项：

```bash
# 详细输出
export VERBOSE=1

# 单GPU模式
export CUDA_VISIBLE_DEVICES=0
```

## 性能优化

### 多GPU训练

```bash
# 使用多个GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 加速训练

- 使用更大的批大小
- 启用混合精度训练
- 使用更快的数据加载器

## 引用

如果使用此代码，请引用相关论文：

```bibtex
@article{pcpo2024,
  title={Prototype-based Cross-domain Prompt Optimization for Few-shot Domain Adaptation},
  author={Author Name},
  journal={Conference/Journal Name},
  year={2024}
}
```