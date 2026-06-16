# PCPO 数据集结构说明

## 支持的数据集

PCPO支持以下4个数据集，每个数据集有不同的域和数据结构：

### 1. Office31
- **路径**: `data/office31/`
- **域**: amazon, dslr, webcam
- **类别数**: 31
- **源模型**: ResNet50

```
data/office31/
├── amazon/
│   ├── back_pack/
│   ├── bike/
│   └── ... (31个类别)
├── dslr/
└── webcam/
```

### 2. OfficeHome
- **路径**: `data/office_home/`
- **域**: art, clipart, product, real_world
- **类别数**: 65
- **源模型**: ResNet50

```
data/office_home/
├── art/
│   ├── Alarm_Clock/
│   ├── Backpack/
│   └── ... (65个类别)
├── clipart/
├── product/
└── real_world/
```

### 3. VisDA17
- **路径**: `data/visda17/`
- **域**: train (源域), test (目标域)
- **类别数**: 12
- **源模型**: ResNet101

```
data/visda17/
├── train/          # 源域 (合成数据)
│   ├── aeroplane/
│   ├── bicycle/
│   └── ... (12个类别)
└── test/           # 目标域 (真实数据)
    ├── aeroplane/
    ├── bicycle/
    └── ... (12个类别)
```

### 4. miniDomainNet
- **路径**: `data/domainnet/` ⚠️ **注意：存储在domainnet目录下**
- **域**: clipart, painting, real, sketch (4个域)
- **类别数**: 126
- **源模型**: ResNet50

```
data/domainnet/     # 注意：不是minidomainnet
├── clipart/
│   ├── aircraft_carrier/
│   ├── airplane/
│   └── ... (126个类别)
├── painting/
├── real/
└── sketch/
```

## 重要说明

### miniDomainNet 特殊路径
miniDomainNet数据集虽然在代码中称为`minidomainnet`，但实际数据存储在`data/domainnet/`目录下。这是因为：

1. DASSL中的数据集定义使用`domainnet`作为目录名
2. miniDomainNet是DomainNet的一个子集
3. 数据集类`miniDomainNet`会自动处理路径映射

### 数据格式
所有数据集都使用标准的ImageFolder格式：
- 每个类别一个文件夹
- 文件夹名即为类别名
- 支持常见图片格式：jpg, jpeg, png, bmp等

## 检查数据结构

使用以下命令检查数据集结构是否正确：

```bash
# 检查所有数据集
python check_data_structure.py data

# 检查特定数据集
python check_data_structure.py data office31 amazon
python check_data_structure.py data minidomainnet clipart  # 会自动检查domainnet/clipart
```

## 训练源域模型

```bash
# Office31
./run_pcpo.sh train-source office31 amazon

# OfficeHome  
./run_pcpo.sh train-source office_home art

# VisDA17
./run_pcpo.sh train-source visda17 train

# miniDomainNet (会自动使用domainnet路径)
./run_pcpo.sh train-source minidomainnet clipart
```

## 运行PCPO实验

```bash
# Office31: Amazon -> DSLR
./run_pcpo.sh single office31 amazon dslr

# OfficeHome: Art -> Clipart
./run_pcpo.sh single office_home art clipart

# VisDA17: Train -> Test
./run_pcpo.sh single visda17 train test

# miniDomainNet: Clipart -> Painting
./run_pcpo.sh single minidomainnet clipart painting
```

## 故障排除

### 常见错误

1. **数据目录不存在**
   ```
   错误: 数据目录不存在: data/minidomainnet/clipart
   ```
   **解决**: miniDomainNet数据应放在`data/domainnet/clipart`

2. **类别文件夹缺失**
   ```
   Warning: Expected 31 classes, found 30
   ```
   **解决**: 检查是否有类别文件夹缺失或命名错误

3. **图片文件格式**
   ```
   Warning: Skipping invalid image xxx.txt
   ```
   **解决**: 确保目录下只有图片文件，移除其他文件

### 数据集下载

- **Office31**: [官方链接](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
- **OfficeHome**: [官方链接](https://www.hemanthdv.org/officeHomeDataset.html)
- **VisDA17**: [官方链接](http://ai.bu.edu/visda-2017/)
- **miniDomainNet**: [官方链接](http://ai.bu.edu/M3SDA/)