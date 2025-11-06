# BPH-PCA数据转换为nnU-Net v2格式

这个工具包用于将BPH-PCA多模态前列腺MRI数据转换为nnU-Net v2所需的格式。

## 功能特点

- ✅ 支持多模态MRI数据（ADC, DWI, T2等）
- ✅ 自动处理BPH和PCA两类数据
- ✅ 生成符合nnU-Net v2命名规范的文件
- ✅ 自动创建dataset.json配置文件
- ✅ 完整的数据验证和错误处理

## 数据结构要求

确保你的BPH-PCA数据按以下结构组织：

```
data/BPH-PCA/
├── BPH/                    # 良性前列腺增生数据
│   ├── ADC/               # ADC模态
│   ├── DWI/               # DWI模态  
│   ├── gaoqing-T2/        # 高清T2模态
│   ├── T2 fs/             # T2脂肪抑制
│   └── T2 not fs/         # T2非脂肪抑制
├── PCA/                    # 前列腺癌数据
│   ├── ADC/
│   ├── DWI/
│   ├── gaoqing-T2/
│   ├── T2 fs/
│   └── T2 not fs/
└── ROI(BPH+PCA)/          # 标注文件
    ├── BPH/               # BPH标注
    └── PCA/               # PCA标注
```

## 使用方法

### 方法1: 快速运行（推荐）

```bash
python script/run_conversion.py
```

这会使用默认设置进行转换。

### 方法2: 自定义参数

```bash
python script/convert_bph_pca_to_nnunet.py \
    --source_dir data/BPH-PCA \
    --output_dir nnUNet_raw \
    --dataset_id 1 \
    --train_ratio 0.8
```

### 参数说明

- `--source_dir`: BPH-PCA数据源目录（默认: data/BPH-PCA）
- `--output_dir`: nnU-Net输出目录（默认: nnUNet_raw）
- `--dataset_id`: 数据集ID（默认: 1）
- `--train_ratio`: 训练集比例（默认: 0.8）
- `--require_all_modalities`: 要求所有5个模态都存在（默认: False）
- `--min_modalities`: 最少需要的模态数量（默认: 2）
- `--use_core_modalities_only`: 只使用4个核心模态，排除gaoqing-T2（推荐）

## 输出格式

转换后会生成以下结构：

```
nnUNet_raw/Dataset001_BPH_PCA/
├── imagesTr/              # 训练图像
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
├── labelsTr/              # 训练标签
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
├── imagesTs/              # 测试图像（可选）
└── dataset.json           # 数据集配置文件
```

## 标签说明

- **0**: 背景
- **1**: BPH（良性前列腺增生）
- **2**: PCA（前列腺癌）

## 后续步骤

转换完成后，按以下步骤使用nnU-Net：

### 1. 设置环境变量

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"  
export nnUNet_results="/path/to/nnUNet_results"
```

### 2. 数据预处理

```bash
nnUNetv2_plan_and_preprocess -d 1
```

### 3. 开始训练

```bash
# 3D全分辨率训练
nnUNetv2_train 1 3d_fullres 0

# 如果需要，也可以训练其他配置
nnUNetv2_train 1 3d_lowres 0
nnUNetv2_train 1 3d_cascade_fullres 0
```

### 4. 模型推理

```bash
nnUNetv2_predict -i input_folder -o output_folder -d 1 -c 3d_fullres -f 0
```

## 依赖包

确保安装以下Python包：

```bash
pip install nibabel numpy tqdm
pip install nnunetv2  # 用于后续训练
```

## 数据完整性处理

### 🔧 三种处理模式

#### 1. 核心模态模式（强烈推荐）
- 只使用4个核心模态（ADC, DWI, T2 fs, T2 not fs）
- 排除gaoqing-T2，解决数据不一致问题
- 确保所有病例具有相同通道数
- 最适合nnU-Net训练

```bash
python script/convert_bph_pca_to_nnunet.py --use_core_modalities_only
```

#### 2. 严格模式
- 要求所有5个模态都存在
- 确保数据完整性
- 可能减少可用病例数（258例）

```bash
python script/convert_bph_pca_to_nnunet.py --require_all_modalities
```

#### 3. 宽松模式（不推荐用于训练）
- 至少需要2个模态
- 允许部分模态缺失
- ⚠️ 会导致通道数不一致，影响nnU-Net训练

```bash
python script/convert_bph_pca_to_nnunet.py --min_modalities 2
```

### ⚠️ 数据筛选规则

**会被跳过的病例**：
1. 缺少标签文件（ROI）
2. 模态数量少于最小要求
3. 严格模式下缺少任何模态
4. 图像形状不一致且无法修复

**会被处理的病例**：
1. 有标签文件
2. 满足最小模态数量要求
3. 至少有一个有效的模态数据

## 注意事项

1. **数据一致性**: 确保所有NII文件的空间分辨率和方向一致
2. **标签格式**: 标签文件应该是二值图像（0为背景，>0为前景）
3. **内存要求**: 处理大型数据集时需要足够的内存
4. **文件命名**: 确保图像和标签文件使用相同的病例ID命名
5. **通道一致性**: 所有病例将具有相同的通道数，确保nnU-Net训练兼容性

## 故障排除

### 常见问题

1. **找不到文件**: 检查数据目录结构是否正确
2. **形状不匹配**: 确保同一病例的所有模态具有相同的空间维度
3. **内存不足**: 考虑分批处理或使用更大内存的机器

### 获取帮助

如果遇到问题，请检查：
1. 数据目录结构是否符合要求
2. 文件权限是否正确
3. Python环境是否安装了所需依赖

## 许可证

本工具仅供研究使用。