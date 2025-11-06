# BPH-PCA多模态前列腺MRI数据转换工具

将BPH-PCA数据集转换为nnU-Net v2格式的完整解决方案。

## ⚡ 快速开始

```bash
# 1. 检查数据
python script/check_data_structure.py

# 2. 执行转换（推荐）
python script/run_conversion.py
# 选择选项3：相似性填充模式

# 3. 开始训练
nnUNetv2_plan_and_preprocess -d 1
nnUNetv2_train 1 3d_fullres 0
```

## 🎯 核心特性

- ✅ **智能相似性填充**: 基于T2模态估算缺失的gaoqing-T2
- ✅ **多模态融合**: 支持ADC、DWI、T2 fs、T2 not fs、gaoqing-T2
- ✅ **自动重采样**: 处理图像形状不一致问题
- ✅ **完整数据利用**: 处理全部459例数据
- ✅ **标准格式输出**: 生成nnU-Net v2兼容格式

## 📊 数据概览

- **总病例数**: 459例（240例BPH + 219例PCA）
- **模态支持**: 5种MRI序列
- **输出格式**: 5通道NIfTI文件
- **预期效果**: Dice > 0.84

## 🔧 处理模式

| 模式 | 通道数 | 数据量 | 推荐度 | 说明 |
|------|--------|--------|--------|------|
| **similarity_fill** | 5 | 459例 | ⭐⭐⭐⭐⭐ | 相似性填充（默认推荐） |
| zero_fill | 5 | 459例 | ⭐⭐⭐⭐ | 0填充 |
| core_4 | 4 | 459例 | ⭐⭐⭐⭐ | 核心4模态 |
| strict_5 | 5 | 258例 | ⭐⭐⭐ | 严格5模态 |

## 📁 项目结构

```
├── script/                        # 核心脚本
│   ├── convert_bph_pca_to_nnunet.py  # 主转换脚本
│   ├── run_conversion.py             # 快速运行
│   └── check_data_structure.py       # 数据检查
├── data/BPH-PCA/                  # 原始数据
└── nnUNet_raw/                    # 输出目录
```

## 💡 使用建议

**推荐配置**: 使用`similarity_fill`模式，获得最佳的数据利用率和分割效果。

详细说明请查看 `script/使用说明.md` 或 `项目结构说明.md`。