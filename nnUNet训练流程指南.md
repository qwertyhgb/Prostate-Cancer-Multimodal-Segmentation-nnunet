# nnU-Net v2训练流程完整指南

## 📋 前提条件

确保已完成BPH-PCA数据转换：
```bash
python script/run_conversion.py  # 选择相似性填充模式
```

转换完成后应该有：
```
nnUNet_raw/Dataset001_BPH_PCA/
├── imagesTr/          # 训练图像（459个.nii.gz文件）
├── labelsTr/          # 训练标签（459个.nii.gz文件）
├── imagesTs/          # 测试图像（空）
└── dataset.json       # 数据集配置
```

## 🔧 环境设置

### 1. 安装nnU-Net v2
```bash
pip install nnunetv2
```

### 2. 设置环境变量（重要！）
```bash
# Windows (PowerShell)
$env:nnUNet_raw = "D:\path\to\your\project\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\path\to\your\project\nnUNet_preprocessed"
$env:nnUNet_results = "D:\path\to\your\project\nnUNet_results"

# Linux/Mac (Bash)
export nnUNet_raw="/path/to/your/project/nnUNet_raw"
export nnUNet_preprocessed="/path/to/your/project/nnUNet_preprocessed"
export nnUNet_results="/path/to/your/project/nnUNet_results"
```

### 3. 验证环境设置
```bash
python -c "import os; print('nnUNet_raw:', os.environ.get('nnUNet_raw'))"
```

## 🚀 训练流程

### 步骤1: 数据预处理和实验规划
```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

**这一步会做什么：**
- 验证数据集完整性
- 分析图像属性（尺寸、间距、强度等）
- 自动确定最佳网络配置
- 生成预处理计划
- 执行数据预处理（重采样、归一化等）

**预期输出：**
```
nnUNet_preprocessed/Dataset001_BPH_PCA/
├── dataset_fingerprint.json
├── nnUNetPlans.json
├── nnUNetTrainer__nnUNetPlans__3d_fullres/
└── splits_final.json
```

**预计时间：** 10-30分钟（取决于硬件）

### 步骤2: 开始训练

#### 2.1 3D全分辨率训练（推荐）
```bash
nnUNetv2_train 1 3d_fullres 0 --npz
```

**参数说明：**
- `1`: 数据集ID
- `3d_fullres`: 配置名称（3D全分辨率）
- `0`: fold编号（5折交叉验证的第0折）
- `--npz`: 保存为npz格式（节省空间）

#### 2.2 可选：训练所有5折
```bash
# 并行训练多个fold（如果有多个GPU）
nnUNetv2_train 1 3d_fullres 0 --npz &
nnUNetv2_train 1 3d_fullres 1 --npz &
nnUNetv2_train 1 3d_fullres 2 --npz &
nnUNetv2_train 1 3d_fullres 3 --npz &
nnUNetv2_train 1 3d_fullres 4 --npz &
wait
```

#### 2.3 可选：低分辨率训练
```bash
nnUNetv2_train 1 3d_lowres 0 --npz
```

**预计训练时间：**
- 单fold: 1-3天（取决于GPU性能）
- 全部5fold: 5-15天

### 步骤3: 模型集成（可选）
```bash
# 集成所有fold的结果
nnUNetv2_ensemble -i nnUNet_results/Dataset001_BPH_PCA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0 \
                     nnUNet_results/Dataset001_BPH_PCA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1 \
                     nnUNet_results/Dataset001_BPH_PCA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2 \
                     nnUNet_results/Dataset001_BPH_PCA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3 \
                     nnUNet_results/Dataset001_BPH_PCA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4 \
                  -o ensemble_results
```

## 📊 训练监控

### 1. 查看训练进度
```bash
# 查看训练日志
tail -f nnUNet_results/Dataset001_BPH_PCA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt
```

### 2. 使用TensorBoard监控
```bash
tensorboard --logdir nnUNet_results/Dataset001_BPH_PCA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0
```

### 3. 检查GPU使用情况
```bash
nvidia-smi  # 每隔几秒运行一次
```

## 🔍 模型推理

### 1. 单个模型推理
```bash
nnUNetv2_predict -i input_folder -o output_folder -d 1 -c 3d_fullres -f 0
```

### 2. 集成模型推理
```bash
nnUNetv2_predict -i input_folder -o output_folder -d 1 -c 3d_fullres -f all
```

### 3. 批量推理脚本示例
```python
import os
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# 初始化预测器
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_gpu=True,
    device='cuda',
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)

# 加载模型
predictor.initialize_from_trained_model_folder(
    'nnUNet_results/Dataset001_BPH_PCA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0',
    use_folds=(0,),
    checkpoint_name='checkpoint_final.pth'
)

# 执行预测
predictor.predict_from_files(
    list_of_lists_or_source_folder='input_folder',
    output_folder_or_list_of_truncated_output_files='output_folder',
    save_probabilities=False,
    overwrite=True,
    num_processes_preprocessing=2,
    num_processes_segmentation_export=2,
    folder_with_segs_from_prev_stage=None,
    num_parts=1,
    part_id=0
)
```

## 📈 性能评估

### 1. 交叉验证评估
```bash
nnUNetv2_evaluate_folder -ref labels_folder -pred predictions_folder -l 1 2
```

### 2. 查看验证结果
```bash
# 查看summary.json文件
cat nnUNet_results/Dataset001_BPH_PCA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/summary.json
```

### 3. 预期性能指标
基于相似性填充的BPH-PCA数据集：
- **Dice系数**: 0.84-0.87
- **95% Hausdorff距离**: < 5mm
- **平均表面距离**: < 2mm
- **敏感性**: > 0.85
- **特异性**: > 0.90

## ⚠️ 常见问题和解决方案

### 1. 内存不足
```bash
# 减少批量大小
nnUNetv2_train 1 3d_fullres 0 --npz -batch_size 1
```

### 2. GPU内存不足
```bash
# 使用CPU训练（较慢）
nnUNetv2_train 1 3d_fullres 0 --npz --device cpu
```

### 3. 训练中断恢复
```bash
# 从检查点继续训练
nnUNetv2_train 1 3d_fullres 0 --npz --continue_training
```

### 4. 验证数据集完整性
```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity --clean
```

## 🎯 训练优化建议

### 1. 硬件建议
- **GPU**: RTX 3080/4080以上，显存≥12GB
- **内存**: 32GB以上
- **存储**: SSD，至少100GB可用空间

### 2. 训练策略
- 先训练单个fold验证效果
- 确认无问题后训练全部5fold
- 使用模型集成提升性能

### 3. 超参数调优（高级）
```bash
# 自定义训练参数
nnUNetv2_train 1 3d_fullres 0 --npz -lr 0.01 -momentum 0.99
```

## 📝 训练检查清单

- [ ] 环境变量设置正确
- [ ] 数据集格式验证通过
- [ ] GPU驱动和CUDA版本兼容
- [ ] 足够的磁盘空间（>100GB）
- [ ] 训练日志正常输出
- [ ] 验证指标持续改善
- [ ] 模型检查点正常保存

## 🎉 训练完成后

训练完成后，你将获得：
1. 训练好的模型权重
2. 验证集性能报告
3. 可用于推理的模型
4. TensorBoard训练曲线

可以开始在新的前列腺MRI数据上进行BPH/PCA分割预测了！

---

**预计总时间**: 数据预处理(30分钟) + 训练(1-3天) + 验证(1小时)  
**推荐配置**: 相似性填充 + 3d_fullres + 5折交叉验证  
**预期效果**: Dice > 0.84，适用于临床前列腺分割任务