import os
import json
import shutil
import nibabel as nib
import numpy as np
from pathlib import Path

def convert_bph_pca_to_nnunet_v2_flexible(data_root_dir, output_dir, task_name="Task999_ProstateMultiModal"):
    """
    将BPH-PCA多模态前列腺MRI数据转换为nnU-Net v2格式（灵活模式）
    即使某些模态缺失，也会转换所有可用的病例
    
    参数:
        data_root_dir: 原始数据根目录 (包含BPH-PCA文件夹的路径)
        output_dir: 输出目录
        task_name: nnU-Net任务名称
    """
    
    # 定义模态映射 (符合项目要求的模态顺序)
    modality_mapping = {
        'ADC': '0000',
        'DWI': '0001', 
        'T2 fs': '0002',
        'T2 not fs': '0003',
        'gaoqing-T2': '0004'
    }
    
    # 创建输出目录结构
    task_dir = os.path.join(output_dir, task_name)
    images_tr_dir = os.path.join(task_dir, 'imagesTr')
    labels_tr_dir = os.path.join(task_dir, 'labelsTr')
    
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)
    
    # 病例列表
    training_cases = []
    case_count = 0
    
    # 处理BPH和PCA病例
    for case_type in ['BPH', 'PCA']:
        case_type_dir = os.path.join(data_root_dir, 'BPH-PCA', case_type)
        roi_dir = os.path.join(data_root_dir, 'BPH-PCA', 'ROI(BPH+PCA)', case_type)
        
        if not os.path.exists(case_type_dir):
            print(f"警告: {case_type_dir} 不存在，跳过")
            continue
            
        # 获取所有病例ID (假设每个模态目录中的文件有相同的命名模式)
        adc_dir = os.path.join(case_type_dir, 'ADC')
        if not os.path.exists(adc_dir):
            print(f"警告: {adc_dir} 不存在，跳过{case_type}")
            continue
            
        # 获取ADC目录中的所有文件作为病例基础
        for file in os.listdir(adc_dir):
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                # 提取病例ID (去掉文件扩展名)
                case_id_base = file.split('.')[0]
                case_id = f"{case_type}_{case_id_base}"
                
                print(f"处理病例: {case_id}")
                
                # 收集所有存在的模态文件
                available_modalities = {}
                for modality, suffix in modality_mapping.items():
                    modality_dir = os.path.join(case_type_dir, modality)
                    
                    # 查找模态文件 (支持.nii和.nii.gz)
                    modality_file = None
                    for ext in ['.nii', '.nii.gz']:
                        potential_file = os.path.join(modality_dir, f"{case_id_base}{ext}")
                        if os.path.exists(potential_file):
                            modality_file = potential_file
                            break
                    
                    if modality_file is not None:
                        available_modalities[modality] = (modality_file, suffix)
                    else:
                        print(f"  警告: 病例 {case_id} 的模态 {modality} 文件不存在")
                
                # 检查标签文件是否存在
                label_file = None
                for ext in ['.nii', '.nii.gz']:
                    potential_label = os.path.join(roi_dir, f"{case_id_base}{ext}")
                    if os.path.exists(potential_label):
                        label_file = potential_label
                        break
                
                if label_file is None:
                    print(f"  警告: 病例 {case_id} 的标签文件不存在，跳过该病例")
                    continue
                
                # 复制所有可用的模态文件
                for modality, (source_file, suffix) in available_modalities.items():
                    target_filename = f"{case_id}_{suffix}.nii.gz"
                    target_path = os.path.join(images_tr_dir, target_filename)
                    
                    # 复制文件，确保为.nii.gz格式
                    if source_file.endswith('.nii.gz'):
                        shutil.copy2(source_file, target_path)
                    else:
                        # 如果是.nii文件，加载并保存为.nii.gz
                        img = nib.load(source_file)
                        nib.save(img, target_path)
                    
                    print(f"  复制模态 {modality}: {target_filename}")
                
                # 复制标签文件
                target_label_filename = f"{case_id}.nii.gz"
                target_label_path = os.path.join(labels_tr_dir, target_label_filename)
                
                if label_file.endswith('.nii.gz'):
                    shutil.copy2(label_file, target_label_path)
                else:
                    # 如果是.nii文件，加载并保存为.nii.gz
                    img = nib.load(label_file)
                    nib.save(img, target_label_path)
                
                print(f"  复制标签: {target_label_filename}")
                
                # 添加到训练案例列表 (使用简化格式，符合nnUNet v2规范)
                training_cases.append({
                    "image": f"./imagesTr/{case_id}",
                    "label": f"./labelsTr/{case_id}.nii.gz"
                })
                
                case_count += 1
    
    # 创建dataset.json (符合nnUNet v2格式要求)
    # 根据实际可用的模态动态生成channel_names
    channel_names = {}
    # 统计所有病例中出现的模态
    all_available_modalities = set()
    for case_type in ['BPH', 'PCA']:
        case_type_dir = os.path.join(data_root_dir, 'BPH-PCA', case_type)
        if not os.path.exists(case_type_dir):
            continue
            
        for modality, suffix in modality_mapping.items():
            modality_dir = os.path.join(case_type_dir, modality)
            if os.path.exists(modality_dir):
                # 检查目录中是否有文件
                files = [f for f in os.listdir(modality_dir) if f.endswith(('.nii', '.nii.gz'))]
                if files:
                    all_available_modalities.add(modality)
    
    # 按照原始顺序排列可用模态
    ordered_modalities = [mod for mod in modality_mapping.keys() if mod in all_available_modalities]
    for idx, modality in enumerate(ordered_modalities):
        # 将模态名称转换为适合JSON的格式
        channel_name = modality.replace(" ", "_").replace("-", "_")
        channel_names[str(idx)] = channel_name
    
    dataset_json = {
        "name": "ProstateMultiModal_BPH_PCA",
        "description": "Prostate segmentation with BPH and PCA cases",
        "reference": "Your Institution",
        "licence": "CC-BY-NC-SA 4.0", 
        "release": "1.0",
        "channel_names": channel_names,
        "labels": {
            "background": 0,
            "prostate": 1
        },
        "numTraining": case_count,
        "numTest": 0,
        "file_ending": ".nii.gz",
        "training": training_cases,
        "test": []
    }
    
    # 保存dataset.json
    dataset_json_path = os.path.join(task_dir, 'dataset.json')
    with open(dataset_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)
    
    print(f"\n转换完成!")
    print(f"总病例数: {case_count}")
    print(f"输出目录: {task_dir}")
    print(f"图像文件: {images_tr_dir}")
    print(f"标签文件: {labels_tr_dir}")
    print(f"配置文件: {dataset_json_path}")
    print(f"实际使用的模态: {list(channel_names.values())}")
    
    return task_dir

def validate_nnunet_dataset_flexible(task_dir):
    """
    验证生成的nnU-Net数据集完整性（灵活模式）
    """
    print(f"\n验证数据集完整性...")
    
    dataset_json_path = os.path.join(task_dir, 'dataset.json')
    images_dir = os.path.join(task_dir, 'imagesTr')
    labels_dir = os.path.join(task_dir, 'labelsTr')
    
    # 检查基本文件结构
    if not os.path.exists(dataset_json_path):
        print("错误: dataset.json 不存在")
        return False
    
    if not os.path.exists(images_dir):
        print("错误: imagesTr 目录不存在")
        return False
        
    if not os.path.exists(labels_dir):
        print("错误: labelsTr 目录不存在")
        return False
    
    # 加载dataset.json
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 获取模态数量
    num_modalities = len(dataset['channel_names'])
    print(f"数据集中定义的模态数量: {num_modalities}")
    
    # 检查所有文件是否存在
    missing_files = []
    for case in dataset['training']:
        # 检查图像文件 (根据nnU-Net v2格式)
        image_base = case['image']
        if image_base.startswith('./'):
            image_base = image_base[2:]  # 移除 "./" 前缀
            
        # 检查所有模态文件是否存在
        for i in range(num_modalities):
            modality_file = f"{image_base}_{i:04d}.nii.gz"
            full_path = os.path.join(task_dir, modality_file)
            if not os.path.exists(full_path):
                missing_files.append(full_path)
        
        # 检查标签
        label_path = os.path.join(task_dir, case['label'].lstrip('./'))
        if not os.path.exists(label_path):
            missing_files.append(label_path)
    
    if missing_files:
        print("警告: 以下文件缺失:")
        for file in missing_files:
            print(f"  {file}")
        return False
    else:
        print("✅ 所有文件都存在")
        return True

if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将BPH-PCA多模态前列腺MRI数据转换为nnU-Net v2格式（灵活模式）')
    parser.add_argument('--data_root', default='data', help='原始数据根目录 (默认: data)')
    parser.add_argument('--output_dir', default='nnUNet_raw_data_base/nnUNet_raw_data', 
                       help='输出目录 (默认: nnUNet_raw_data_base/nnUNet_raw_data)')
    parser.add_argument('--task_name', default='Task999_ProstateMultiModal', 
                       help='nnU-Net任务名称 (默认: Task999_ProstateMultiModal)')
    
    args = parser.parse_args()
    
    # 配置路径
    data_root = args.data_root
    output_dir = args.output_dir
    task_name = args.task_name
    
    # 执行转换
    task_dir = convert_bph_pca_to_nnunet_v2_flexible(data_root, output_dir, task_name)
    
    # 验证数据集
    validate_nnunet_dataset_flexible(task_dir)
    
    print(f"\n下一步:")
    print(f"1. 设置环境变量: export nnUNet_raw='{output_dir}'")
    print(f"2. 运行nnUNet规划: nnUNetv2_plan_and_preprocess -d 999 --verify_dataset_integrity")