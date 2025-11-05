#!/usr/bin/env python3
"""
将BPH-PCA多模态前列腺MRI数据(.nii格式)转换为nnU-Net格式
"""

import os
import json
import nibabel as nib
import numpy as np
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_bph_pca_nii_to_nnunet(base_path="data/BPH-PCA", 
                                  nnunet_raw_base="nnUNet_raw_data_base/nnUNet_raw_data"):
    # 配置路径
    task_name = "Task999_ProstateMultiModal"
    task_path = os.path.join(nnunet_raw_base, task_name)
    
    # 创建目录
    os.makedirs(os.path.join(task_path, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(task_path, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(task_path, "imagesTs"), exist_ok=True)
    
    # 模态映射
    modalities = {
        'ADC': '0000',
        'DWI': '0001', 
        'T2 fs': '0002',
        'T2 not fs': '0003',
        'gaoqing-T2': '0004'
    }
    
    training_data = []
    case_count = 0
    
    # 处理BPH和PCA病例
    for disease_type in ['BPH', 'PCA']:
        logger.info(f"处理 {disease_type} 病例...")
        
        # 找出该疾病类型的所有病例
        cases = set()
        for modality in modalities.keys():
            modality_path = os.path.join(base_path, disease_type, modality)
            if os.path.exists(modality_path):
                for file in os.listdir(modality_path):
                    if file.endswith('.nii') and not file.endswith('.nii.gz'):
                        case_id = file.replace('.nii', '')
                        cases.add(case_id)
        
        logger.info(f"找到 {disease_type} 病例: {len(cases)}")
        
        # 处理每个病例
        for idx, case_id in enumerate(cases):
            nnunet_case_id = f"{disease_type}_{case_id}"
            missing_modalities = []
            
            # 复制所有模态
            for modality, channel in modalities.items():
                source_path = os.path.join(base_path, disease_type, modality, f"{case_id}.nii")
                
                if os.path.exists(source_path):
                    target_path = os.path.join(task_path, "imagesTr", f"{nnunet_case_id}_{channel}.nii.gz")
                    
                    # 使用nibabel读取并保存为.nii.gz格式
                    try:
                        img = nib.load(source_path)
                        # 确保数据是正确的方向
                        if len(img.shape) == 3:  # 3D volume
                            nib.save(img, target_path)
                        else:
                            logger.warning(f"{source_path} 不是3D体积，跳过")
                            missing_modalities.append(modality)
                    except Exception as e:
                        logger.error(f"错误处理 {source_path}: {e}")
                        missing_modalities.append(modality)
                else:
                    logger.warning(f"病例 {nnunet_case_id} 缺失模态 {modality}")
                    missing_modalities.append(modality)
            
            # 复制标签
            label_source = os.path.join(base_path, 'ROI(BPH+PCA)', disease_type, f"{case_id}.nii")
            label_found = False
            
            if os.path.exists(label_source):
                label_target = os.path.join(task_path, "labelsTr", f"{nnunet_case_id}.nii.gz")
                try:
                    label_img = nib.load(label_source)
                    # 确保标签是整数类型
                    label_data = label_img.get_fdata()
                    if not np.issubdtype(label_data.dtype, np.integer):
                        logger.info(f"将标签 {label_source} 转换为整数类型")
                        label_data = label_data.astype(np.uint8)
                        label_img = nib.Nifti1Image(label_data, label_img.affine, label_img.header)
                    
                    nib.save(label_img, label_target)
                    label_found = True
                except Exception as e:
                    logger.error(f"错误处理标签 {label_source}: {e}")
            else:
                logger.warning(f"病例 {nnunet_case_id} 缺失标签")
            
            # 如果病例完整或只有少量模态缺失，添加到训练列表
            if label_found and len(missing_modalities) < 3:  # 允许最多缺失2个模态
                # 构建正确的图像路径列表（nnU-Net格式）
                # 修复：使用正确的nnU-Net图像路径格式
                training_data.append({
                    "image": f"./imagesTr/{nnunet_case_id}",
                    "label": f"./labelsTr/{nnunet_case_id}.nii.gz"
                })
                case_count += 1
                logger.info(f"成功添加病例: {nnunet_case_id} (缺失模态: {missing_modalities})")
            else:
                logger.info(f"跳过病例: {nnunet_case_id} (标签缺失或太多模态缺失)")
                
            # 显示进度
            if (idx + 1) % 10 == 0:
                logger.info(f"已处理 {disease_type} 病例: {idx + 1}/{len(cases)}")
    
    # 创建dataset.json，符合nnUNet v2格式要求
    dataset_info = {
        "name": "ProstateMultiModal_BPH_PCA",
        "description": "Prostate segmentation with BPH and PCA cases",
        "reference": "Your Institution",
        "licence": "CC-BY-NC-SA 4.0",
        "release": "1.0",
        # 修改：使用channel_names替代modality以符合nnUNet v2格式要求
        "channel_names": {
            "0": "ADC",
            "1": "DWI",  # 修正：与字典中保持一致
            "2": "T2_fs", 
            "3": "T2_not_fs",
            "4": "gaoqing-T2"  # 修正：与字典中保持一致
        },
        "labels": {
            "0": "background",
            "1": "prostate"
        },
        "numTraining": case_count,
        "numTest": 0,
        # 添加：指定文件扩展名以符合nnUNet v2格式要求
        "file_ending": ".nii.gz",
        "training": training_data,
        "test": []
    }
    
    with open(os.path.join(task_path, "dataset.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)
    
    logger.info(f"\n转换完成! 成功处理 {case_count} 个病例")
    logger.info(f"数据保存在: {task_path}")
    
    return case_count, task_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将BPH-PCA多模态前列腺MRI数据转换为nnU-Net格式')
    parser.add_argument('--base_path', type=str, default="data/BPH-PCA", 
                        help='源数据路径')
    parser.add_argument('--output_path', type=str, default="nnUNet_raw_data_base/nnUNet_raw_data",
                        help='输出数据路径')
    
    args = parser.parse_args()
    
    try:
        case_count, task_path = convert_bph_pca_nii_to_nnunet(args.base_path, args.output_path)
    except Exception as e:
        logger.error(f"转换过程中发生错误: {e}")
        raise