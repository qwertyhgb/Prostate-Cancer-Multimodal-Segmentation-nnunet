#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import argparse
from collections import defaultdict

def create_grouped_datasets(original_dataset_dir, output_base_dir):
    """
    根据模态完整性创建分组数据集
    
    参数:
        original_dataset_dir (str): 原始数据集目录
        output_base_dir (str): 输出基础目录
        
    返回:
        dict: 分组信息
    """
    print("=" * 60)
    print("创建分组数据集")
    print("=" * 60)
    
    # 检查原始数据集
    if not os.path.exists(original_dataset_dir):
        raise FileNotFoundError(f"原始数据集目录 {original_dataset_dir} 不存在")
    
    dataset_json_path = os.path.join(original_dataset_dir, 'dataset.json')
    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError("原始数据集中的 dataset.json 文件不存在")
    
    # 读取原始数据集信息
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        original_dataset = json.load(f)
    
    print(f"📊 原始数据集包含 {original_dataset['numTraining']} 个训练病例")
    
    # 分析每个病例的模态完整性
    complete_cases = []  # 包含所有模态的病例
    missing_cases = []   # 缺失gaoqing-T2模态的病例
    
    images_dir = os.path.join(original_dataset_dir, 'imagesTr')
    labels_dir = os.path.join(original_dataset_dir, 'labelsTr')
    
    # 检查每个病例
    for case in original_dataset['training']:
        image_base = case['image']
        if image_base.startswith('./'):
            image_base = image_base[2:]
        
        # 检查模态文件数量
        modality_count = 0
        mod_idx = 0
        while True:
            modality_file = f"{image_base}_{mod_idx:04d}.nii.gz"
            full_image_path = os.path.join(original_dataset_dir, modality_file)
            
            if os.path.exists(full_image_path):
                modality_count += 1
                mod_idx += 1
            else:
                break
        
        # 根据模态数量分类
        if modality_count == 5:
            complete_cases.append(case)
        elif modality_count == 4:
            missing_cases.append(case)
        else:
            print(f"  ⚠️  案例 {image_base} 有异常模态数量: {modality_count}")
    
    print(f"✅ 完整模态病例数: {len(complete_cases)}")
    print(f"✅ 缺失模态病例数: {len(missing_cases)}")
    
    # 创建完整模态数据集 (Group A)
    group_a_dir = os.path.join(output_base_dir, 'Dataset001_ProstateBPHPCA_GroupA')
    create_group_dataset(group_a_dir, original_dataset, complete_cases, "完整模态组")
    
    # 创建缺失模态数据集 (Group B)
    group_b_dir = os.path.join(output_base_dir, 'Dataset002_ProstateBPHPCA_GroupB')
    create_group_dataset(group_b_dir, original_dataset, missing_cases, "缺失模态组", 
                        exclude_modality='gaoqing_T2')
    
    # 生成分组信息报告
    grouping_info = {
        'group_a': {
            'name': '完整模态组',
            'dataset_id': 1,
            'dataset_name': 'Dataset001_ProstateBPHPCA_GroupA',
            'cases_count': len(complete_cases),
            'modalities': ['ADC', 'DWI', 'T2_fs', 'T2_not_fs', 'gaoqing_T2']
        },
        'group_b': {
            'name': '缺失模态组',
            'dataset_id': 2,
            'dataset_name': 'Dataset002_ProstateBPHPCA_GroupB',
            'cases_count': len(missing_cases),
            'modalities': ['ADC', 'DWI', 'T2_fs', 'T2_not_fs']
        }
    }
    
    # 保存分组信息
    grouping_info_path = os.path.join(output_base_dir, 'grouping_info.json')
    with open(grouping_info_path, 'w', encoding='utf-8') as f:
        json.dump(grouping_info, f, indent=4, ensure_ascii=False)
    
    print(f"\n💾 分组信息已保存到: {grouping_info_path}")
    
    # 显示使用说明
    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    print("训练完整模态组模型:")
    print("  nnUNetv2_train 1 3d_fullres 0")
    print("\n训练缺失模态组模型:")
    print("  nnUNetv2_train 2 3d_fullres 0")
    print("\n设置环境变量:")
    print("  export nnUNet_raw='nnUNet_raw_data_base'")
    print("  export nnUNet_preprocessed='nnUNet_preprocessed'")
    print("  export nnUNet_results='nnUNet_trained_models'")
    
    return grouping_info

def create_group_dataset(dataset_dir, original_dataset, cases, group_name, exclude_modality=None):
    """
    创建分组数据集
    
    参数:
        dataset_dir (str): 数据集目录
        original_dataset (dict): 原始数据集信息
        cases (list): 该组的病例列表
        group_name (str): 组名称
        exclude_modality (str): 要排除的模态名称
    """
    print(f"\n📁 创建 {group_name}: {dataset_dir}")
    
    # 创建目录结构
    images_tr_dir = os.path.join(dataset_dir, 'imagesTr')
    labels_tr_dir = os.path.join(dataset_dir, 'labelsTr')
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)
    
    # 复制图像文件
    print(f"  📎 复制 {len(cases)} 个病例的图像文件...")
    for case in cases:
        image_base = case['image']
        if image_base.startswith('./'):
            image_base = image_base[2:]
        
        # 复制所有相关图像文件
        mod_idx = 0
        while True:
            modality_file = f"{image_base}_{mod_idx:04d}.nii.gz"
            src_path = os.path.join(os.path.dirname(dataset_dir), 
                                   os.path.basename(os.path.dirname(dataset_dir)).replace('_GroupA', '').replace('_GroupB', ''),
                                   modality_file)
            
            if os.path.exists(src_path):
                dst_path = os.path.join(images_tr_dir, os.path.basename(modality_file))
                shutil.copy2(src_path, dst_path)
                mod_idx += 1
            else:
                break
    
    # 复制标签文件
    print(f"  📎 复制 {len(cases)} 个病例的标签文件...")
    for case in cases:
        label_path = case['label']
        if label_path.startswith('./'):
            label_path = label_path[2:]
        
        src_path = os.path.join(os.path.dirname(dataset_dir),
                               os.path.basename(os.path.dirname(dataset_dir)).replace('_GroupA', '').replace('_GroupB', ''),
                               label_path)
        dst_path = os.path.join(labels_tr_dir, os.path.basename(label_path))
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    # 创建新的dataset.json
    if exclude_modality:
        # 为缺失模态组创建4模态配置
        new_channel_names = {}
        original_channels = original_dataset['channel_names']
        channel_idx = 0
        for i in range(5):  # 原始5个模态
            modality_name = original_channels.get(str(i), f"modality_{i:04d}")
            if modality_name != exclude_modality:
                new_channel_names[str(channel_idx)] = modality_name
                channel_idx += 1
    else:
        # 为完整模态组保留所有模态
        new_channel_names = original_dataset['channel_names']
    
    new_dataset = {
        "name": f"ProstateMultiModal_BPH_PCA_{group_name}",
        "description": f"Prostate segmentation with BPH and PCA cases - {group_name}",
        "reference": "Your Institution",
        "licence": "CC-BY-NC-SA 4.0",
        "release": "1.0",
        "channel_names": new_channel_names,
        "labels": original_dataset['labels'],
        "numTraining": len(cases),
        "numTest": 0,
        "file_ending": original_dataset['file_ending'],
        "training": cases,
        "test": []
    }
    
    dataset_json_path = os.path.join(dataset_dir, 'dataset.json')
    with open(dataset_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, indent=4, ensure_ascii=False)
    
    print(f"  ✅ {group_name} 数据集创建完成")

def main():
    parser = argparse.ArgumentParser(description='创建分组数据集以处理模态缺失问题')
    parser.add_argument('--original_dataset_dir', required=True, 
                       help='原始数据集目录')
    parser.add_argument('--output_base_dir', default='nnUNet_raw_data_base',
                       help='输出基础目录 (默认: nnUNet_raw_data_base)')
    
    args = parser.parse_args()
    
    try:
        grouping_info = create_grouped_datasets(args.original_dataset_dir, args.output_base_dir)
        print(f"\n🎉 分组数据集创建成功!")
        print(f"   完整模态组: {grouping_info['group_a']['cases_count']} 个病例")
        print(f"   缺失模态组: {grouping_info['group_b']['cases_count']} 个病例")
        return 0
    except Exception as e:
        print(f"❌ 创建分组数据集时出现错误: {e}")
        return 1

if __name__ == "__main__":
    exit(main())