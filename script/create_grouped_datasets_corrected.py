#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import argparse
from collections import defaultdict

def create_grouped_datasets_corrected(original_dataset_dir, output_base_dir):
    """
    根据模态完整性创建分组数据集（正确版）
    
    参数:
        original_dataset_dir (str): 原始数据集目录
        output_base_dir (str): 输出基础目录
        
    返回:
        dict: 分组信息
    """
    print("=" * 60)
    print("创建分组数据集（正确版）")
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
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"原始数据集 imagesTr 目录不存在: {images_dir}")
    
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"原始数据集 labelsTr 目录不存在: {labels_dir}")
    
    # 检查每个病例
    print("🔍 分析每个病例的模态完整性...")
    case_analysis = []
    
    for i, case in enumerate(original_dataset['training']):
        image_base = case['image']
        label_path = case['label']
        
        if image_base.startswith('./'):
            image_base = image_base[2:]  # 移除 "./" 前缀
        
        # 检查模态文件数量 (0000到0004共5个模态)
        modality_files = []
        for mod_idx in range(5):
            modality_file = f"{os.path.basename(image_base)}_{mod_idx:04d}.nii.gz"
            full_image_path = os.path.join(images_dir, modality_file)
            
            if os.path.exists(full_image_path):
                modality_files.append(modality_file)
        
        case_info = {
            'case': case,
            'modality_files': modality_files,
            'count': len(modality_files)
        }
        case_analysis.append(case_info)
        
        # 显示前几个案例作为示例
        if i < 5:
            print(f"   案例: {os.path.basename(image_base)} - {len(modality_files)} 个模态")
    
    # 分类病例
    for case_info in case_analysis:
        if case_info['count'] == 5:
            complete_cases.append(case_info)
        elif case_info['count'] == 4:
            missing_cases.append(case_info)
    
    print(f"\n✅ 分析完成:")
    print(f"   完整模态病例数 (5个模态): {len(complete_cases)}")
    print(f"   缺失模态病例数 (4个模态): {len(missing_cases)}")
    print(f"   异常病例数 (其他模态数): {len(case_analysis) - len(complete_cases) - len(missing_cases)}")
    
    # 创建完整模态数据集 (Group A)
    group_a_dir = os.path.join(output_base_dir, 'Dataset999_ProstateBPHPCA_GroupA')
    create_group_dataset_corrected(group_a_dir, original_dataset, complete_cases, "完整模态组")
    
    # 创建缺失模态数据集 (Group B)
    group_b_dir = os.path.join(output_base_dir, 'Dataset002_ProstateBPHPCA_GroupB')
    create_group_dataset_corrected(group_b_dir, original_dataset, missing_cases, "缺失模态组", 
                                  exclude_modality_index=4)  # gaoqing-T2是第5个模态(索引4)
    
    # 生成分组信息报告
    grouping_info = {
        'group_a': {
            'name': '完整模态组',
            'dataset_id': 1,
            'dataset_name': 'Dataset999_ProstateBPHPCA_GroupA',
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
    print("  nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")
    print("  nnUNetv2_train 1 3d_fullres 0")
    print("\n训练缺失模态组模型:")
    print("  nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity")
    print("  nnUNetv2_train 2 3d_fullres 0")
    print("\n设置环境变量:")
    print("  export nnUNet_raw='nnUNet_raw_data_base'")
    print("  export nnUNet_preprocessed='nnUNet_preprocessed'")
    print("  export nnUNet_results='nnUNet_trained_models'")
    
    return grouping_info

def create_group_dataset_corrected(dataset_dir, original_dataset, cases_with_files, group_name, exclude_modality_index=None):
    """
    创建分组数据集（正确版）
    
    参数:
        dataset_dir (str): 数据集目录
        original_dataset (dict): 原始数据集信息
        cases_with_files (list): 该组的病例列表及文件列表
        group_name (str): 组名称
        exclude_modality_index (int): 要排除的模态索引
    """
    print(f"\n📁 创建 {group_name}: {dataset_dir}")
    
    # 创建目录结构
    images_tr_dir = os.path.join(dataset_dir, 'imagesTr')
    labels_tr_dir = os.path.join(dataset_dir, 'labelsTr')
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)
    
    print(f"  📎 复制 {len(cases_with_files)} 个病例的文件...")
    
    # 复制图像和标签文件
    original_images_dir = os.path.join(os.path.dirname(dataset_dir), 
                                      'Dataset999_ProstateBPHPCA',
                                      'imagesTr')
    original_labels_dir = os.path.join(os.path.dirname(dataset_dir),
                                      'Dataset999_ProstateBPHPCA',
                                      'labelsTr')
    
    copied_cases = []
    for i, case_info in enumerate(cases_with_files):
        case = case_info['case']
        modality_files = case_info['modality_files']
        
        image_base = case['image']
        label_path = case['label']
        
        if image_base.startswith('./'):
            image_base = image_base[2:]
        
        if label_path.startswith('./'):
            label_path = label_path[2:]
        
        # 复制模态文件
        for modality_file in modality_files:
            src_path = os.path.join(original_images_dir, modality_file)
            dst_path = os.path.join(images_tr_dir, modality_file)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"     警告: 源文件不存在 {src_path}")
        
        # 复制标签文件
        src_label_path = os.path.join(original_labels_dir, os.path.basename(label_path))
        dst_label_path = os.path.join(labels_tr_dir, os.path.basename(label_path))
        
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
            copied_cases.append(case)
        else:
            print(f"     警告: 标签文件不存在 {src_label_path}")
        
        if (i + 1) % 50 == 0:
            print(f"     已处理 {i + 1} 个病例...")
    
    print(f"  ✅ 成功复制 {len(copied_cases)} 个病例的文件")
    
    # 创建新的dataset.json
    if exclude_modality_index is not None:
        # 为缺失模态组创建4模态配置
        new_channel_names = {}
        original_channels = original_dataset['channel_names']
        channel_idx = 0
        for i in range(5):  # 原始5个模态
            if i != exclude_modality_index:
                modality_name = original_channels.get(str(i), f"modality_{i:04d}")
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
        "numTraining": len(copied_cases),
        "numTest": 0,
        "file_ending": original_dataset['file_ending'],
        "training": copied_cases,
        "test": []
    }
    
    dataset_json_path = os.path.join(dataset_dir, 'dataset.json')
    with open(dataset_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, indent=4, ensure_ascii=False)
    
    print(f"  ✅ {group_name} 数据集创建完成，包含 {len(copied_cases)} 个病例")

def main():
    parser = argparse.ArgumentParser(description='创建分组数据集以处理模态缺失问题（正确版）')
    parser.add_argument('--original_dataset_dir', required=True, 
                       help='原始数据集目录')
    parser.add_argument('--output_base_dir', default='nnUNet_raw_data_base',
                       help='输出基础目录 (默认: nnUNet_raw_data_base)')
    
    args = parser.parse_args()
    
    try:
        grouping_info = create_grouped_datasets_corrected(args.original_dataset_dir, args.output_base_dir)
        print(f"\n🎉 分组数据集创建成功!")
        print(f"   完整模态组: {grouping_info['group_a']['cases_count']} 个病例")
        print(f"   缺失模态组: {grouping_info['group_b']['cases_count']} 个病例")
        return 0
    except Exception as e:
        print(f"❌ 创建分组数据集时出现错误: {e}")
        return 1

if __name__ == "__main__":
    exit(main())