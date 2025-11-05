#!/usr/bin/env python3
"""
检查并修复数据集中的图像路径格式问题
"""

import json
import os

def check_and_fix_dataset_json():
    # 数据集路径
    dataset_path = "nnUNet_raw_data_base/nnUNet_raw_data/Task999_ProstateMultiModal"
    dataset_json_path = os.path.join(dataset_path, "dataset.json")
    
    # 读取数据集信息
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    print(f"检查前的训练样本数: {len(dataset_info['training'])}")
    
    # 检查图像路径格式
    for i, item in enumerate(dataset_info['training']):
        image_path = item['image']
        label_path = item['label']
        
        # 检查image路径是否包含模态后缀
        if not image_path.endswith(('_0000', '_0001', '_0002', '_0003', '_0004')):
            # 修正image路径，移除可能的文件扩展名，添加默认的模态后缀
            if image_path.endswith('.nii.gz'):
                image_path = image_path[:-7]  # 移除 .nii.gz
            elif image_path.endswith('.nii'):
                image_path = image_path[:-4]   # 移除 .nii
            
            # 更新image路径
            dataset_info['training'][i]['image'] = image_path
    
    # 保存修复后的数据集
    with open(dataset_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)
    
    print("数据集路径格式检查和修复完成!")

if __name__ == "__main__":
    check_and_fix_dataset_json()