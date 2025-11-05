#!/usr/bin/env python3
"""
修复数据集中的重复条目问题
"""

import json
import os

def fix_dataset_json():
    # 数据集路径
    dataset_path = "nnUNet_raw_data_base/nnUNet_raw_data/Task999_ProstateMultiModal"
    dataset_json_path = os.path.join(dataset_path, "dataset.json")
    
    # 读取数据集信息
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    print(f"修复前的训练样本数: {len(dataset_info['training'])}")
    
    # 使用字典去重，以image字段为键
    unique_training = {}
    for item in dataset_info['training']:
        # 使用image路径作为唯一标识符
        identifier = item['image']
        if identifier not in unique_training:
            unique_training[identifier] = item
        else:
            print(f"发现重复项: {identifier}")
    
    # 更新训练数据
    dataset_info['training'] = list(unique_training.values())
    dataset_info['numTraining'] = len(dataset_info['training'])
    
    print(f"修复后的训练样本数: {len(dataset_info['training'])}")
    
    # 保存修复后的数据集
    with open(dataset_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)
    
    print("数据集修复完成!")

if __name__ == "__main__":
    fix_dataset_json()