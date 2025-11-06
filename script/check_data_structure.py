#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPH-PCA数据结构检查工具

用于验证数据是否符合转换要求
"""

import os
from pathlib import Path
from collections import defaultdict

def check_data_structure(data_dir: str = "data/BPH-PCA"):
    """检查BPH-PCA数据结构"""
    
    print("🔍 BPH-PCA数据结构检查")
    print("=" * 50)
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    print(f"📁 检查目录: {data_path.absolute()}")
    
    # 检查主要目录
    required_dirs = ['BPH', 'PCA', 'ROI(BPH+PCA)']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
        else:
            print(f"✅ 找到目录: {dir_name}")
    
    if missing_dirs:
        print(f"❌ 缺少目录: {missing_dirs}")
        return False
    
    # 检查模态目录
    modalities = ['ADC', 'DWI', 'gaoqing-T2', 'T2 fs', 'T2 not fs']
    
    print(f"\n📋 检查模态目录:")
    for category in ['BPH', 'PCA']:
        print(f"\n  {category}:")
        category_path = data_path / category
        
        for modality in modalities:
            modality_path = category_path / modality
            if modality_path.exists():
                nii_files = list(modality_path.glob("*.nii"))
                print(f"    ✅ {modality}: {len(nii_files)} 个文件")
            else:
                print(f"    ❌ {modality}: 目录不存在")
    
    # 检查ROI目录
    print(f"\n📋 检查ROI目录:")
    roi_path = data_path / "ROI(BPH+PCA)"
    
    for category in ['BPH', 'PCA']:
        category_roi_path = roi_path / category
        if category_roi_path.exists():
            roi_files = list(category_roi_path.glob("*.nii"))
            print(f"  ✅ {category} ROI: {len(roi_files)} 个文件")
        else:
            print(f"  ❌ {category} ROI: 目录不存在")
    
    # 统计病例数量
    print(f"\n📊 病例统计:")
    case_stats = defaultdict(lambda: defaultdict(int))
    
    for category in ['BPH', 'PCA']:
        category_path = data_path / category
        if not category_path.exists():
            continue
            
        # 统计每个模态的病例数
        for modality in modalities:
            modality_path = category_path / modality
            if modality_path.exists():
                nii_files = list(modality_path.glob("*.nii"))
                case_ids = [f.stem for f in nii_files]
                case_stats[category][modality] = len(case_ids)
        
        # 统计ROI
        roi_category_path = roi_path / category
        if roi_category_path.exists():
            roi_files = list(roi_category_path.glob("*.nii"))
            case_stats[category]['ROI'] = len(roi_files)
    
    # 显示统计结果
    for category, modality_counts in case_stats.items():
        print(f"\n  {category}:")
        for modality, count in modality_counts.items():
            print(f"    {modality}: {count} 例")
    
    # 检查数据一致性
    print(f"\n🔍 数据一致性检查:")
    
    for category in ['BPH', 'PCA']:
        if category not in case_stats:
            continue
            
        print(f"\n  {category}:")
        
        # 获取所有模态的病例ID
        all_case_ids = {}
        category_path = data_path / category
        
        for modality in modalities:
            modality_path = category_path / modality
            if modality_path.exists():
                nii_files = list(modality_path.glob("*.nii"))
                case_ids = set(f.stem for f in nii_files)
                all_case_ids[modality] = case_ids
        
        # 获取ROI病例ID
        roi_category_path = roi_path / category
        if roi_category_path.exists():
            roi_files = list(roi_category_path.glob("*.nii"))
            roi_case_ids = set(f.stem for f in roi_files)
            all_case_ids['ROI'] = roi_case_ids
        
        # 找到共同的病例ID
        if all_case_ids:
            common_cases = set.intersection(*all_case_ids.values())
            print(f"    共同病例数: {len(common_cases)}")
            
            # 检查每个模态缺失的病例
            for modality, case_ids in all_case_ids.items():
                missing = common_cases - case_ids
                if missing:
                    print(f"    ⚠️  {modality} 缺失病例: {len(missing)} 个")
                else:
                    print(f"    ✅ {modality}: 完整")
    
    print(f"\n✅ 数据结构检查完成")
    return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检查BPH-PCA数据结构')
    parser.add_argument('--data_dir', type=str, default='data/BPH-PCA',
                       help='BPH-PCA数据目录')
    
    args = parser.parse_args()
    
    success = check_data_structure(args.data_dir)
    
    if success:
        print(f"\n💡 数据结构检查通过，可以进行转换")
        print(f"   运行: python script/run_conversion.py")
    else:
        print(f"\n❌ 数据结构检查失败，请修正后重试")

if __name__ == "__main__":
    main()