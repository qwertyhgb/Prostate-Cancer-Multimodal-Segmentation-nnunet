#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速运行BPH-PCA数据转换的脚本

这是一个简化的运行脚本，用于快速执行数据转换
"""

import os
import sys
import argparse
from pathlib import Path

# 添加脚本目录到Python路径
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from convert_bph_pca_to_nnunet import BPHPCAToNnUNetConverter

def main():
    """快速转换函数"""
    print("🚀 BPH-PCA数据转换工具")
    print("=" * 50)
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='BPH-PCA数据转换工具')
    parser.add_argument('--source_dir', type=str, default='data/BPH-PCA',
                       help='BPH-PCA数据源目录 (默认: data/BPH-PCA)')
    parser.add_argument('--output_dir', type=str, default='nnUNet_raw',
                       help='nnU-Net输出目录 (默认: nnUNet_raw)')
    parser.add_argument('--dataset_id', type=int, default=1,
                       help='数据集ID (默认: 1)')
    parser.add_argument('--mode', type=str, default=None,
                       choices=['core_4', 'zero_fill', 'similarity_fill', 'strict_5'],
                       help='处理模式: core_4, zero_fill, similarity_fill, strict_5')
    parser.add_argument('--no_input', action='store_true',
                       help='无交互模式，使用默认设置')
    
    args = parser.parse_args()
    
    # 默认路径设置
    source_dir = args.source_dir
    output_dir = args.output_dir
    dataset_id = args.dataset_id
    
    # 检查源目录是否存在
    if not Path(source_dir).exists():
        print(f"❌ 错误: 源目录不存在 - {source_dir}")
        print("请确保BPH-PCA数据位于正确的目录中")
        return
    
    print(f"📁 源目录: {source_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🏷️  数据集ID: {dataset_id}")
    
    # 处理模式选择
    processing_mode = args.mode
    
    if not args.no_input and processing_mode is None:
        # 询问数据处理策略
        print(f"\n🔧 数据处理策略选项:")
        print(f"1. 核心模态模式: 只使用4个核心模态，确保数据一致性")
        print(f"2. 0填充模式: 使用5个模态，缺失的gaoqing-T2用0填充")
        print(f"3. 相似性填充模式: 使用5个模态，缺失的gaoqing-T2用相似性填充（最推荐）")
        print(f"4. 严格模式: 要求所有5个模态都存在")
        
        mode_choice = input("选择模式 (1/2/3/4, 默认3): ").strip()
        
        if mode_choice == '1':
            processing_mode = 'core_4'
            print("✓ 选择核心模态模式：使用4个核心模态")
        elif mode_choice == '2':
            processing_mode = 'zero_fill'
            print("✓ 选择0填充模式：缺失的gaoqing-T2用0填充")
        elif mode_choice == '4':
            processing_mode = 'strict_5'
            print("✓ 选择严格模式：要求所有5个模态")
        else:
            processing_mode = 'similarity_fill'
            print("✓ 选择相似性填充模式：缺失的gaoqing-T2用相似性填充")
            print("💡 优势：保留T2加权信息，比0填充效果更好")
    elif processing_mode is None:
        # 默认使用相似性填充模式
        processing_mode = 'similarity_fill'
        print("✓ 使用默认模式：相似性填充模式")
    
    # 询问用户确认（除非使用无交互模式）
    if not args.no_input:
        response = input("\n是否继续转换? (y/n): ").lower().strip()
        if response != 'y' and response != 'yes':
            print("❌ 转换已取消")
            return
    
    try:
        # 创建转换器
        converter = BPHPCAToNnUNetConverter(
            source_dir=source_dir,
            output_dir=output_dir,
            dataset_id=dataset_id,
            processing_mode=processing_mode
        )
        
        # 执行转换
        processed_cases = converter.convert()
        
        if processed_cases:
            print(f"\n🎉 转换成功完成！")
            print(f"📊 处理了 {len(processed_cases)} 个病例")
            
            # 显示下一步操作
            print(f"\n💡 接下来的步骤:")
            print(f"1. 设置nnU-Net环境变量:")
            print(f"   export nnUNet_raw=\"{Path(output_dir).absolute()}\"")
            print(f"   export nnUNet_preprocessed=\"path/to/nnUNet_preprocessed\"")
            print(f"   export nnUNet_results=\"path/to/nnUNet_results\"")
            print(f"\n2. 运行nnU-Net预处理:")
            print(f"   nnUNetv2_plan_and_preprocess -d {dataset_id}")
            print(f"\n3. 开始训练:")
            print(f"   nnUNetv2_train {dataset_id} 3d_fullres 0")
            
            # 提供快速训练命令
            print(f"\n⚡ 快速开始训练:")
            print(f"   nnUNetv2_train {dataset_id} 3d_fullres 0 --npz")
        else:
            print("❌ 没有成功处理任何病例，请检查数据格式")
            
    except Exception as e:
        print(f"❌ 转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()