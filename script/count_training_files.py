import os
import argparse
import json
from pathlib import Path

def count_training_data(data_root_dir, task_dir):
    """
    统计训练文件和标签的数量
    
    参数:
        data_root_dir: 原始数据根目录
        task_dir: 处理后的任务目录
    """
    # 统计原始数据
    print("=" * 50)
    print("原始数据统计")
    print("=" * 50)
    
    total_cases = 0
    bph_cases = 0
    pca_cases = 0
    
    # 检查BPH数据
    bph_dir = os.path.join(data_root_dir, 'BPH-PCA', 'BPH')
    if os.path.exists(bph_dir):
        adc_dir = os.path.join(bph_dir, 'ADC')
        if os.path.exists(adc_dir):
            bph_files = [f for f in os.listdir(adc_dir) if f.endswith(('.nii', '.nii.gz'))]
            bph_cases = len(bph_files)
            print(f"BPH病例数: {bph_cases}")
            
            # 显示各模态情况
            modalities = ['ADC', 'DWI', 'T2 fs', 'T2 not fs', 'gaoqing-T2']
            for modality in modalities:
                modality_dir = os.path.join(bph_dir, modality)
                if os.path.exists(modality_dir):
                    modality_files = [f for f in os.listdir(modality_dir) if f.endswith(('.nii', '.nii.gz'))]
                    print(f"  {modality}: {len(modality_files)} 文件")
                else:
                    print(f"  {modality}: 0 文件 (目录不存在)")
    
    # 检查PCA数据
    pca_dir = os.path.join(data_root_dir, 'BPH-PCA', 'PCA')
    if os.path.exists(pca_dir):
        adc_dir = os.path.join(pca_dir, 'ADC')
        if os.path.exists(adc_dir):
            pca_files = [f for f in os.listdir(adc_dir) if f.endswith(('.nii', '.nii.gz'))]
            pca_cases = len(pca_files)
            print(f"PCA病例数: {pca_cases}")
            
            # 显示各模态情况
            modalities = ['ADC', 'DWI', 'T2 fs', 'T2 not fs', 'gaoqing-T2']
            for modality in modalities:
                modality_dir = os.path.join(pca_dir, modality)
                if os.path.exists(modality_dir):
                    modality_files = [f for f in os.listdir(modality_dir) if f.endswith(('.nii', '.nii.gz'))]
                    print(f"  {modality}: {len(modality_files)} 文件")
                else:
                    print(f"  {modality}: 0 文件 (目录不存在)")
    
    total_cases = bph_cases + pca_cases
    print(f"原始数据总病例数: {total_cases}")
    
    # 统计处理后的数据
    print("\n" + "=" * 50)
    print("处理后的数据统计")
    print("=" * 50)
    
    images_tr_dir = os.path.join(task_dir, 'imagesTr')
    labels_tr_dir = os.path.join(task_dir, 'labelsTr')
    dataset_json_path = os.path.join(task_dir, 'dataset.json')
    
    if os.path.exists(task_dir):
        # 统计图像文件
        if os.path.exists(images_tr_dir):
            # 每个病例有5个模态文件
            image_files = [f for f in os.listdir(images_tr_dir) if f.endswith('.nii.gz')]
            unique_cases_from_images = len(set([f.split('_')[0] + '_' + f.split('_')[1] for f in image_files if '_' in f and len(f.split('_')) >= 2]))
            print(f"处理后的图像文件数: {len(image_files)} ({unique_cases_from_images} 个病例，每个病例5个模态)")
        else:
            print("imagesTr 目录不存在")
            
        # 统计标签文件
        if os.path.exists(labels_tr_dir):
            label_files = [f for f in os.listdir(labels_tr_dir) if f.endswith('.nii.gz')]
            print(f"处理后的标签文件数: {len(label_files)}")
        else:
            print("labelsTr 目录不存在")
            
        # 检查dataset.json
        if os.path.exists(dataset_json_path):
            with open(dataset_json_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
            print(f"dataset.json 中记录的训练病例数: {dataset_info.get('numTraining', 0)}")
            
            # 显示模态信息
            if 'channel_names' in dataset_info:
                print(f"数据集中的模态: {list(dataset_info['channel_names'].values())}")
        else:
            print("dataset.json 文件不存在")
    else:
        print("处理后的任务目录不存在")

def main():
    parser = argparse.ArgumentParser(description='统计训练文件和标签的数量')
    parser.add_argument('--data_root', default='data', help='原始数据根目录')
    parser.add_argument('--output_dir', default='nnUNet_raw_data_base/nnUNet_raw_data', help='输出目录')
    parser.add_argument('--task_name', default='Task999_ProstateMultiModal', help='任务名称')
    
    args = parser.parse_args()
    
    data_root_dir = args.data_root
    task_dir = os.path.join(args.output_dir, args.task_name)
    
    count_training_data(data_root_dir, task_dir)

if __name__ == "__main__":
    main()