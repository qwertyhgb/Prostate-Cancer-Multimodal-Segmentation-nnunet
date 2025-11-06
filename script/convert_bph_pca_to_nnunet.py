#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPH-PCA多模态前列腺MRI数据转换为nnU-Net v2格式

该脚本将BPH-PCA数据集转换为nnU-Net v2所需的格式：
- 支持多模态MRI数据（ADC, DWI, T2等）
- 自动处理BPH和PCA两类数据
- 生成符合nnU-Net v2命名规范的文件
- 创建数据集描述文件
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import zoom

class BPHPCAToNnUNetConverter:
    """BPH-PCA数据转换器"""
    
    def __init__(self, source_dir: str, output_dir: str, dataset_id: int = 1, 
                 processing_mode: str = 'zero_fill'):
        """
        初始化转换器
        
        Args:
            source_dir: BPH-PCA数据源目录
            output_dir: nnU-Net输出目录
            dataset_id: 数据集ID（默认1）
            processing_mode: 处理模式 ('core_4', 'zero_fill', 'strict_5')
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.dataset_id = dataset_id
        self.dataset_name = f"Dataset{dataset_id:03d}_BPH_PCA"
        self.processing_mode = processing_mode
        
        # 根据处理模式设置参数
        if processing_mode == 'core_4':
            # 只使用4个核心模态
            self.modality_mapping = {
                'ADC': '0000',
                'DWI': '0001', 
                'T2 fs': '0002',
                'T2 not fs': '0003'
            }
            self.zero_fill_missing = False
            self.min_modalities = 4
            print("🔧 核心模态模式：使用ADC, DWI, T2 fs, T2 not fs（4通道）")
            
        elif processing_mode == 'zero_fill':
            # 使用5个模态，缺失的用0填充
            self.modality_mapping = {
                'ADC': '0000',
                'DWI': '0001', 
                'T2 fs': '0002',
                'T2 not fs': '0003',
                'gaoqing-T2': '0004'
            }
            self.zero_fill_missing = True
            self.min_modalities = 4
            print("🔧 0填充模式：使用5个模态，缺失的gaoqing-T2用0填充（5通道）")
            
        elif processing_mode == 'similarity_fill':
            # 使用5个模态，缺失的用相似性填充
            self.modality_mapping = {
                'ADC': '0000',
                'DWI': '0001', 
                'T2 fs': '0002',
                'T2 not fs': '0003',
                'gaoqing-T2': '0004'
            }
            self.zero_fill_missing = True
            self.similarity_fill = True
            self.min_modalities = 4
            print("🔧 相似性填充模式：使用5个模态，缺失的gaoqing-T2用相似性填充（5通道）")
            
        elif processing_mode == 'strict_5':
            # 严格要求所有5个模态
            self.modality_mapping = {
                'ADC': '0000',
                'DWI': '0001', 
                'T2 fs': '0002',
                'T2 not fs': '0003',
                'gaoqing-T2': '0004'
            }
            self.zero_fill_missing = False
            self.min_modalities = 5
            print("🔧 严格模式：要求所有5个模态都存在（5通道）")
            
        else:
            raise ValueError(f"未知的处理模式: {processing_mode}")
        
        # 设置相似性填充标志
        if not hasattr(self, 'similarity_fill'):
            self.similarity_fill = False
        
        # 固定参数
        self.enable_resampling = True  # 始终启用重采样
        
        # 类别标签
        self.label_mapping = {
            'BPH': 1,  # 良性前列腺增生
            'PCA': 2   # 前列腺癌
        }
        
        self._setup_directories()
    
    def _resample_image(self, image_data: np.ndarray, target_shape: tuple, 
                       case_id: str, modality: str) -> np.ndarray:
        """
        重采样图像到目标形状
        
        Args:
            image_data: 原始图像数据
            target_shape: 目标形状
            case_id: 病例ID
            modality: 模态名称
            
        Returns:
            重采样后的图像数据
        """
        if image_data.shape == target_shape:
            return image_data
        
        # 计算缩放因子
        zoom_factors = [target_shape[i] / image_data.shape[i] for i in range(len(target_shape))]
        
        print(f"   🔄 重采样 {case_id} 的 {modality}: {image_data.shape} -> {target_shape}")
        
        # 使用三次样条插值进行重采样
        try:
            resampled_data = zoom(image_data, zoom_factors, order=1, mode='nearest')
            
            # 确保输出形状正确（由于浮点数精度问题可能有微小差异）
            if resampled_data.shape != target_shape:
                # 如果形状仍不匹配，使用裁剪或填充
                resampled_data = self._adjust_shape(resampled_data, target_shape)
            
            return resampled_data.astype(np.float32)
            
        except Exception as e:
            print(f"   ❌ 重采样失败 {case_id} 的 {modality}: {e}")
            return None
    
    def _adjust_shape(self, data: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        通过裁剪或填充调整数据形状
        """
        current_shape = data.shape
        adjusted_data = data.copy()
        
        for i in range(len(target_shape)):
            if current_shape[i] > target_shape[i]:
                # 裁剪
                slice_obj = [slice(None)] * len(current_shape)
                slice_obj[i] = slice(0, target_shape[i])
                adjusted_data = adjusted_data[tuple(slice_obj)]
            elif current_shape[i] < target_shape[i]:
                # 填充
                pad_width = [(0, 0)] * len(current_shape)
                pad_width[i] = (0, target_shape[i] - current_shape[i])
                adjusted_data = np.pad(adjusted_data, pad_width, mode='constant', constant_values=0)
        
        return adjusted_data
    
    def _similarity_fill_gaoqing_t2(self, t2_fs_data: np.ndarray, 
                                   t2_not_fs_data: np.ndarray, 
                                   case_id: str) -> np.ndarray:
        """
        基于T2模态相似性填充gaoqing-T2
        
        原理: gaoqing-T2 ≈ enhanced(α × T2_fs + β × T2_not_fs)
        """
        print(f"   🎨 相似性填充: {case_id} 的 gaoqing-T2 (基于T2 fs + T2 not fs)")
        
        # 参数设置
        t2_fs_weight = 0.6      # T2 fs权重（通常对比度更好）
        t2_not_fs_weight = 0.4  # T2 not fs权重
        enhancement_factor = 0.3 # 边缘增强因子
        sigma = 0.5             # 高斯滤波参数
        
        # 步骤1: 基础加权融合
        base_estimate = t2_fs_weight * t2_fs_data + t2_not_fs_weight * t2_not_fs_data
        
        # 步骤2: 边缘增强（模拟高清效果）
        smoothed = ndimage.gaussian_filter(base_estimate, sigma=sigma)
        enhanced = base_estimate + enhancement_factor * (base_estimate - smoothed)
        
        # 步骤3: 对比度优化
        p1, p99 = np.percentile(enhanced, (1, 99))
        enhanced = np.clip(enhanced, p1, p99)
        
        # 步骤4: 强度归一化到合理范围
        original_min = min(t2_fs_data.min(), t2_not_fs_data.min())
        original_max = max(t2_fs_data.max(), t2_not_fs_data.max())
        
        enhanced_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
        normalized = enhanced_norm * (original_max - original_min) + original_min
        
        # 步骤5: 细微纹理增强
        texture_kernel = np.array([[-0.05, -0.1, -0.05],
                                  [-0.1,   1.4, -0.1], 
                                  [-0.05, -0.1, -0.05]])
        
        textured_data = np.zeros_like(normalized)
        for i in range(normalized.shape[2]):  # 逐层处理
            textured_data[:, :, i] = ndimage.convolve(
                normalized[:, :, i], texture_kernel, mode='reflect'
            )
        
        # 步骤6: 混合原始和纹理增强版本
        final_result = 0.8 * normalized + 0.2 * textured_data
        
        # 确保数据类型和范围
        final_result = np.clip(final_result, original_min, original_max)
        
        print(f"     ✅ 相似性填充完成: 强度范围 [{final_result.min():.2f}, {final_result.max():.2f}]")
        
        return final_result.astype(np.float32)
    
    def _setup_directories(self):
        """创建nnU-Net目录结构"""
        self.dataset_dir = self.output_dir / self.dataset_name
        self.images_tr_dir = self.dataset_dir / "imagesTr"
        self.labels_tr_dir = self.dataset_dir / "labelsTr"
        self.images_ts_dir = self.dataset_dir / "imagesTs"
        
        # 创建目录
        for dir_path in [self.images_tr_dir, self.labels_tr_dir, self.images_ts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✓ 创建数据集目录: {self.dataset_dir}")
    

    
    def _check_modalities_for_case(self, case_id: str, category: str) -> Dict[str, Path]:
        """检查某个病例的所有模态文件"""
        available_modalities = {}
        category_dir = self.source_dir / category
        
        for modality, _ in self.modality_mapping.items():
            modality_dir = category_dir / modality
            nii_file = modality_dir / f"{case_id}.nii"
            
            if nii_file.exists():
                available_modalities[modality] = nii_file
        
        return available_modalities
    
    def _validate_case_completeness(self, case_id: str, category: str) -> Tuple[bool, Dict[str, Path]]:
        """验证病例数据完整性"""
        modalities = self._check_modalities_for_case(case_id, category)
        
        # 检查标签文件
        roi_dir = self.source_dir / "ROI(BPH+PCA)" / category
        label_file = roi_dir / f"{case_id}.nii"
        if not label_file.exists():
            print(f"⚠️  跳过 {case_id}: 没有标签文件")
            return False, {}
        
        # 检查模态完整性
        if self.processing_mode == 'strict_5':
            # 严格模式：要求所有5个模态都存在
            missing_modalities = set(self.modality_mapping.keys()) - set(modalities.keys())
            if missing_modalities:
                print(f"⚠️  跳过 {case_id}: 缺少模态 {missing_modalities}")
                return False, {}
        else:
            # 其他模式：至少需要4个核心模态
            core_modalities = {'ADC', 'DWI', 'T2 fs', 'T2 not fs'}
            available_core = set(modalities.keys()) & core_modalities
            if len(available_core) < 4:
                missing_core = core_modalities - available_core
                print(f"⚠️  跳过 {case_id}: 缺少核心模态 {missing_core}")
                return False, {}
        
        return True, modalities
    
    def _combine_modalities(self, modalities: Dict[str, Path], case_id: str) -> str:
        """合并多模态数据为nnU-Net格式"""
        # 确定要处理的模态列表
        if self.zero_fill_missing:
            # 0填充模式：处理所有定义的模态
            target_modalities = list(self.modality_mapping.keys())
            num_channels = len(target_modalities)
        else:
            # 常规模式：只处理可用的模态
            target_modalities = [m for m in self.modality_mapping.keys() if m in modalities]
            num_channels = len(target_modalities)
        
        if not target_modalities:
            raise ValueError(f"没有可用的模态数据")
        
        # 读取第一个可用模态作为参考
        reference_modality = None
        for modality in target_modalities:
            if modality in modalities:
                reference_modality = modality
                break
        
        if not reference_modality:
            raise ValueError(f"没有找到参考模态")
        
        ref_img = nib.load(modalities[reference_modality])
        ref_shape = ref_img.shape
        ref_affine = ref_img.affine
        
        # 创建多通道数据
        combined_data = np.zeros((*ref_shape, num_channels), dtype=np.float32)
        valid_channels = 0
        missing_modalities = []
        
        for i, modality in enumerate(target_modalities):
            if modality in modalities:
                # 处理存在的模态
                try:
                    file_path = modalities[modality]
                    img = nib.load(file_path)
                    data = img.get_fdata().astype(np.float32)
                    
                    # 检查形状是否一致
                    if data.shape != ref_shape:
                        if self.enable_resampling:
                            # 尝试重采样
                            resampled_data = self._resample_image(data, ref_shape, case_id, modality)
                            if resampled_data is not None:
                                data = resampled_data
                            else:
                                if self.zero_fill_missing:
                                    print(f"   🔄 重采样失败，使用0填充: {case_id} 的 {modality}")
                                    data = np.zeros(ref_shape, dtype=np.float32)
                                    missing_modalities.append(modality)
                                else:
                                    print(f"⚠️  跳过: {case_id} 的 {modality} 模态重采样失败")
                                    continue
                        else:
                            if self.zero_fill_missing:
                                print(f"   🔄 形状不一致，使用0填充: {case_id} 的 {modality}")
                                data = np.zeros(ref_shape, dtype=np.float32)
                                missing_modalities.append(modality)
                            else:
                                print(f"⚠️  跳过: {case_id} 的 {modality} 模态形状不一致: {data.shape} vs {ref_shape}")
                                continue
                    
                    combined_data[..., i] = data
                    valid_channels += 1
                    
                except Exception as e:
                    if self.zero_fill_missing:
                        print(f"   🔄 读取失败，使用0填充: {case_id} 的 {modality}")
                        combined_data[..., i] = np.zeros(ref_shape, dtype=np.float32)
                        missing_modalities.append(modality)
                        valid_channels += 1
                    else:
                        print(f"⚠️  警告: 读取 {case_id} 的 {modality} 模态失败: {e}")
                        continue
            else:
                # 处理缺失的模态
                if self.zero_fill_missing:
                    if self.similarity_fill and modality == 'gaoqing-T2':
                        # 对gaoqing-T2使用相似性填充
                        if 'T2 fs' in modalities and 'T2 not fs' in modalities:
                            try:
                                # 加载T2模态数据
                                t2_fs_img = nib.load(modalities['T2 fs'])
                                t2_not_fs_img = nib.load(modalities['T2 not fs'])
                                
                                t2_fs_data = t2_fs_img.get_fdata().astype(np.float32)
                                t2_not_fs_data = t2_not_fs_img.get_fdata().astype(np.float32)
                                
                                # 重采样到参考形状（如果需要）
                                if t2_fs_data.shape != ref_shape:
                                    t2_fs_data = self._resample_image(t2_fs_data, ref_shape, case_id, 'T2 fs')
                                if t2_not_fs_data.shape != ref_shape:
                                    t2_not_fs_data = self._resample_image(t2_not_fs_data, ref_shape, case_id, 'T2 not fs')
                                
                                # 执行相似性填充
                                filled_data = self._similarity_fill_gaoqing_t2(
                                    t2_fs_data, t2_not_fs_data, case_id
                                )
                                combined_data[..., i] = filled_data
                                
                            except Exception as e:
                                print(f"   ❌ 相似性填充失败，使用0填充: {case_id} - {e}")
                                combined_data[..., i] = np.zeros(ref_shape, dtype=np.float32)
                                missing_modalities.append(modality)
                        else:
                            print(f"   ⚠️  缺少T2模态，使用0填充: {case_id} 的 {modality}")
                            combined_data[..., i] = np.zeros(ref_shape, dtype=np.float32)
                            missing_modalities.append(modality)
                    else:
                        # 其他模态使用0填充
                        print(f"   🔄 模态缺失，使用0填充: {case_id} 的 {modality}")
                        combined_data[..., i] = np.zeros(ref_shape, dtype=np.float32)
                        missing_modalities.append(modality)
                    
                    valid_channels += 1
                # 如果不是0填充模式，直接跳过缺失的模态
        
        if valid_channels == 0:
            raise ValueError(f"没有有效的模态数据")
        
        # 0填充模式下不需要裁剪，因为所有通道都已填充
        if not self.zero_fill_missing and valid_channels < num_channels:
            combined_data = combined_data[..., :valid_channels]
        
        # 输出0填充统计信息
        if missing_modalities:
            print(f"   📊 {case_id} 0填充模态: {missing_modalities}")
        
        # 保存合并后的图像
        output_filename = f"{case_id}_{self.dataset_id:03d}.nii.gz"
        output_path = self.images_tr_dir / output_filename
        
        # 创建合适的header
        header = nib.Nifti1Header()
        header.set_data_dtype(np.float32)
        
        combined_img = nib.Nifti1Image(combined_data, ref_affine, header)
        nib.save(combined_img, output_path)
        
        return output_filename
    
    def _process_label(self, case_id: str, category: str) -> str:
        """处理标签文件"""
        roi_dir = self.source_dir / "ROI(BPH+PCA)" / category
        label_file = roi_dir / f"{case_id}.nii"
        
        if not label_file.exists():
            print(f"⚠️  警告: 找不到 {case_id} 的标签文件")
            return None
        
        # 读取标签
        label_img = nib.load(label_file)
        label_data = label_img.get_fdata().astype(np.uint8)
        
        # 将标签值映射为类别
        label_value = self.label_mapping[category]
        label_data = np.where(label_data > 0, label_value, 0)
        
        # 保存标签
        output_filename = f"{case_id}_{self.dataset_id:03d}.nii.gz"
        output_path = self.labels_tr_dir / output_filename
        
        # 创建合适的header
        header = nib.Nifti1Header()
        header.set_data_dtype(np.uint8)
        
        label_img_new = nib.Nifti1Image(label_data, label_img.affine, header)
        nib.save(label_img_new, output_path)
        
        return output_filename
    
    def _create_dataset_json(self, processed_cases: List[Dict]):
        """创建dataset.json文件"""
        # 统计模态信息
        all_modalities = set()
        for case in processed_cases:
            all_modalities.update(case['modalities'])
        
        modality_dict = {}
        for i, modality in enumerate(sorted(all_modalities)):
            modality_dict[str(i)] = modality
        
        dataset_json = {
            "channel_names": modality_dict,
            "labels": {
                "background": 0,
                "BPH": 1,
                "PCA": 2
            },
            "numTraining": len(processed_cases),
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",
            "dataset_name": self.dataset_name,
            "description": "BPH-PCA多模态前列腺MRI分割数据集",
            "reference": "BPH-PCA Dataset for Prostate Segmentation",
            "licence": "研究使用",
            "tensorImageSize": "3D"
        }
        
        # 保存dataset.json
        json_path = self.dataset_dir / "dataset.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_json, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 创建dataset.json: {json_path}")
        return dataset_json
    
    def convert(self):
        """执行数据转换"""
        print(f"🚀 开始转换BPH-PCA数据集到nnU-Net v2格式...")
        print(f"📁 源目录: {self.source_dir}")
        print(f"📁 输出目录: {self.dataset_dir}")
        
        processed_cases = []
        
        # 处理BPH和PCA数据
        for category in ['BPH', 'PCA']:
            print(f"\n📋 处理 {category} 数据...")
            
            category_dir = self.source_dir / category
            if not category_dir.exists():
                print(f"⚠️  跳过不存在的目录: {category_dir}")
                continue
            
            # 获取该类别的所有病例
            case_ids = set()
            for modality_dir in category_dir.iterdir():
                if modality_dir.is_dir():
                    for nii_file in modality_dir.glob("*.nii"):
                        case_ids.add(nii_file.stem)
            
            case_ids = sorted(list(case_ids))
            print(f"   找到 {len(case_ids)} 个 {category} 病例")
            
            # 处理每个病例
            for case_id in tqdm(case_ids, desc=f"处理{category}"):
                # 验证病例完整性
                is_valid, modalities = self._validate_case_completeness(case_id, category)
                
                if not is_valid:
                    continue
                
                # 合并多模态数据
                try:
                    image_filename = self._combine_modalities(modalities, case_id)
                    label_filename = self._process_label(case_id, category)
                    
                    if image_filename and label_filename:
                        processed_cases.append({
                            'case_id': case_id,
                            'category': category,
                            'modalities': list(modalities.keys()),
                            'image_file': image_filename,
                            'label_file': label_filename
                        })
                
                except Exception as e:
                    print(f"❌ 处理 {case_id} 时出错: {e}")
                    continue
        
        print(f"\n✅ 成功处理 {len(processed_cases)} 个病例")
        
        # 创建dataset.json
        dataset_info = self._create_dataset_json(processed_cases)
        
        # 输出统计信息
        self._print_statistics(processed_cases, dataset_info)
        
        return processed_cases
    
    def _print_statistics(self, processed_cases: List[Dict], dataset_info: Dict):
        """打印统计信息"""
        print(f"\n📊 数据集统计信息:")
        print(f"   总病例数: {len(processed_cases)}")
        
        # 按类别统计
        category_counts = {}
        for case in processed_cases:
            category = case['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category, count in category_counts.items():
            print(f"   {category}: {count} 例")
        
        # 模态统计
        modality_counts = {}
        for case in processed_cases:
            for modality in case['modalities']:
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        print(f"\n   可用模态:")
        for modality, count in modality_counts.items():
            print(f"   {modality}: {count} 例")
        
        print(f"\n📁 输出文件:")
        print(f"   图像目录: {self.images_tr_dir}")
        print(f"   标签目录: {self.labels_tr_dir}")
        print(f"   配置文件: {self.dataset_dir}/dataset.json")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将BPH-PCA数据转换为nnU-Net v2格式')
    parser.add_argument('--source_dir', type=str, default='data/BPH-PCA',
                       help='BPH-PCA数据源目录')
    parser.add_argument('--output_dir', type=str, default='nnUNet_raw',
                       help='nnU-Net输出目录')
    parser.add_argument('--dataset_id', type=int, default=1,
                       help='数据集ID')
    parser.add_argument('--mode', type=str, default='similarity_fill',
                       choices=['core_4', 'zero_fill', 'similarity_fill', 'strict_5'],
                       help='处理模式: core_4(4通道), zero_fill(5通道0填充), similarity_fill(5通道相似性填充), strict_5(严格5通道)')
    
    args = parser.parse_args()
    
    mode_descriptions = {
        'core_4': '核心4模态模式（ADC, DWI, T2 fs, T2 not fs）',
        'zero_fill': '0填充5模态模式（缺失gaoqing-T2用0填充）',
        'similarity_fill': '相似性填充5模态模式（缺失gaoqing-T2用相似性填充）',
        'strict_5': '严格5模态模式（要求所有模态都存在）'
    }
    
    print(f"🔧 数据处理设置:")
    print(f"   处理模式: {mode_descriptions[args.mode]}")
    print(f"   数据集ID: {args.dataset_id}")
    
    # 创建转换器并执行转换
    converter = BPHPCAToNnUNetConverter(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        processing_mode=args.mode
    )
    
    processed_cases = converter.convert()
    
    print(f"\n🎉 数据转换完成！")
    print(f"💡 接下来可以运行以下命令进行nnU-Net训练:")
    print(f"   nnUNetv2_plan_and_preprocess -d {args.dataset_id}")
    print(f"   nnUNetv2_train {args.dataset_id} 3d_fullres 0")


if __name__ == "__main__":
    main()
