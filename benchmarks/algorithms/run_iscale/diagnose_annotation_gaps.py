#!/usr/bin/env python
"""
诊断iSCALE细胞类型注释中的空白区域
找出为什么中间有空白，以及如何改善
"""

import numpy as np
import pickle
import sys
import os
from PIL import Image
import pandas as pd

def diagnose_annotation_gaps(output_base):
    """
    诊断注释空白的原因
    
    参数:
    - output_base: iSCALE输出目录
    """
    
    prefix = os.path.join(output_base, 'MotherImage/')
    annot_dir = os.path.join(prefix, 'iSCALE_output/annotations/')
    
    print("="*70)
    print("  iSCALE 注释空白诊断")
    print("="*70)
    print(f"\n检查目录: {prefix}\n")
    
    # ========== 1. 检查组织掩膜 ==========
    print("【1】检查组织掩膜 (Tissue Mask)")
    print("-" * 70)
    
    mask_path = os.path.join(prefix, 'filterRGB/mask-small-refined.png')
    if os.path.exists(mask_path):
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        total_pixels = mask.size
        tissue_pixels = (mask > 0).sum()
        tissue_ratio = tissue_pixels / total_pixels * 100
        
        print(f"✓ 掩膜文件: {mask_path}")
        print(f"  总像素: {total_pixels:,}")
        print(f"  组织像素: {tissue_pixels:,} ({tissue_ratio:.1f}%)")
        print(f"  被过滤: {total_pixels - tissue_pixels:,} ({100-tissue_ratio:.1f}%)")
        
        if tissue_ratio < 50:
            print(f"  ⚠️  组织区域较少 - 可能掩膜过于严格")
    else:
        print(f"✗ 未找到掩膜文件: {mask_path}")
    
    # ========== 2. 检查Confidence分布 ==========
    print("\n【2】检查Confidence（置信度）分布")
    print("-" * 70)
    
    conf_path = os.path.join(annot_dir, 'confidence.png')
    if os.path.exists(conf_path):
        # 从pickle读取原始confidence数据
        # 注：confidence.png是可视化后的，我们需要检查原始score
        print(f"✓ Confidence图存在: {conf_path}")
        print("  提示: 白色区域 = NaN（无置信度）")
    else:
        print(f"✗ 未找到: {conf_path}")
    
    # ========== 3. 检查Marker覆盖度 ==========
    print("\n【3】检查Marker基因覆盖度")
    print("-" * 70)
    
    # 读取marker文件
    marker_file = os.path.join(prefix, 'markers.csv')
    if os.path.exists(marker_file):
        markers_df = pd.read_csv(marker_file)
        
        cell_types = markers_df['label'].unique()
        print(f"✓ Marker文件: {marker_file}")
        print(f"  细胞类型数量: {len(cell_types)}")
        print(f"\n  细胞类型统计:")
        
        type_counts = markers_df['label'].value_counts()
        for ct, count in type_counts.items():
            print(f"    • {ct}: {count} 个marker")
        
        # 检查是否缺少重要组织类型
        print(f"\n  🔍 缺失的常见组织类型检查:")
        common_types = {
            'Stromal': ['Fibroblast'],
            'Vascular': ['Endothelial'],
            'Smooth Muscle': ['Perivascular-Like'],
            'Adipose': ['Adipocyte'],
        }
        
        missing_categories = []
        for category, expected in common_types.items():
            found = any(any(exp.lower() in ct.lower() for exp in expected) 
                       for ct in cell_types)
            if found:
                print(f"    ✓ {category}: 已覆盖")
            else:
                print(f"    ✗ {category}: 缺失（可能导致空白）")
                missing_categories.append(category)
        
        if missing_categories:
            print(f"\n  ⚠️  缺少的组织类型: {', '.join(missing_categories)}")
            print(f"      这些区域可能显示为空白")
    else:
        print(f"✗ 未找到marker文件: {marker_file}")
    
    # ========== 4. 检查不同阈值的注释覆盖率 ==========
    print("\n【4】检查不同阈值的注释覆盖率")
    print("-" * 70)
    
    for threshold_name, threshold_val in [('threshold001', 0.01), 
                                          ('threshold005', 0.05), 
                                          ('threshold010', 0.10)]:
        threshold_dir = os.path.join(annot_dir, threshold_name)
        labels_pickle = os.path.join(threshold_dir, 'labels.pickle')
        
        if os.path.exists(labels_pickle):
            with open(labels_pickle, 'rb') as f:
                labels = pickle.load(f)
            
            total_pixels = labels.size
            unclassified = (labels == 0).sum()
            classified = (labels > 0).sum()
            classified_ratio = classified / total_pixels * 100
            
            print(f"\n  {threshold_name} (阈值={threshold_val}):")
            print(f"    总像素: {total_pixels:,}")
            print(f"    已注释: {classified:,} ({classified_ratio:.1f}%)")
            print(f"    未分类: {unclassified:,} ({100-classified_ratio:.1f}%)")
    
    # ========== 5. 给出建议 ==========
    print("\n" + "="*70)
    print("  💡 诊断结果与建议")
    print("="*70)
    
    print("\n空白区域的主要原因：")
    print("  1. 组织掩膜过滤 - 背景/空腔/切片缺损被排除")
    print("  2. Marker覆盖不全 - 某些组织类型没有对应marker")
    print("  3. 置信度阈值 - 低置信度区域被标记为\"Unclassified\"")
    
    print("\n改善空白的方案（按推荐顺序）：")
    print("\n  【方案1】补全Marker Panel（最推荐）")
    print("    - 添加缺失的组织类型marker（基质、血管、平滑肌等）")
    print("    - 使用更全面的marker数据库（如PanglaoDB、CellMarker）")
    print("    - 保持科学严谨性")
    
    print("\n  【方案2】调整置信度阈值")
    print("    - 当前: threshold005 (0.05)")
    print("    - 推荐: threshold001 (0.01) - 减少空白")
    print("    - 或自定义阈值 0.03")
    
    print("\n  【方案3】调整组织掩膜（谨慎）")
    print("    - 检查 mask-small-refined.png 是否过于严格")
    print("    - 可以在 refine_mask.py 中调整参数")
    
    print("\n  【方案4】可视化处理（仅用于展示）")
    print("    - 给低置信度区域一个\"Other\"类别")
    print("    - 不建议用于科学分析")
    
    print("\n汇报中的标准解释：")
    print('  "注释图中的空白区域对应以下情况：')
    print('   (1) 被组织掩膜过滤的背景或切片缺损区域')
    print('   (2) 预测置信度低于阈值的不确定区域')
    print('   (3) 缺少相应marker的组织类型（如基质、血管等）')
    print('   这是为避免过度注释而采取的保守策略。"')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python diagnose_annotation_gaps.py <output_base>")
        print("\n示例:")
        print("  python diagnose_annotation_gaps.py /data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/xenium_benchmark_output")
        sys.exit(1)
    
    output_base = sys.argv[1]
    diagnose_annotation_gaps(output_base)

