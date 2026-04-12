#!/usr/bin/env python
"""
改善iSCALE注释覆盖率的实用脚本
提供多种方案减少空白
"""

import numpy as np
import pickle
import sys
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def adjust_confidence_threshold(annot_dir, new_threshold=0.03, output_suffix='_adjusted'):
    """
    方案2: 调整置信度阈值，重新生成注释
    
    参数:
    - annot_dir: annotations目录
    - new_threshold: 新的置信度阈值（推荐0.03）
    - output_suffix: 输出文件后缀
    """
    
    print(f"调整置信度阈值为: {new_threshold}")
    
    # 读取scores（从threshold001目录的pickle或重新计算）
    # 这里假设我们从已有的labels重新处理
    threshold_dirs = [
        os.path.join(annot_dir, 'threshold001'),
        os.path.join(annot_dir, 'threshold005'),
        os.path.join(annot_dir, 'threshold010')
    ]
    
    # 读取confidence
    conf_found = False
    for thr_dir in threshold_dirs:
        try:
            # 尝试找到原始的confidence数据
            # 注：这里简化处理，实际可能需要从score重新计算
            pass
        except:
            pass
    
    print("提示: 完整实现需要访问原始scores数据")
    print("建议: 修改 pixannot_percentile.py 中的阈值列表")
    print(f"  将第141行改为: for threshold in [{new_threshold}, 0.05, 0.1]:")


def create_comprehensive_markers(output_file='markers_comprehensive.csv'):
    """
    方案1: 创建更全面的marker文件模板
    """
    
    print("创建全面的marker文件模板...")
    
    # 基于brca_markers.tsv，添加缺失的组织类型
    additional_markers = {
        # 基质细胞（已有Fibroblast，补充）
        'Stromal': [
            'COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'SPARC'
        ],
        # 血管（已有Endothelial，补充）
        'Vascular': [
            'VWF', 'PECAM1', 'CD34', 'ENG'
        ],
        # 脂肪细胞
        'Adipocyte': [
            'ADIPOQ', 'LEP', 'PLIN1', 'FABP4', 'LPL'
        ],
        # 平滑肌（已有Perivascular，补充）
        'Smooth_Muscle': [
            'ACTA2', 'MYH11', 'TAGLN', 'CNN1', 'MYLK'
        ],
        # 神经
        'Nerve': [
            'S100B', 'SOX10', 'PLP1', 'MPZ'
        ]
    }
    
    markers_list = []
    for cell_type, genes in additional_markers.items():
        for gene in genes:
            markers_list.append({'gene': gene, 'label': cell_type})
    
    df = pd.DataFrame(markers_list)
    df.to_csv(output_file, index=False)
    
    print(f"✓ 创建补充marker文件: {output_file}")
    print(f"  包含 {len(additional_markers)} 种补充组织类型")
    print(f"  总共 {len(df)} 个marker基因")
    print("\n使用方法:")
    print(f"  1. 将 {output_file} 与你的 brca_markers.csv 合并")
    print(f"  2. 或单独使用此文件运行注释")


def visualize_with_other_category(labels_pickle, output_path, label_names_file):
    """
    方案4: 给低置信度区域一个"Other"类别（仅用于可视化）
    """
    
    print("创建包含'Other'类的可视化...")
    
    with open(labels_pickle, 'rb') as f:
        labels = pickle.load(f)
    
    with open(label_names_file, 'r') as f:
        label_names = [line.strip() for line in f]
    
    # 将Unclassified改为Other
    labels_vis = labels.copy()
    # 这里labels=0是Unclassified，改为特定颜色
    
    # 创建颜色映射
    n_labels = len(label_names)
    colors = plt.cm.Set3(np.linspace(0, 1, n_labels))
    colors[0] = [0.9, 0.9, 0.9, 1.0]  # Unclassified用浅灰色
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(labels_vis, cmap=ListedColormap(colors), interpolation='nearest')
    ax.set_title('Cell Type Annotation (with "Other" category)', fontsize=14)
    ax.axis('off')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=label_names[i]) 
                      for i in range(min(len(label_names), 10))]  # 只显示前10个
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存可视化: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='改善iSCALE注释覆盖率')
    parser.add_argument('--action', choices=['diagnose', 'markers', 'visualize'], 
                       required=True,
                       help='操作类型')
    parser.add_argument('--output_base', type=str,
                       help='iSCALE输出目录')
    parser.add_argument('--threshold', type=float, default=0.03,
                       help='新的置信度阈值（默认0.03）')
    
    args = parser.parse_args()
    
    if args.action == 'markers':
        create_comprehensive_markers()
    
    elif args.action == 'visualize' and args.output_base:
        annot_dir = os.path.join(args.output_base, 'MotherImage/iSCALE_output/annotations/')
        labels_pickle = os.path.join(annot_dir, 'threshold005/labels.pickle')
        label_names = os.path.join(annot_dir, 'threshold005/label-names.txt')
        output = os.path.join(annot_dir, 'labels_with_other.png')
        
        if os.path.exists(labels_pickle):
            visualize_with_other_category(labels_pickle, output, label_names)
        else:
            print(f"✗ 找不到: {labels_pickle}")
    
    else:
        print("请指定有效的操作和参数")


if __name__ == '__main__':
    # 如果直接运行（无参数），显示帮助
    if len(sys.argv) == 1:
        print("="*70)
        print("  改善iSCALE注释覆盖率")
        print("="*70)
        print("\n使用方法:")
        print("\n1. 创建补充marker文件（推荐）:")
        print("   python improve_annotation_coverage.py --action markers")
        print("\n2. 创建包含Other类的可视化:")
        print("   python improve_annotation_coverage.py --action visualize --output_base /path/to/output")
        print("\n3. 调整置信度阈值（需要修改pixannot_percentile.py）:")
        print("   - 打开 iSCALE/pixannot_percentile.py")
        print("   - 找到第141行: for threshold in [0.01, 0.05, 0.1]:")
        print("   - 改为: for threshold in [0.01, 0.03, 0.05, 0.1]:")
        print("   - 重新运行注释")
    else:
        main()

