#!/usr/bin/env python
"""
检查iSCALE注释输出的实际文件结构
"""

import os
import sys

def check_annotation_output(annotation_dir):
    """检查注释输出目录的文件结构"""
    
    if not os.path.exists(annotation_dir):
        print(f"✗ 目录不存在: {annotation_dir}")
        return
    
    print("="*70)
    print("  iSCALE 注释输出文件检查")
    print("="*70)
    print(f"\n检查目录: {annotation_dir}\n")
    
    # 检查根目录文件
    print("📁 根目录文件:")
    root_files = []
    for item in os.listdir(annotation_dir):
        item_path = os.path.join(annotation_dir, item)
        if os.path.isfile(item_path):
            root_files.append(item)
            print(f"  ✓ {item}")
    
    if not root_files:
        print("  (无文件)")
    
    # 检查子目录
    print("\n📁 子目录:")
    subdirs = []
    for item in os.listdir(annotation_dir):
        item_path = os.path.join(annotation_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
            print(f"  ✓ {item}/")
    
    if not subdirs:
        print("  (无子目录)")
    
    # 检查每个阈值目录
    print("\n📊 阈值目录详细内容:")
    for threshold_dir in ['threshold001', 'threshold005', 'threshold010']:
        threshold_path = os.path.join(annotation_dir, threshold_dir)
        if os.path.exists(threshold_path):
            print(f"\n  {threshold_dir}/:")
            for item in os.listdir(threshold_path):
                item_path = os.path.join(threshold_path, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path) / 1024  # KB
                    print(f"    ✓ {item} ({size:.1f} KB)")
                elif os.path.isdir(item_path):
                    n_files = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                    print(f"    ✓ {item}/ ({n_files} 个文件)")
    
    # 检查scores目录
    scores_path = os.path.join(annotation_dir, 'scores')
    if os.path.exists(scores_path):
        print(f"\n  scores/ (各细胞类型的得分图):")
        score_files = [f for f in os.listdir(scores_path) if f.endswith('.png')]
        for score_file in sorted(score_files)[:10]:  # 只显示前10个
            print(f"    ✓ {score_file}")
        if len(score_files) > 10:
            print(f"    ... 还有 {len(score_files) - 10} 个文件")
    
    # 检查masks目录（在阈值目录下）
    print("\n📊 掩膜文件位置:")
    for threshold_dir in ['threshold001', 'threshold005', 'threshold010']:
        masks_path = os.path.join(annotation_dir, threshold_dir, 'masks')
        if os.path.exists(masks_path):
            mask_files = [f for f in os.listdir(masks_path) if f.endswith('.png')]
            if mask_files:
                print(f"  {threshold_dir}/masks/: {len(mask_files)} 个掩膜文件")
    
    # 总结
    print("\n" + "="*70)
    print("  📍 labels.png 文件位置")
    print("="*70)
    print("\n根据代码逻辑，labels.png 在以下位置:")
    for threshold_dir in ['threshold001', 'threshold005', 'threshold010']:
        labels_path = os.path.join(annotation_dir, threshold_dir, 'labels.png')
        if os.path.exists(labels_path):
            print(f"  ✓ {labels_path}")
        else:
            print(f"  ✗ {labels_path} (未找到)")
    
    print("\n💡 提示:")
    print("  - threshold001/ 对应置信度阈值 0.01 (最宽松)")
    print("  - threshold005/ 对应置信度阈值 0.05 (中等)")
    print("  - threshold010/ 对应置信度阈值 0.1 (最严格)")
    print("  - 建议查看 threshold005/labels.png (中等阈值，平衡效果)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python check_annotation_output.py <annotation_directory>")
        print("\n示例:")
        print("  python check_annotation_output.py /path/to/annotations/")
        sys.exit(1)
    
    annotation_dir = sys.argv[1]
    if not annotation_dir.endswith('/'):
        annotation_dir += '/'
    
    check_annotation_output(annotation_dir)




