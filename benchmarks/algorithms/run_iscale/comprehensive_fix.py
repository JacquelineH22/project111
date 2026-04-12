#!/usr/bin/env python
"""
iSCALE注释空白的综合解决方案
同时处理: tissue mask + cutoff + threshold
"""

import numpy as np
import os
import sys
from PIL import Image
from scipy import ndimage
from skimage import morphology
import shutil

def fix_mask_aggressive(output_base, dilation=20, min_size=20):
    """
    方案1: 激进调整tissue mask
    """
    prefix = os.path.join(output_base, 'MotherImage/')
    mask_path = os.path.join(prefix, 'mask-small.png')
    output_path = os.path.join(prefix, 'filterRGB/mask-small-refined.png')
    backup_path = os.path.join(prefix, 'filterRGB/mask-small-refined-BACKUP.png')
    
    print("="*70)
    print("  方案1: 激进调整Tissue Mask")
    print("="*70)
    
    # 备份原掩膜
    if os.path.exists(output_path) and not os.path.exists(backup_path):
        shutil.copy2(output_path, backup_path)
        print(f"✓ 备份原掩膜: {backup_path}")
    
    # 读取mask-small.png
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    mask_binary = mask > 0
    orig_ratio = mask_binary.sum() / mask_binary.size * 100
    
    print(f"\n原始掩膜保留: {orig_ratio:.1f}%")
    print(f"调整参数: dilation={dilation}, min_size={min_size}")
    
    # 激进的形态学操作
    kernel = np.ones((dilation, dilation))
    
    # 1. 先闭运算填充孔洞
    mask_closed = ndimage.binary_closing(mask_binary, structure=kernel, iterations=2)
    
    # 2. 膨胀扩大区域
    mask_dilated = ndimage.binary_dilation(mask_closed, structure=kernel, iterations=2)
    
    # 3. 填充内部孔洞
    mask_filled = ndimage.binary_fill_holes(mask_dilated)
    
    # 4. 移除小碎片（非常宽松）
    mask_cleaned = morphology.remove_small_objects(mask_filled, min_size=min_size)
    
    # 转换并保存
    mask_final = (mask_cleaned * 255).astype(np.uint8)
    Image.fromarray(mask_final).save(output_path)
    
    new_ratio = mask_cleaned.sum() / mask_cleaned.size * 100
    
    print(f"\n✓ 保存新掩膜: {output_path}")
    print(f"  新掩膜保留: {new_ratio:.1f}%")
    print(f"  增加: {new_ratio - orig_ratio:.1f}%")
    
    return output_path


def create_modified_marker_score(iscale_dir):
    """
    方案2: 修改marker_score.py，降低cutoff
    """
    marker_score_path = os.path.join(iscale_dir, 'marker_score.py')
    backup_path = os.path.join(iscale_dir, 'marker_score_BACKUP.py')
    
    print("\n" + "="*70)
    print("  方案2: 调整Expression Cutoff")
    print("="*70)
    
    # 备份原文件
    if not os.path.exists(backup_path):
        shutil.copy2(marker_score_path, backup_path)
        print(f"✓ 备份原文件: {backup_path}")
    
    # 读取原文件
    with open(marker_score_path, 'r') as f:
        content = f.read()
    
    # 检查并修改cutoff
    if 'cutoff = 1e-5' in content or 'cutoff=1e-5' in content:
        # 将cutoff改为0或更小的值
        modified_content = content.replace('cutoff = 1e-5', 'cutoff = 0.0')
        modified_content = modified_content.replace('cutoff=1e-5', 'cutoff=0.0')
        
        with open(marker_score_path, 'w') as f:
            f.write(modified_content)
        
        print(f"✓ 修改cutoff: 1e-5 → 0.0")
        print(f"  位置: {marker_score_path}")
        print(f"  说明: 不再过滤低表达像素")
        
        return True
    else:
        print(f"⚠ 未找到cutoff=1e-5，可能已修改或版本不同")
        return False


def generate_run_script(output_base, iscale_dir):
    """
    生成重新运行注释的脚本
    """
    script_path = os.path.join(iscale_dir, 'rerun_annotation_fixed.sh')
    
    script_content = f"""#!/bin/bash
# 使用修复后的参数重新运行注释

set -e

echo "=============================================="
echo "  重新运行细胞类型注释（修复版）"
echo "=============================================="

cd {iscale_dir}

# 确认marker文件
MARKER_FILE="{output_base}/MotherImage/markers.csv"
if [ ! -f "$MARKER_FILE" ]; then
    echo "✗ 找不到marker文件: $MARKER_FILE"
    echo "请先准备marker文件"
    exit 1
fi

echo "✓ Marker文件: $MARKER_FILE"
echo ""

# 运行注释（使用新的掩膜和cutoff）
python pixannot_percentile.py \\
    {output_base}/MotherImage/ \\
    $MARKER_FILE \\
    {output_base}/MotherImage/iSCALE_output/annotations_fixed/

echo ""
echo "=============================================="
echo "✓ 注释完成！"
echo "=============================================="
echo ""
echo "查看结果:"
echo "  threshold001: {output_base}/MotherImage/iSCALE_output/annotations_fixed/threshold001/labels.png"
echo "  threshold005: {output_base}/MotherImage/iSCALE_output/annotations_fixed/threshold005/labels.png"
echo ""
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    
    print("\n" + "="*70)
    print("  生成运行脚本")
    print("="*70)
    print(f"✓ 脚本位置: {script_path}")
    print(f"\n运行方式:")
    print(f"  bash {script_path}")
    
    return script_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='综合修复iSCALE注释空白')
    parser.add_argument('output_base', help='iSCALE输出目录')
    parser.add_argument('--iscale_dir', default='~/Benchmark/iSCALE/iSCALE',
                       help='iSCALE代码目录')
    parser.add_argument('--dilation', type=int, default=20,
                       help='Mask膨胀大小（默认20，更激进）')
    parser.add_argument('--min_size', type=int, default=20,
                       help='最小连通域（默认20，更宽松）')
    parser.add_argument('--fix_cutoff', action='store_true',
                       help='是否修改cutoff（需要修改源代码）')
    
    args = parser.parse_args()
    
    iscale_dir = os.path.expanduser(args.iscale_dir)
    
    print("\n" + "="*70)
    print("  iSCALE 注释空白综合修复")
    print("="*70)
    print(f"\n输出目录: {args.output_base}")
    print(f"iSCALE目录: {iscale_dir}")
    
    # 方案1: 修复mask
    fix_mask_aggressive(args.output_base, args.dilation, args.min_size)
    
    # 方案2: 修复cutoff（可选）
    if args.fix_cutoff:
        create_modified_marker_score(iscale_dir)
    else:
        print("\n⚠ 跳过cutoff修改（使用 --fix_cutoff 启用）")
    
    # 生成运行脚本
    script_path = generate_run_script(args.output_base, iscale_dir)
    
    print("\n" + "="*70)
    print("  ✅ 修复完成！")
    print("="*70)
    print("\n下一步:")
    print(f"  1. 运行: bash {script_path}")
    print(f"  2. 查看结果中的空白是否减少")
    print(f"  3. 如果还不够，重新运行本脚本并增大 --dilation 参数")
    print("\n提示:")
    print("  - 原始文件已备份（*_BACKUP.*）")
    print("  - 要恢复原始设置，重命名备份文件即可")


if __name__ == '__main__':
    main()







