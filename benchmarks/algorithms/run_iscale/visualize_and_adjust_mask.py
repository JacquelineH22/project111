#!/usr/bin/env python
"""
可视化和调整iSCALE的组织掩膜
帮助判断掩膜是否过于严格，并提供调整方案
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

# 处理大图像 - 增加PIL的大小限制
Image.MAX_IMAGE_PIXELS = None  # 移除限制（或设置更大的值）

def visualize_mask_overlay(output_base):
    """
    将掩膜叠加到H&E图像上，查看过滤是否合理
    """
    
    prefix = os.path.join(output_base, 'MotherImage/')
    
    print("="*70)
    print("  组织掩膜可视化与调整")
    print("="*70)
    
    # 读取H&E图像
    he_path = os.path.join(prefix, 'he.tiff')
    if not os.path.exists(he_path):
        he_path = os.path.join(prefix, 'he.png')
    
    if not os.path.exists(he_path):
        print(f"✗ 找不到H&E图像")
        return
    
    print(f"\n读取H&E图像: {he_path}")
    
    # 设置最大显示尺寸（避免内存问题）
    max_display_size = 2048
    
    # 读取大图像并下采样以加速显示
    with Image.open(he_path) as img:
        width, height = img.size
        print(f"  原始尺寸: {width} × {height} ({width*height:,} pixels)")
        
        # 如果图像太大，下采样
        max_display_size = 2048
        if max(width, height) > max_display_size:
            scale = max_display_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            print(f"  下采样到: {new_width} × {new_height} (用于显示)")
            he_img = np.array(img.resize((new_width, new_height), Image.LANCZOS))
        else:
            he_img = np.array(img)
    
    # 读取掩膜
    mask_path = os.path.join(prefix, 'filterRGB/mask-small-refined.png')
    mask_orig_path = os.path.join(prefix, 'mask-small.png')
    
    if not os.path.exists(mask_path):
        print(f"✗ 找不到refined掩膜: {mask_path}")
        return
    
    print(f"读取refined掩膜: {mask_path}")
    with Image.open(mask_path) as img:
        # 掩膜需要调整到与H&E图像相同的尺寸
        target_size = (he_img.shape[1], he_img.shape[0])  # (width, height)
        print(f"  调整掩膜尺寸到: {target_size}")
        mask_refined = np.array(img.resize(target_size, Image.NEAREST))
        
        if mask_refined.ndim == 3:
            mask_refined = mask_refined[:, :, 0]
    
    # 读取原始掩膜（如果存在）
    mask_orig = None
    if os.path.exists(mask_orig_path):
        print(f"读取原始掩膜: {mask_orig_path}")
        with Image.open(mask_orig_path) as img:
            # 调整到与H&E相同尺寸
            target_size = (he_img.shape[1], he_img.shape[0])  # (width, height)
            mask_orig = np.array(img.resize(target_size, Image.NEAREST))
            
            if mask_orig.ndim == 3:
                mask_orig = mask_orig[:, :, 0]
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. 原始H&E图像
    if he_img.ndim == 2:
        axes[0, 0].imshow(he_img, cmap='gray')
    else:
        axes[0, 0].imshow(he_img)
    axes[0, 0].set_title('H&E Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Refined掩膜
    axes[0, 1].imshow(mask_refined, cmap='gray')
    axes[0, 1].set_title(f'Refined Mask\n保留: {(mask_refined > 0).sum() / mask_refined.size * 100:.1f}%', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. 掩膜叠加到H&E
    if he_img.ndim == 3:
        he_display = he_img.copy()
    else:
        he_display = np.stack([he_img]*3, axis=-1)
    
    # 创建半透明红色覆盖（被过滤的区域）
    overlay = he_display.copy()
    overlay[mask_refined == 0] = [255, 0, 0]  # 被过滤区域显示红色
    
    # 混合
    alpha = 0.3
    blended = (alpha * overlay + (1 - alpha) * he_display).astype(np.uint8)
    
    axes[1, 0].imshow(blended)
    axes[1, 0].set_title('H&E with Mask Overlay\n(红色 = 被过滤区域)', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 4. 原始掩膜对比（如果存在）
    if mask_orig is not None:
        diff = (mask_orig > 0).astype(int) - (mask_refined > 0).astype(int)
        
        # 创建对比图：绿色=保留，红色=被refine过滤，灰色=原本就过滤
        compare = np.zeros((*mask_orig.shape, 3), dtype=np.uint8)
        compare[mask_refined > 0] = [0, 255, 0]  # 保留的区域（绿色）
        compare[diff > 0] = [255, 0, 0]          # refine后被过滤（红色）
        compare[mask_orig == 0] = [128, 128, 128] # 原本就被过滤（灰色）
        
        axes[1, 1].imshow(compare)
        axes[1, 1].set_title('Mask Comparison\n绿=保留, 红=refine过滤, 灰=原本过滤', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
    else:
        # 显示统计信息
        axes[1, 1].text(0.5, 0.5, 
                       f"组织掩膜统计\n\n"
                       f"总像素: {mask_refined.size:,}\n"
                       f"组织像素: {(mask_refined > 0).sum():,}\n"
                       f"组织比例: {(mask_refined > 0).sum() / mask_refined.size * 100:.1f}%\n\n"
                       f"被过滤: {(mask_refined == 0).sum():,}\n"
                       f"过滤比例: {(mask_refined == 0).sum() / mask_refined.size * 100:.1f}%",
                       ha='center', va='center', fontsize=14,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(prefix, 'mask_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 保存可视化: {output_path}")
    
    plt.show()
    
    # 给出判断和建议
    tissue_ratio = (mask_refined > 0).sum() / mask_refined.size
    
    print("\n" + "="*70)
    print("  判断与建议")
    print("="*70)
    
    if tissue_ratio < 0.3:
        print("\n⚠️  组织占比过低 (<30%)，掩膜可能过于严格")
        print("\n建议：")
        print("  1. 查看上面的可视化图，检查红色区域是否有组织")
        print("  2. 如果红色区域确实是组织，需要放宽掩膜")
    elif tissue_ratio < 0.5:
        print("\n⚠️  组织占比偏低 (30-50%)，可能需要调整")
        print("\n建议：")
        print("  1. 查看可视化图，判断是否合理")
        print("  2. 如果有明显的组织被过滤，建议调整")
    else:
        print("\n✓ 组织占比正常 (>50%)")
        print("  空白可能来自其他原因（marker覆盖、置信度阈值等）")


def adjust_mask_threshold(output_base, dilation_size=5, min_size=100):
    """
    调整掩膜参数，生成更宽松的掩膜
    
    参数:
    - dilation_size: 膨胀操作的kernel大小（增大=保留更多）
    - min_size: 最小连通域大小（减小=保留更多）
    """
    
    from scipy import ndimage
    from skimage import morphology
    
    prefix = os.path.join(output_base, 'MotherImage/')
    mask_path = os.path.join(prefix, 'mask-small.png')
    
    if not os.path.exists(mask_path):
        print(f"✗ 找不到原始掩膜: {mask_path}")
        return
    
    print(f"\n调整掩膜参数...")
    print(f"  膨胀大小: {dilation_size}")
    print(f"  最小连通域: {min_size}")
    
    # 读取原始掩膜
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    mask_binary = mask > 0
    
    # 应用形态学操作
    # 1. 闭运算（填充小孔）
    kernel = np.ones((dilation_size, dilation_size))
    mask_closed = ndimage.binary_closing(mask_binary, structure=kernel)
    
    # 2. 膨胀（扩大组织区域）
    mask_dilated = ndimage.binary_dilation(mask_closed, structure=kernel)
    
    # 3. 移除小连通域
    mask_cleaned = morphology.remove_small_objects(mask_dilated, min_size=min_size)
    
    # 转换为uint8
    mask_adjusted = (mask_cleaned * 255).astype(np.uint8)
    
    # 保存
    output_path = os.path.join(prefix, 'filterRGB/mask-small-refined-adjusted.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(mask_adjusted).save(output_path)
    
    print(f"\n✓ 保存调整后的掩膜: {output_path}")
    print(f"  原始组织比例: {mask_binary.sum() / mask_binary.size * 100:.1f}%")
    print(f"  调整后比例: {mask_cleaned.sum() / mask_cleaned.size * 100:.1f}%")
    print(f"  增加: {(mask_cleaned.sum() - mask_binary.sum()) / mask_binary.size * 100:.1f}%")
    
    print("\n⚠️  使用调整后的掩膜:")
    print(f"  1. 将 mask-small-refined-adjusted.png 重命名为 mask-small-refined.png")
    print(f"  2. 或修改脚本中的掩膜路径")
    print(f"  3. 重新运行注释")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化和调整组织掩膜')
    parser.add_argument('output_base', type=str,
                       help='iSCALE输出目录')
    parser.add_argument('--action', choices=['visualize', 'adjust'], 
                       default='visualize',
                       help='操作类型')
    parser.add_argument('--dilation', type=int, default=5,
                       help='膨胀kernel大小（默认5）')
    parser.add_argument('--min_size', type=int, default=100,
                       help='最小连通域大小（默认100）')
    
    args = parser.parse_args()
    
    if args.action == 'visualize':
        visualize_mask_overlay(args.output_base)
    elif args.action == 'adjust':
        adjust_mask_threshold(args.output_base, args.dilation, args.min_size)


if __name__ == '__main__':
    main()

