#!/usr/bin/env python
"""
将整张Xenium pseudo-Visium数据切分成多个daughter captures
模拟论文中的Xenium benchmarking场景

论文设置：
- 母图：整张Xenium H&E (12mm × 24mm)
- 子图：多个3.2mm × 3.2mm的区域 (D1, D2, ...)
- 每个子图包含该区域的pseudo-Visium spots

你的数据：
- 母图：6.49mm × 5.12mm
- 可以切分成多个3.2mm × 3.2mm的子图
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
from pathlib import Path


def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def visualize_grid(locs, grid_info, output_dir):
    """可视化切分网格"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制所有spots
    ax.scatter(locs['x'], locs['y'], c='lightgray', s=1, alpha=0.5, label='All spots')
    
    # 绘制每个capture区域
    colors = plt.cm.tab20(np.linspace(0, 1, len(grid_info)))
    
    for i, (name, info) in enumerate(grid_info.items()):
        x_min, x_max = info['x_range']
        y_min, y_max = info['y_range']
        n_spots = info['n_spots']
        
        # 绘制矩形框
        rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                            linewidth=2, edgecolor=colors[i], 
                            facecolor=colors[i], alpha=0.2)
        ax.add_patch(rect)
        
        # 添加标签
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        ax.text(cx, cy, f'{name}\n{n_spots} spots', 
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title('Daughter Captures Grid Layout', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'captures_grid.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ 网格可视化: {output_dir}/captures_grid.png")


def split_xenium_data(h5ad_path, he_path, output_base, 
                     capture_size_mm=3.2, pixel_scale=0.2125,
                     overlap_mm=0.0, min_spots=100):
    """
    切分Xenium数据成多个captures
    
    参数：
    - h5ad_path: 输入的h5ad文件
    - he_path: H&E图像路径
    - output_base: 输出目录
    - capture_size_mm: 每个capture的大小（mm）
    - pixel_scale: 像素尺度（µm/pixel）
    - overlap_mm: capture之间的重叠（mm）
    - min_spots: 最少spots数量（少于此数量的capture会被丢弃）
    """
    
    import anndata as ad
    
    print_header("切分Xenium数据为多个Daughter Captures")
    
    print(f"参数设置:")
    print(f"  Capture大小: {capture_size_mm} mm × {capture_size_mm} mm")
    print(f"  像素尺度: {pixel_scale} µm/pixel")
    print(f"  重叠: {overlap_mm} mm")
    print(f"  最少spots: {min_spots}")
    
    # 转换单位
    capture_size_pixels = capture_size_mm * 1000 / pixel_scale  # mm -> pixels
    overlap_pixels = overlap_mm * 1000 / pixel_scale
    step_pixels = capture_size_pixels - overlap_pixels
    
    print(f"\n像素单位:")
    print(f"  Capture大小: {capture_size_pixels:.0f} pixels")
    print(f"  步长: {step_pixels:.0f} pixels")
    
    # ========== 1. 读取数据 ==========
    print_header("1. 读取数据")
    
    print("读取h5ad...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  ✓ {adata.shape[0]:,} spots × {adata.shape[1]} genes")
    
    # 提取表达矩阵和位置
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    cnts = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    locs = adata.obs[['x_pixel', 'y_pixel']].copy()
    locs.columns = ['x', 'y']
    
    # 数据范围
    x_min, x_max = locs['x'].min(), locs['x'].max()
    y_min, y_max = locs['y'].min(), locs['y'].max()
    x_span = x_max - x_min
    y_span = y_max - y_min
    
    x_mm = x_span * pixel_scale / 1000
    y_mm = y_span * pixel_scale / 1000
    
    print(f"\n数据范围:")
    print(f"  X: {x_min:.0f} - {x_max:.0f} pixels ({x_mm:.2f} mm)")
    print(f"  Y: {y_min:.0f} - {y_max:.0f} pixels ({y_mm:.2f} mm)")
    
    # ========== 2. 计算网格 ==========
    print_header("2. 计算切分网格")
    
    # 计算可以切分成多少个captures
    n_x = int((x_span - overlap_pixels) / step_pixels)
    n_y = int((y_span - overlap_pixels) / step_pixels)
    
    print(f"可以切分成: {n_x} × {n_y} = {n_x * n_y} 个captures")
    
    if n_x * n_y == 0:
        print(f"⚠ 警告: 数据太小，无法切分成{capture_size_mm}mm的captures")
        print(f"  建议: 减小capture_size_mm或使用完整数据作为单个capture")
        return
    
    # ========== 3. 切分数据 ==========
    print_header("3. 切分数据")
    
    grid_info = {}
    capture_id = 1
    
    for i in range(n_x):
        for j in range(n_y):
            # 计算capture的边界
            cap_x_min = x_min + i * step_pixels
            cap_x_max = cap_x_min + capture_size_pixels
            cap_y_min = y_min + j * step_pixels
            cap_y_max = cap_y_min + capture_size_pixels
            
            # 选择该区域内的spots
            mask = (
                (locs['x'] >= cap_x_min) & (locs['x'] < cap_x_max) &
                (locs['y'] >= cap_y_min) & (locs['y'] < cap_y_max)
            )
            
            n_spots_in_capture = mask.sum()
            
            if n_spots_in_capture < min_spots:
                print(f"  ⊗ D{capture_id} ({i},{j}): 只有{n_spots_in_capture}个spots，跳过")
                continue
            
            # 提取该区域的数据
            locs_cap = locs[mask].copy()
            cnts_cap = cnts.loc[locs_cap.index].copy()
            
            # 保存信息
            capture_name = f"D{capture_id}"
            grid_info[capture_name] = {
                'grid_pos': (i, j),
                'x_range': (cap_x_min, cap_x_max),
                'y_range': (cap_y_min, cap_y_max),
                'n_spots': n_spots_in_capture,
                'locs': locs_cap,
                'cnts': cnts_cap
            }
            
            print(f"  ✓ {capture_name} ({i},{j}): {n_spots_in_capture:,} spots, "
                  f"X=[{cap_x_min:.0f},{cap_x_max:.0f}], Y=[{cap_y_min:.0f},{cap_y_max:.0f}]")
            
            capture_id += 1
    
    if len(grid_info) == 0:
        print("✗ 没有生成任何有效的capture！")
        return
    
    print(f"\n总共生成 {len(grid_info)} 个有效的captures")
    
    # ========== 4. 保存数据 ==========
    print_header("4. 保存数据")
    
    # 创建输出目录
    unaligned_dir = os.path.join(output_base, 'DaughterCaptures/UnallignedToMother')
    aligned_dir = os.path.join(output_base, 'DaughterCaptures/AllignedToMother')
    mother_dir = os.path.join(output_base, 'MotherImage')
    
    os.makedirs(unaligned_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(mother_dir, exist_ok=True)
    
    # 保存每个capture
    for name, info in grid_info.items():
        # 保存到AllignedToMother（坐标已经在母图坐标系中）
        cap_dir = os.path.join(aligned_dir, name)
        os.makedirs(cap_dir, exist_ok=True)
        
        info['cnts'].to_csv(os.path.join(cap_dir, 'cnts.tsv'), sep='\t')
        info['locs'].to_csv(os.path.join(cap_dir, 'locs.tsv'), sep='\t')
        
        print(f"  ✓ {name}: {cap_dir}")
    
    # 复制H&E图像
    he_dst = os.path.join(mother_dir, 'he-raw.png')
    shutil.copy2(he_path, he_dst)
    print(f"  ✓ H&E图像: {he_dst}")
    
    # 创建radius文件
    window_size = 55  # µm
    radius_raw = (window_size / 2) / pixel_scale
    with open(os.path.join(mother_dir, 'radius-raw.txt'), 'w') as f:
        f.write(f"{radius_raw:.2f}\n")
    print(f"  ✓ radius-raw.txt: {radius_raw:.2f} pixels")
    
    # 可视化
    visualize_grid(locs, grid_info, output_base)
    
    # ========== 5. 生成摘要 ==========
    print_header("✅ 切分完成！")
    
    print(f"输出目录: {output_base}")
    print(f"\n生成的captures:")
    for name, info in grid_info.items():
        print(f"  • {name}: {info['n_spots']:,} spots")
    
    print(f"\n总计:")
    print(f"  • Captures数量: {len(grid_info)}")
    print(f"  • 总spots数: {sum(info['n_spots'] for info in grid_info.values()):,}")
    print(f"  • 原始spots数: {len(locs):,}")
    print(f"  • 覆盖率: {sum(info['n_spots'] for info in grid_info.values()) / len(locs) * 100:.1f}%")
    
    print(f"\n📂 目录结构:")
    print(f"{output_base}/")
    print(f"├── DaughterCaptures/")
    print(f"│   └── AllignedToMother/")
    for name in grid_info.keys():
        print(f"│       ├── {name}/")
        print(f"│       │   ├── cnts.tsv")
        print(f"│       │   └── locs.tsv")
    print(f"├── MotherImage/")
    print(f"│   ├── he-raw.png")
    print(f"│   └── radius-raw.txt")
    print(f"└── captures_grid.png")
    
    print(f"\n🎯 下一步:")
    print(f"1. 查看网格图: {output_base}/captures_grid.png")
    print(f"2. 运行iSCALE:")
    print(f"   cd /data1/linxin/Benchmark/iSCALE/iSCALE")
    print(f"   # 修改run_iscale_xenium.sh中的:")
    print(f"   #   INPUT_BASE=\"{output_base}\"")
    print(f"   #   daughterCapture_folders=({' '.join([f'\\"{name}\\"' for name in grid_info.keys()])})")
    print(f"   bash run_iscale_xenium.sh")


if __name__ == '__main__':
    # ==================== 配置参数 ====================
    
    # 输入文件
    H5AD_PATH = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/SC/simulated_data.h5ad'
    HE_PATH = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/HE/align_he.png'
    
    # 输出目录
    OUTPUT_BASE = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/iscale_input_multi_captures'
    
    # 切分参数（参考论文）
    CAPTURE_SIZE_MM = 3.2    # 每个capture的大小（mm），论文用的3.2mm
    PIXEL_SCALE = 0.2125     # Xenium像素尺度（µm/pixel）
    OVERLAP_MM = 0.0         # capture之间的重叠（mm），0表示不重叠
    MIN_SPOTS = 100          # 最少spots数量，少于此数量的capture会被丢弃
    
    # ==================== 运行切分 ====================
    
    split_xenium_data(
        h5ad_path=H5AD_PATH,
        he_path=HE_PATH,
        output_base=OUTPUT_BASE,
        capture_size_mm=CAPTURE_SIZE_MM,
        pixel_scale=PIXEL_SCALE,
        overlap_mm=OVERLAP_MM,
        min_spots=MIN_SPOTS
    )

