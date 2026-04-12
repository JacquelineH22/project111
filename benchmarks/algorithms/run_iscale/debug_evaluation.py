#!/usr/bin/env python
"""
调试iSCALE评估 - 找出Pearson R很低的原因

可能的问题：
1. 坐标映射错误（最可能）
2. 数据归一化不一致
3. 预测矩阵尺寸问题
4. 基因名不匹配
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import anndata as ad


# ==================== 路径配置 ====================
TRUTH_DIR = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/xenium_benchmark/ground_truth'
ISCALE_OUTPUT = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/xenium_benchmark/iscale_output/MotherImage'
ISCALE_PREDICTIONS = os.path.join(ISCALE_OUTPUT, 'iSCALE_output/super_res_gene_expression/cnts-super-refined')

# 也检查其他可能的输出目录
ALTERNATE_OUTPUTS = [
    os.path.join(ISCALE_OUTPUT, 'iSCALE_output/super_res_gene_expression/cnts-super'),
    os.path.join(ISCALE_OUTPUT, 'iSCALE_output/super_res_gene_expression/cnts-super-merged'),
]


def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def debug_step1_load_data():
    """Step 1: 检查数据加载"""
    print_header("Step 1: 检查数据加载")
    
    # 加载ground truth
    print("加载ground truth...")
    h5ad_path = os.path.join(TRUTH_DIR, 'xenium_ground_truth.h5ad')
    
    if os.path.exists(h5ad_path):
        adata = ad.read_h5ad(h5ad_path)
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X
        cnts_truth = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
        locs_truth = adata.obs[['x_pixel', 'y_pixel']].copy()
        locs_truth.columns = ['x', 'y']
    else:
        cnts_truth = pd.read_csv(os.path.join(TRUTH_DIR, 'cnts_truth.tsv'), sep='\t', index_col=0)
        locs_truth = pd.read_csv(os.path.join(TRUTH_DIR, 'locs_truth.tsv'), sep='\t', index_col=0)
    
    print(f"✓ Ground truth: {cnts_truth.shape[0]:,} spots × {cnts_truth.shape[1]} genes")
    print(f"  坐标范围: X=[{locs_truth['x'].min():.0f}, {locs_truth['x'].max():.0f}], "
          f"Y=[{locs_truth['y'].min():.0f}, {locs_truth['y'].max():.0f}]")
    print(f"  表达值范围: {cnts_truth.min().min():.2f} - {cnts_truth.max().max():.2f}")
    
    # 加载iSCALE输出目录的locs和cnts
    print("\n检查iSCALE训练用的数据...")
    iscale_locs_path = os.path.join(ISCALE_OUTPUT, 'locs.tsv')
    iscale_cnts_path = os.path.join(ISCALE_OUTPUT, 'cnts.tsv')
    
    if os.path.exists(iscale_locs_path):
        iscale_locs = pd.read_csv(iscale_locs_path, sep='\t', index_col=0)
        iscale_cnts = pd.read_csv(iscale_cnts_path, sep='\t', index_col=0)
        print(f"✓ iSCALE训练数据: {iscale_cnts.shape[0]:,} spots × {iscale_cnts.shape[1]} genes")
        print(f"  坐标范围: X=[{iscale_locs['x'].min():.0f}, {iscale_locs['x'].max():.0f}], "
              f"Y=[{iscale_locs['y'].min():.0f}, {iscale_locs['y'].max():.0f}]")
        print(f"  表达值范围: {iscale_cnts.min().min():.2f} - {iscale_cnts.max().max():.2f}")
    
    # 加载预测结果
    print("\n加载iSCALE预测...")
    pred_files = list(filter(lambda x: x.endswith('.pickle'), os.listdir(ISCALE_PREDICTIONS)))
    
    if len(pred_files) == 0:
        print(f"✗ 错误: 预测目录为空 {ISCALE_PREDICTIONS}")
        sys.exit(1)
    
    first_gene = pred_files[0].replace('.pickle', '')
    with open(os.path.join(ISCALE_PREDICTIONS, pred_files[0]), 'rb') as f:
        pred_matrix = pickle.load(f)
    
    print(f"✓ 预测结果: {len(pred_files)} 个基因")
    print(f"  第一个基因: {first_gene}")
    print(f"  预测矩阵形状: {pred_matrix.shape}")
    print(f"  预测值范围: {np.nanmin(pred_matrix):.4f} - {np.nanmax(pred_matrix):.4f}")
    print(f"  预测值类型: {pred_matrix.dtype}")
    
    # 检查he.tiff尺寸
    print("\n检查预处理后的H&E图像...")
    he_tiff_path = os.path.join(ISCALE_OUTPUT, 'he.tiff')
    if os.path.exists(he_tiff_path):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        he_img = Image.open(he_tiff_path)
        print(f"✓ he.tiff尺寸: {he_img.size[0]} × {he_img.size[1]}")
        print(f"  预测矩阵与图像尺寸比较:")
        print(f"    预测: {pred_matrix.shape[1]} × {pred_matrix.shape[0]}")
        print(f"    图像: {he_img.size[0]} × {he_img.size[1]}")
        
        # 计算下采样因子
        factor_x = he_img.size[0] / pred_matrix.shape[1]
        factor_y = he_img.size[1] / pred_matrix.shape[0]
        print(f"    下采样因子: X={factor_x:.2f}, Y={factor_y:.2f}")
    
    return cnts_truth, locs_truth, pred_matrix, first_gene


def debug_step2_coordinate_mapping(locs_truth, pred_shape):
    """Step 2: 调试坐标映射"""
    print_header("Step 2: 调试坐标映射")
    
    print("Ground truth spots的坐标范围:")
    print(f"  X: {locs_truth['x'].min():.0f} - {locs_truth['x'].max():.0f}")
    print(f"  Y: {locs_truth['y'].min():.0f} - {locs_truth['y'].max():.0f}")
    
    print(f"\niSCALE预测矩阵的尺寸:")
    print(f"  Height: {pred_shape[0]}, Width: {pred_shape[1]}")
    
    # 尝试不同的映射方法
    print("\n尝试不同的坐标映射方法:")
    
    methods = {
        "方法1：直接使用（不缩放）": {
            'x': locs_truth['x'].astype(int),
            'y': locs_truth['y'].astype(int)
        },
        "方法2：缩放 0.2125→0.5": {
            'x': (locs_truth['x'] * 0.2125 / 0.5).astype(int),
            'y': (locs_truth['y'] * 0.2125 / 0.5).astype(int)
        },
        "方法3：除以16（下采样因子）": {
            'x': (locs_truth['x'] / 16).astype(int),
            'y': (locs_truth['y'] / 16).astype(int)
        },
        "方法4：先缩放再下采样": {
            'x': (locs_truth['x'] * 0.2125 / 0.5 / 16).astype(int),
            'y': (locs_truth['y'] * 0.2125 / 0.5 / 16).astype(int)
        }
    }
    
    for method_name, coords in methods.items():
        x_coords = coords['x']
        y_coords = coords['y']
        
        # 检查有多少坐标在范围内
        valid = (
            (x_coords >= 0) & (x_coords < pred_shape[1]) &
            (y_coords >= 0) & (y_coords < pred_shape[0])
        )
        
        n_valid = valid.sum()
        pct_valid = n_valid / len(valid) * 100
        
        print(f"\n{method_name}:")
        print(f"  映射后X范围: {x_coords.min():.0f} - {x_coords.max():.0f}")
        print(f"  映射后Y范围: {y_coords.min():.0f} - {y_coords.max():.0f}")
        print(f"  有效spots: {n_valid:,} / {len(valid):,} ({pct_valid:.1f}%)")
        
        if pct_valid < 50:
            print(f"  ⚠ 警告: 超过50%的spots坐标超出预测矩阵范围！")
    
    return methods


def debug_step3_test_extraction(cnts_truth, locs_truth, pred_matrix, gene_name, methods):
    """Step 3: 测试不同方法的预测值提取"""
    print_header("Step 3: 测试预测值提取")
    
    truth_values = cnts_truth[gene_name].values
    print(f"测试基因: {gene_name}")
    print(f"Truth值范围: {truth_values.min():.2f} - {truth_values.max():.2f}")
    print(f"Truth平均值: {truth_values.mean():.2f}")
    
    for method_name, coords in methods.items():
        x_coords = coords['x']
        y_coords = coords['y']
        
        # 提取预测值
        pred_values = np.full(len(x_coords), np.nan)
        
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            if 0 <= y < pred_matrix.shape[0] and 0 <= x < pred_matrix.shape[1]:
                pred_values[i] = pred_matrix[y, x]
        
        # 计算相关性
        valid = ~np.isnan(pred_values)
        if valid.sum() < 10:
            print(f"\n{method_name}: ✗ 有效值太少 ({valid.sum()})")
            continue
        
        truth_valid = truth_values[valid]
        pred_valid = pred_values[valid]
        
        try:
            r, p = pearsonr(truth_valid, pred_valid)
            rmse = np.sqrt(np.mean((truth_valid - pred_valid)**2))
            
            print(f"\n{method_name}:")
            print(f"  有效spots: {valid.sum():,}")
            print(f"  Pred值范围: {pred_valid.min():.4f} - {pred_valid.max():.4f}")
            print(f"  Pred平均值: {pred_valid.mean():.4f}")
            print(f"  Pearson R: {r:.4f} (p={p:.2e})")
            print(f"  RMSE: {rmse:.4f}")
            
            if abs(r) > 0.3:
                print(f"  ✓✓✓ 这个方法看起来正确！")
        
        except Exception as e:
            print(f"\n{method_name}: ✗ 计算失败 - {e}")


def debug_step4_visualize(cnts_truth, locs_truth, pred_matrix, gene_name):
    """Step 4: 可视化真实值和预测值的空间分布"""
    print_header("Step 4: 可视化空间分布")
    
    # 创建输出目录
    debug_dir = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/xenium_benchmark/debug_plots'
    os.makedirs(debug_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Ground truth (spot级别)
    ax = axes[0]
    truth_values = cnts_truth[gene_name].values
    scatter = ax.scatter(locs_truth['x'], locs_truth['y'], 
                        c=truth_values, cmap='viridis', s=5)
    ax.set_title(f'Ground Truth - {gene_name}\n(Spot-level, {len(truth_values)} spots)', 
                fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Expression')
    
    # iSCALE预测 (像素级别)
    ax = axes[1]
    im = ax.imshow(pred_matrix, cmap='viridis', aspect='auto')
    ax.set_title(f'iSCALE Prediction - {gene_name}\n(Pixel-level, {pred_matrix.shape})', 
                fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Expression')
    
    # 预测矩阵的统计信息
    ax = axes[2]
    ax.hist(pred_matrix.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.nanmean(pred_matrix), color='red', linestyle='--', 
              linewidth=2, label=f'Mean={np.nanmean(pred_matrix):.4f}')
    ax.set_xlabel('Predicted Expression')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Prediction Distribution - {gene_name}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(debug_dir, f'{gene_name}_spatial_debug.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 保存调试图: {output_path}")
    
    return truth_values


def debug_step5_check_files():
    """Step 5: 检查关键文件"""
    print_header("Step 5: 检查iSCALE输出文件")
    
    # 检查he.tiff
    he_path = os.path.join(ISCALE_OUTPUT, 'he.tiff')
    if os.path.exists(he_path):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(he_path)
        print(f"✓ he.tiff: {img.size[0]} × {img.size[1]} pixels")
    else:
        print(f"✗ 找不到he.tiff")
    
    # 检查locs.tsv和cnts.tsv
    locs_path = os.path.join(ISCALE_OUTPUT, 'locs.tsv')
    cnts_path = os.path.join(ISCALE_OUTPUT, 'cnts.tsv')
    
    if os.path.exists(locs_path):
        locs = pd.read_csv(locs_path, sep='\t', index_col=0)
        print(f"✓ locs.tsv: {len(locs)} spots")
        print(f"  坐标范围: X=[{locs['x'].min():.0f}, {locs['x'].max():.0f}], "
              f"Y=[{locs['y'].min():.0f}, {locs['y'].max():.0f}]")
    
    if os.path.exists(cnts_path):
        cnts = pd.read_csv(cnts_path, sep='\t', index_col=0)
        print(f"✓ cnts.tsv: {cnts.shape[0]} spots × {cnts.shape[1]} genes")
    
    # 检查embeddings文件
    emb_path = os.path.join(ISCALE_OUTPUT, 'embeddings-hist.pickle')
    if os.path.exists(emb_path):
        emb = pickle.load(open(emb_path, 'rb'))
        print(f"✓ embeddings-hist.pickle存在")
        if isinstance(emb, dict):
            for key, val in emb.items():
                if isinstance(val, np.ndarray):
                    print(f"    {key}: {val.shape}")


def debug_step6_direct_comparison():
    """Step 6: 直接比较训练spots的预测"""
    print_header("Step 6: 在训练spots上直接比较")
    
    # 加载训练数据
    cnts_train = pd.read_csv(os.path.join(ISCALE_OUTPUT, 'cnts.tsv'), sep='\t', index_col=0)
    locs_train = pd.read_csv(os.path.join(ISCALE_OUTPUT, 'locs.tsv'), sep='\t', index_col=0)
    
    # 加载预测
    pred_files = list(filter(lambda x: x.endswith('.pickle'), os.listdir(ISCALE_PREDICTIONS)))
    first_gene = pred_files[0].replace('.pickle', '')
    
    with open(os.path.join(ISCALE_PREDICTIONS, pred_files[0]), 'rb') as f:
        pred_matrix = pickle.load(f)
    
    print(f"测试基因: {first_gene}")
    print(f"训练数据: {cnts_train.shape[0]} spots")
    print(f"预测矩阵: {pred_matrix.shape}")
    
    # 方法：从预测矩阵的对应位置提取值
    # iSCALE的预测矩阵是下采样的（factor=16）
    factor = 16
    x_coords = (locs_train['x'] / factor).astype(int)
    y_coords = (locs_train['y'] / factor).astype(int)
    
    print(f"\n使用下采样因子={factor}:")
    print(f"  映射后坐标范围: X=[{x_coords.min()}, {x_coords.max()}], Y=[{y_coords.min()}, {y_coords.max()}]")
    
    # 提取预测值
    pred_values = []
    for x, y in zip(x_coords, y_coords):
        if 0 <= y < pred_matrix.shape[0] and 0 <= x < pred_matrix.shape[1]:
            pred_values.append(pred_matrix[y, x])
        else:
            pred_values.append(np.nan)
    
    pred_values = np.array(pred_values)
    truth_values = cnts_train[first_gene].values
    
    valid = ~np.isnan(pred_values)
    print(f"  有效映射: {valid.sum()} / {len(valid)}")
    
    if valid.sum() > 10:
        r, p = pearsonr(truth_values[valid], pred_values[valid])
        print(f"  Pearson R: {r:.4f}")
        print(f"  这应该是训练集的性能（期望>0.7）")


def main():
    """主调试函数"""
    
    print_header("iSCALE 评估调试工具")
    
    # Step 1: 加载数据
    cnts_truth, locs_truth, pred_matrix, first_gene = debug_step1_load_data()
    
    # Step 2: 测试坐标映射
    methods = debug_step2_coordinate_mapping(locs_truth, pred_matrix.shape)
    
    # Step 3: 测试提取
    debug_step3_test_extraction(cnts_truth, locs_truth, pred_matrix, first_gene, methods)
    
    # Step 4: 可视化
    debug_step4_visualize(cnts_truth, locs_truth, pred_matrix, first_gene)
    
    # Step 5: 检查文件
    debug_step5_check_files()
    
    # Step 6: 直接比较训练数据
    debug_step6_direct_comparison()
    
    print_header("调试完成")
    print("\n💡 根据上述输出，找出Pearson R最高的方法，那就是正确的坐标映射方式")
    print("   然后更新evaluate_iscale_predictions.py中的map_spots_to_pixels函数")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

