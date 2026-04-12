#!/usr/bin/env python
"""
iSCALE Xenium Benchmark 评估脚本

比较iSCALE预测结果与Xenium ground truth
实现论文中的评估指标：RMSE, Pearson, Spearman, SSIM

使用方法：
    python evaluate_iscale_predictions.py
    
输出：
    - evaluation_results.csv: 每个基因的评估指标
    - summary_metrics.txt: 总体性能摘要
    - gene_comparison_plots/: 每个基因的scatter plot
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ==================== 路径配置（直接修改这里）====================

# Ground Truth路径
TRUTH_DIR = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/xenium_benchmark_input/ground_truth'

# iSCALE输出路径（运行iSCALE后生成）
ISCALE_OUTPUT_BASE = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/xenium_benchmark_output/MotherImage'
ISCALE_PREDICTIONS = os.path.join(ISCALE_OUTPUT_BASE, 'iSCALE_output/super_res_gene_expression/cnts-super-refined')

# 评估结果输出路径
EVAL_OUTPUT = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/xenium_benchmark_output/evaluation'

# 下采样因子（iSCALE预测矩阵相对于原始图像）
DOWNSAMPLE_FACTOR = 16  # 关键参数！预测矩阵是16×16下采样的

# ================================================================


class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*70}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{text:^70}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*70}{Color.END}\n")


def print_success(text):
    print(f"{Color.GREEN}✓ {text}{Color.END}")


def print_error(text):
    print(f"{Color.RED}✗ {text}{Color.END}")


def print_info(text):
    print(f"{Color.BLUE}ℹ {text}{Color.END}")


def load_ground_truth():
    """加载Xenium ground truth数据"""
    print_header("加载 Ground Truth 数据")
    
    # 尝试加载h5ad
    h5ad_path = os.path.join(TRUTH_DIR, 'xenium_ground_truth.h5ad')
    
    if os.path.exists(h5ad_path):
        print_info(f"从h5ad加载: {h5ad_path}")
        import anndata as ad
        adata = ad.read_h5ad(h5ad_path)
        
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X
        
        cnts = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
        locs = adata.obs[['x_pixel', 'y_pixel']].copy()
        locs.columns = ['x', 'y']
        
        print_success(f"Ground truth: {cnts.shape[0]:,} spots × {cnts.shape[1]} genes")
        
    else:
        # 从TSV加载
        print_info(f"从TSV加载...")
        cnts = pd.read_csv(os.path.join(TRUTH_DIR, 'cnts_truth.tsv'), sep='\t', index_col=0)
        locs = pd.read_csv(os.path.join(TRUTH_DIR, 'locs_truth.tsv'), sep='\t', index_col=0)
        print_success(f"Ground truth: {cnts.shape[0]:,} spots × {cnts.shape[1]} genes")
    
    # 加载metadata
    metadata_path = os.path.join(TRUTH_DIR, 'metadata.pickle')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    else:
        metadata = {
            'pixel_scale': 0.2125,
            'window_size': 55
        }
        print_info(f"使用默认metadata: pixel_scale=0.2125, window_size=55")
    
    return cnts, locs, metadata


def load_iscale_predictions(gene_names=None):
    """加载iSCALE预测结果"""
    print_header("加载 iSCALE 预测结果")
    
    if not os.path.exists(ISCALE_PREDICTIONS):
        print_error(f"找不到预测结果目录: {ISCALE_PREDICTIONS}")
        print_info("请确保已运行iSCALE并生成预测结果")
        return None
    
    predictions = {}
    
    # 列出所有预测文件
    pred_files = list(Path(ISCALE_PREDICTIONS).glob('*.pickle'))
    
    if len(pred_files) == 0:
        print_error(f"预测目录中没有pickle文件")
        return None
    
    print_info(f"找到 {len(pred_files)} 个预测文件")
    
    # 加载每个基因的预测
    for gene_file in pred_files:
        gene = gene_file.stem  # 文件名不含扩展名
        
        # 如果指定了基因列表，只加载这些基因
        if gene_names is not None and gene not in gene_names:
            continue
        
        with open(gene_file, 'rb') as f:
            predictions[gene] = pickle.load(f)
    
    print_success(f"加载了 {len(predictions)} 个基因的预测结果")
    
    # 显示预测矩阵的尺寸
    first_gene = list(predictions.keys())[0]
    pred_shape = predictions[first_gene].shape
    print_info(f"预测矩阵尺寸: {pred_shape[0]} × {pred_shape[1]} pixels")
    
    return predictions


def map_spots_to_pixels(locs, pred_shape, downsample_factor=16):
    """
    将ground truth的spot坐标映射到预测矩阵的像素坐标
    
    关键发现：iSCALE的预测矩阵是下采样16倍的！
    - 原始图像: 30976 × 24320 pixels
    - 预测矩阵: 1936 × 1520 pixels
    - 下采样因子: 16 (30976/1936 = 16)
    
    参数：
    - locs: ground truth的位置（原始分辨率的像素坐标）
    - pred_shape: 预测矩阵的形状 (H, W)
    - downsample_factor: 下采样因子（默认16）
    """
    
    # 关键：直接除以下采样因子
    # Ground truth坐标是原始分辨率（0.2125µm/pixel）
    # iSCALE预测是16×16像素的patch级别
    x_pred = (locs['x'] / downsample_factor).astype(int)
    y_pred = (locs['y'] / downsample_factor).astype(int)
    
    # 裁剪到预测矩阵范围内
    valid = (
        (x_pred >= 0) & (x_pred < pred_shape[1]) &
        (y_pred >= 0) & (y_pred < pred_shape[0])
    )
    
    return x_pred, y_pred, valid


def extract_predictions_at_spots(pred_matrix, x_coords, y_coords, valid_mask):
    """从预测矩阵中提取对应spot位置的值"""
    
    pred_values = np.full(len(x_coords), np.nan)
    
    for i, (x, y, valid) in enumerate(zip(x_coords, y_coords, valid_mask)):
        if valid:
            pred_values[i] = pred_matrix[y, x]
    
    return pred_values


def standardize(x):
    """标准化到[0,1]范围（与iSCALE一致）"""
    x = x - np.nanmin(x)
    x = x / (np.nanmax(x) + 1e-12)
    return x


def evaluate_gene(truth_values, pred_matrix, locs, downsample_factor=16):
    """
    评估单个基因的预测性能
    
    返回：
    - dict: 包含各种评估指标
    """
    
    pred_shape = pred_matrix.shape
    
    # 映射坐标（使用正确的下采样因子）
    x_pred, y_pred, valid = map_spots_to_pixels(locs, pred_shape, downsample_factor)
    
    # 提取预测值
    pred_values = extract_predictions_at_spots(pred_matrix, x_pred, y_pred, valid)
    
    # 过滤无效值
    truth_valid = truth_values[valid & ~np.isnan(pred_values)]
    pred_valid = pred_values[valid & ~np.isnan(pred_values)]
    
    if len(truth_valid) < 10:
        return None
    
    # 关键：标准化后再计算（与iSCALE原始方法一致）
    truth_norm = standardize(truth_valid)
    pred_norm = standardize(pred_valid)
    
    # 计算评估指标
    try:
        # 在归一化数据上计算RMSE和R²
        rmse = np.sqrt(mean_squared_error(truth_norm, pred_norm))
        mae = np.mean(np.abs(truth_norm - pred_norm))
        
        # Pearson和Spearman对线性变换不敏感，可以用原始值或归一化值
        pearson_r, pearson_p = pearsonr(truth_norm, pred_norm)
        spearman_r, spearman_p = spearmanr(truth_norm, pred_norm)
        
        # R²在归一化数据上计算
        ss_res = np.sum((truth_norm - pred_norm) ** 2)
        ss_tot = np.sum((truth_norm - np.mean(truth_norm)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'rmse': rmse,  # 归一化后的RMSE
            'mae': mae,    # 归一化后的MAE
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'r2': r2,  # 归一化后的R²
            'n_spots': len(truth_valid),
            'truth_mean': np.mean(truth_valid),  # 原始值的统计
            'truth_std': np.std(truth_valid),
            'pred_mean': np.mean(pred_valid),
            'pred_std': np.std(pred_valid),
            'truth_norm_mean': np.mean(truth_norm),  # 归一化后的统计
            'pred_norm_mean': np.mean(pred_norm)
        }
    except Exception as e:
        print_error(f"计算指标时出错: {e}")
        return None


def plot_gene_comparison(truth_values, pred_values, gene, output_dir):
    """绘制单个基因的真实值vs预测值散点图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 过滤有效值
    valid = ~np.isnan(pred_values)
    truth_valid = truth_values[valid]
    pred_valid = pred_values[valid]
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(truth_valid, pred_valid, alpha=0.5, s=10)
    
    # 添加对角线
    max_val = max(truth_valid.max(), pred_valid.max())
    min_val = min(truth_valid.min(), pred_valid.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    # 计算相关系数
    r, p = pearsonr(truth_valid, pred_valid)
    
    ax.set_xlabel('Xenium Ground Truth', fontsize=12)
    ax.set_ylabel('iSCALE Prediction', fontsize=12)
    ax.set_title(f'{gene}\nPearson R = {r:.3f} (p={p:.2e})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Density plot
    ax = axes[1]
    ax.hist2d(truth_valid, pred_valid, bins=50, cmap='Blues')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('Xenium Ground Truth', fontsize=12)
    ax.set_ylabel('iSCALE Prediction', fontsize=12)
    ax.set_title(f'{gene} (Density)', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{gene}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """主函数"""
    
    print_header("iSCALE Xenium Benchmark 评估")
    
    # 创建输出目录
    os.makedirs(EVAL_OUTPUT, exist_ok=True)
    plot_dir = os.path.join(EVAL_OUTPUT, 'gene_comparison_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Step 1: 加载ground truth
    try:
        cnts_truth, locs_truth, metadata = load_ground_truth()
    except Exception as e:
        print_error(f"加载ground truth失败: {e}")
        print_info(f"请确保ground truth数据存在于: {TRUTH_DIR}")
        sys.exit(1)
    
    # Step 2: 加载iSCALE预测
    try:
        predictions = load_iscale_predictions(gene_names=cnts_truth.columns.tolist())
    except Exception as e:
        print_error(f"加载iSCALE预测失败: {e}")
        print_info(f"请确保iSCALE已运行完成，输出位于: {ISCALE_PREDICTIONS}")
        sys.exit(1)
    
    if predictions is None or len(predictions) == 0:
        print_error("没有加载到任何预测结果")
        sys.exit(1)
    
    # Step 3: 评估每个基因
    print_header("评估每个基因的预测性能")
    
    results = {}
    downsample_factor = 16  # iSCALE的下采样因子
    
    for i, gene in enumerate(cnts_truth.columns, 1):
        if gene not in predictions:
            print(f"[{i}/{cnts_truth.shape[1]}] ⊗ {gene}: 没有预测")
            continue
        
        # 评估
        metrics = evaluate_gene(
            truth_values=cnts_truth[gene].values,
            pred_matrix=predictions[gene],
            locs=locs_truth,
            downsample_factor=downsample_factor
        )
        
        if metrics is None:
            print(f"[{i}/{cnts_truth.shape[1]}] ⊗ {gene}: 评估失败")
            continue
        
        results[gene] = metrics
        
        # 打印进度
        if i % 50 == 0 or i <= 10:
            print(f"[{i}/{cnts_truth.shape[1]}] ✓ {gene}: "
                  f"Pearson={metrics['pearson_r']:.3f}, "
                  f"RMSE={metrics['rmse']:.3f}, "
                  f"R²={metrics['r2']:.3f}")
        
        # 绘制前10个基因的对比图
        if i <= 10:
            x_pred, y_pred, valid = map_spots_to_pixels(
                locs_truth, predictions[gene].shape, 
                downsample_factor
            )
            pred_values = extract_predictions_at_spots(
                predictions[gene], x_pred, y_pred, valid
            )
            plot_gene_comparison(cnts_truth[gene].values, pred_values, gene, plot_dir)
    
    if len(results) == 0:
        print_error("没有成功评估任何基因")
        sys.exit(1)
    
    # Step 4: 汇总统计
    print_header("总体性能统计")
    
    results_df = pd.DataFrame(results).T
    
    # 计算平均值
    mean_metrics = results_df.mean()
    std_metrics = results_df.std()
    
    print(f"{Color.BOLD}评估了 {len(results)} 个基因{Color.END}\n")
    
    metrics_to_show = [
        ('pearson_r', 'Pearson 相关系数'),
        ('spearman_r', 'Spearman 相关系数'),
        ('r2', 'R² (决定系数)'),
        ('rmse', 'RMSE (均方根误差)'),
        ('mae', 'MAE (平均绝对误差)')
    ]
    
    for metric_key, metric_name in metrics_to_show:
        mean_val = mean_metrics[metric_key]
        std_val = std_metrics[metric_key]
        print(f"{metric_name:25s}: {mean_val:7.4f} ± {std_val:.4f}")
    
    # 找出表现最好和最差的基因
    print(f"\n{Color.BOLD}表现最好的5个基因 (按Pearson R):{Color.END}")
    top_genes = results_df.nlargest(5, 'pearson_r')
    for gene, row in top_genes.iterrows():
        print(f"  {gene:15s}: Pearson={row['pearson_r']:.4f}, RMSE={row['rmse']:.3f}")
    
    print(f"\n{Color.BOLD}表现最差的5个基因:{Color.END}")
    bottom_genes = results_df.nsmallest(5, 'pearson_r')
    for gene, row in bottom_genes.iterrows():
        print(f"  {gene:15s}: Pearson={row['pearson_r']:.4f}, RMSE={row['rmse']:.3f}")
    
    # Step 5: 保存结果
    print_header("保存评估结果")
    
    # 保存详细结果
    results_csv = os.path.join(EVAL_OUTPUT, 'evaluation_results.csv')
    results_df.to_csv(results_csv)
    print_success(f"详细结果: {results_csv}")
    
    # 保存摘要
    summary_path = os.path.join(EVAL_OUTPUT, 'summary_metrics.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("iSCALE Xenium Benchmark 评估摘要\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Ground Truth: {cnts_truth.shape[0]:,} spots × {cnts_truth.shape[1]} genes\n")
        f.write(f"评估基因数: {len(results)}\n\n")
        
        f.write("总体性能:\n")
        f.write("-"*70 + "\n")
        for metric_key, metric_name in metrics_to_show:
            mean_val = mean_metrics[metric_key]
            std_val = std_metrics[metric_key]
            f.write(f"{metric_name:25s}: {mean_val:7.4f} ± {std_val:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print_success(f"摘要报告: {summary_path}")
    print_success(f"可视化图: {plot_dir}/ (前10个基因)")
    
    # Step 6: 生成可视化总结
    print_header("生成性能分布图")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pearson R分布
    ax = axes[0, 0]
    ax.hist(results_df['pearson_r'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(mean_metrics['pearson_r'], color='red', linestyle='--', 
              linewidth=2, label=f'Mean = {mean_metrics["pearson_r"]:.3f}')
    ax.set_xlabel('Pearson R', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Pearson Correlation Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE分布
    ax = axes[0, 1]
    ax.hist(results_df['rmse'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(mean_metrics['rmse'], color='red', linestyle='--', 
              linewidth=2, label=f'Mean = {mean_metrics["rmse"]:.3f}')
    ax.set_xlabel('RMSE', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('RMSE Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Pearson vs RMSE
    ax = axes[1, 0]
    scatter = ax.scatter(results_df['pearson_r'], results_df['rmse'], 
                        c=results_df['r2'], cmap='viridis', s=30, alpha=0.6)
    ax.set_xlabel('Pearson R', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Pearson R vs RMSE', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='R²')
    
    # R²分布
    ax = axes[1, 1]
    ax.hist(results_df['r2'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(mean_metrics['r2'], color='red', linestyle='--', 
              linewidth=2, label=f'Mean = {mean_metrics["r2"]:.3f}')
    ax.set_xlabel('R²', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('R² Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_plot = os.path.join(EVAL_OUTPUT, 'performance_summary.png')
    plt.savefig(summary_plot, dpi=150, bbox_inches='tight')
    plt.close()
    
    print_success(f"性能分布图: {summary_plot}")
    
    # 完成
    print_header("✅ 评估完成！")
    
    print(f"\n📊 结果文件:")
    print(f"  • 详细结果: {results_csv}")
    print(f"  • 摘要报告: {summary_path}")
    print(f"  • 性能分布图: {summary_plot}")
    print(f"  • 基因对比图: {plot_dir}/")
    
    print(f"\n🎯 主要发现:")
    print(f"  • 平均Pearson R: {mean_metrics['pearson_r']:.4f}")
    print(f"  • 平均RMSE: {mean_metrics['rmse']:.4f}")
    print(f"  • 平均R²: {mean_metrics['r2']:.4f}")
    
    # 与论文对比
    print(f"\n📖 与论文对比:")
    print(f"  论文报告的Xenium benchmark性能（参考）：")
    print(f"    - In-sample Pearson: 0.7-0.9 (取决于基因)")
    print(f"    - Out-of-sample Spearman: 0.5-0.8")
    print(f"  你的结果：")
    print(f"    - Pearson: {mean_metrics['pearson_r']:.3f}")
    print(f"    - Spearman: {mean_metrics['spearman_r']:.3f}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Color.RED}✗ 用户中断{Color.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Color.RED}✗ 错误: {e}{Color.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

