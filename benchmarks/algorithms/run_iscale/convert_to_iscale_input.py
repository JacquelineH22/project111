#!/usr/bin/env python
"""
将Xenium pseudo-Visium数据转换为iSCALE输入格式

输入：
  - H&E图像: /data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/HE/align_he.png
  - ST数据: /data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/SC/simulated_data.h5ad

输出：
  - iSCALE格式数据: /data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/iscale_input/

作者: AI Assistant
日期: 2026-01-16
"""

import os
import sys
import shutil
import pandas as pd
import anndata as ad
import numpy as np
from pathlib import Path


def print_section(title):
    """打印分节标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_step(step_num, total, description):
    """打印步骤信息"""
    print(f"\n[{step_num}/{total}] {description}")


def print_success(message):
    """打印成功信息"""
    print(f"  ✓ {message}")


def print_error(message):
    """打印错误信息"""
    print(f"  ✗ {message}")


def print_info(message):
    """打印提示信息"""
    print(f"  ℹ {message}")


def main():
    """主函数"""
    
    # ==================== 配置参数 ====================
    H5AD_PATH = '/data1/hounaiqiao/wzr/Simulated_Xenium/brca_rep1/w55/simulated_square_spot_data.h5ad'
    HE_PATH = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/HE/align_he.png'
    OUTPUT_BASE = '/data1/hounaiqiao/wzr/Simulated_Xenium/brca_rep1/w55/iscale_input'
    
    # Xenium参数
    WINDOW_SIZE = 55      # µm (spot直径)
    PIXEL_SCALE = 0.2125  # µm/pixel (Xenium原始分辨率)
    
    TOTAL_STEPS = 8
    
    print_section("转换 Xenium Pseudo-Visium → iSCALE 输入格式")
    
    print_info(f"H&E图像: {HE_PATH}")
    print_info(f"ST数据: {H5AD_PATH}")
    print_info(f"输出目录: {OUTPUT_BASE}")
    
    # ==================== Step 1: 检查输入文件 ====================
    print_step(1, TOTAL_STEPS, "检查输入文件...")
    
    if not os.path.exists(H5AD_PATH):
        print_error(f"找不到h5ad文件: {H5AD_PATH}")
        sys.exit(1)
    print_success(f"h5ad文件存在 ({os.path.getsize(H5AD_PATH)/(1024**3):.2f} GB)")
    
    if not os.path.exists(HE_PATH):
        print_error(f"找不到H&E图像: {HE_PATH}")
        sys.exit(1)
    print_success(f"H&E图像存在 ({os.path.getsize(HE_PATH)/(1024**2):.2f} MB)")
    
    # ==================== Step 2: 读取h5ad文件 ====================
    print_step(2, TOTAL_STEPS, "读取h5ad文件...")
    
    try:
        adata = ad.read_h5ad(H5AD_PATH)
        print_success(f"成功读取AnnData对象")
        print_info(f"维度: {adata.shape[0]:,} spots × {adata.shape[1]} genes")
        print_info(f"Spot IDs: {adata.obs_names[0]} ... {adata.obs_names[-1]}")
        print_info(f"基因: {adata.var_names[0]}, {adata.var_names[1]}, ...")
    except Exception as e:
        print_error(f"读取h5ad失败: {e}")
        sys.exit(1)
    
    # ==================== Step 3: 提取表达矩阵 ====================
    print_step(3, TOTAL_STEPS, "提取基因表达矩阵...")
    
    try:
        # 转换为稠密矩阵
        if hasattr(adata.X, 'toarray'):
            X_dense = adata.X.toarray()
            print_info("从稀疏矩阵转换为稠密矩阵")
        else:
            X_dense = adata.X
        
        # 创建DataFrame (spots × genes)
        cnts = pd.DataFrame(
            X_dense,
            index=adata.obs_names,
            columns=adata.var_names
        )
        
        print_success(f"表达矩阵: {cnts.shape[0]:,} spots × {cnts.shape[1]} genes")
        print_info(f"表达值范围: {cnts.min().min():.2f} - {cnts.max().max():.2f}")
        print_info(f"平均表达: {cnts.mean().mean():.2f}")
        print_info(f"非零率: {(cnts > 0).sum().sum() / cnts.size * 100:.2f}%")
        
    except Exception as e:
        print_error(f"提取表达矩阵失败: {e}")
        sys.exit(1)
    
    # ==================== Step 4: 提取位置信息 ====================
    print_step(4, TOTAL_STEPS, "提取位置信息...")
    
    try:
        # 检查必需的列
        if 'x_pixel' not in adata.obs.columns or 'y_pixel' not in adata.obs.columns:
            print_error("h5ad.obs中缺少x_pixel或y_pixel列")
            print_info(f"可用列: {list(adata.obs.columns)}")
            sys.exit(1)
        
        # 提取坐标并重命名为iSCALE期望的列名
        locs = adata.obs[['x_pixel', 'y_pixel']].copy()
        locs.columns = ['x', 'y']
        
        # 统计信息
        x_min, x_max = locs['x'].min(), locs['x'].max()
        y_min, y_max = locs['y'].min(), locs['y'].max()
        x_span = x_max - x_min
        y_span = y_max - y_min
        
        print_success(f"位置信息: {locs.shape[0]:,} spots")
        print_info(f"X坐标范围: {x_min:.2f} - {x_max:.2f} pixels (跨度: {x_span:.2f})")
        print_info(f"Y坐标范围: {y_min:.2f} - {y_max:.2f} pixels (跨度: {y_span:.2f})")
        
        # 计算物理尺寸
        x_mm = x_span * PIXEL_SCALE / 1000
        y_mm = y_span * PIXEL_SCALE / 1000
        print_info(f"覆盖范围: {x_mm:.2f}mm × {y_mm:.2f}mm")
        
    except Exception as e:
        print_error(f"提取位置信息失败: {e}")
        sys.exit(1)
    
    # ==================== Step 5: 验证数据一致性 ====================
    print_step(5, TOTAL_STEPS, "验证数据一致性...")
    
    try:
        # 检查索引匹配
        if not (cnts.index == locs.index).all():
            print_error("表达矩阵和位置信息的索引不匹配！")
            print_info(f"cnts索引: {cnts.index[:3].tolist()} ...")
            print_info(f"locs索引: {locs.index[:3].tolist()} ...")
            sys.exit(1)
        
        print_success("索引完全匹配")
        
        # 检查spots数量
        assert cnts.shape[0] == locs.shape[0], "Spots数量不一致"
        print_success(f"Spots数量一致: {cnts.shape[0]:,}")
        
        # 检查NaN值
        if cnts.isna().any().any():
            n_nan = cnts.isna().sum().sum()
            print_info(f"表达矩阵中有{n_nan}个NaN，将填充为0")
            cnts = cnts.fillna(0)
        
        if locs.isna().any().any():
            print_error("位置信息中有NaN值！")
            sys.exit(1)
        
        print_success("数据验证通过")
        
    except Exception as e:
        print_error(f"数据验证失败: {e}")
        sys.exit(1)
    
    # ==================== Step 6: 创建iSCALE目录结构 ====================
    print_step(6, TOTAL_STEPS, "创建iSCALE目录结构...")
    
    try:
        # 创建目录
        capture_dir = os.path.join(OUTPUT_BASE, 'DaughterCaptures/AllignedToMother/D1')
        mother_dir = os.path.join(OUTPUT_BASE, 'MotherImage')
        
        os.makedirs(capture_dir, exist_ok=True)
        os.makedirs(mother_dir, exist_ok=True)
        
        print_success(f"Capture目录: {capture_dir}")
        print_success(f"Mother目录: {mother_dir}")
        
    except Exception as e:
        print_error(f"创建目录失败: {e}")
        sys.exit(1)
    
    # ==================== Step 7: 保存数据文件 ====================
    print_step(7, TOTAL_STEPS, "保存数据文件...")
    
    try:
        # 保存cnts.tsv
        cnts_path = os.path.join(capture_dir, 'cnts.tsv')
        print_info("保存 cnts.tsv ...")
        cnts.to_csv(cnts_path, sep='\t')
        cnts_size = os.path.getsize(cnts_path) / (1024 * 1024)
        print_success(f"cnts.tsv ({cnts_size:.2f} MB)")
        
        # 保存locs.tsv
        locs_path = os.path.join(capture_dir, 'locs.tsv')
        print_info("保存 locs.tsv ...")
        locs.to_csv(locs_path, sep='\t')
        locs_size = os.path.getsize(locs_path) / (1024 * 1024)
        print_success(f"locs.tsv ({locs_size:.2f} MB)")
        
        # 验证文件可读
        test_cnts = pd.read_csv(cnts_path, sep='\t', index_col=0, nrows=5)
        test_locs = pd.read_csv(locs_path, sep='\t', index_col=0)
        print_success("文件验证: 可以正常读取")
        
    except Exception as e:
        print_error(f"保存数据文件失败: {e}")
        sys.exit(1)
    
    # ==================== Step 8: 复制H&E图像和创建配置 ====================
    print_step(8, TOTAL_STEPS, "复制H&E图像和创建配置文件...")
    
    try:
        # 复制H&E图像
        he_dst = os.path.join(mother_dir, 'he-raw.png')
        print_info(f"复制 H&E图像...")
        shutil.copy2(HE_PATH, he_dst)
        he_size = os.path.getsize(he_dst) / (1024 * 1024)
        print_success(f"he-raw.png ({he_size:.2f} MB)")
        
        # 创建radius文件
        radius_pixel = (WINDOW_SIZE / 2) / PIXEL_SCALE
        radius_path = os.path.join(mother_dir, 'radius-raw.txt')
        with open(radius_path, 'w') as f:
            f.write(f"{radius_pixel:.2f}\n")
        print_success(f"radius-raw.txt ({radius_pixel:.2f} pixels = {WINDOW_SIZE/2} µm)")
        
    except Exception as e:
        print_error(f"复制文件失败: {e}")
        sys.exit(1)
    
    # ==================== 完成 ====================
    print_section("✅ 转换完成！")
    
    print("\n📂 输出目录结构:")
    print(f"{OUTPUT_BASE}/")
    print(f"├── DaughterCaptures/")
    print(f"│   └── AllignedToMother/")
    print(f"│       └── D1/")
    print(f"│           ├── cnts.tsv     ({cnts.shape[0]:,} spots × {cnts.shape[1]} genes, {cnts_size:.2f} MB)")
    print(f"│           └── locs.tsv     ({locs.shape[0]:,} spots, {locs_size:.2f} MB)")
    print(f"└── MotherImage/")
    print(f"    ├── he-raw.png           (30786×24241 pixels, {he_size:.2f} MB)")
    print(f"    └── radius-raw.txt       ({radius_pixel:.2f} pixels)")
    
    print("\n📊 数据摘要:")
    print(f"  • Spots数量:      {cnts.shape[0]:,}")
    print(f"  • 基因数量:       {cnts.shape[1]}")
    print(f"  • 覆盖范围:       {x_mm:.2f}mm × {y_mm:.2f}mm")
    print(f"  • 图像分辨率:     {PIXEL_SCALE} µm/pixel (Xenium原始)")
    print(f"  • Spot直径:       {WINDOW_SIZE} µm")
    print(f"  • Spot半径:       {radius_pixel:.2f} pixels")
    
    print("\n🎯 下一步操作:")
    print("\n# 1. 进入iSCALE工作目录")
    print("cd /data1/linxin/Benchmark/iSCALE/iSCALE")
    
    print("\n# 2. 创建运行脚本")
    print("cat > run_my_xenium.sh << 'EOF'")
    print("#!/bin/bash")
    print("set -e")
    print("")
    print("# 参数设置")
    print(f"prefix_general=\"benchmark_data/iscale_input/\"")
    print("prefix=\"${prefix_general}MotherImage/\"")
    print("device=\"cuda\"")
    print(f"n_genes={cnts.shape[1]}  # 全部基因")
    print("n_clusters=15")
    print("dist_ST=100")
    print("")
    print("# 预处理图像")
    print("python preprocess.py --prefix=${prefix} --image --outputDir=${prefix}")
    print("")
    print("# 整合数据")
    print("python stitch_locs_cnts_relativeToM.py ${prefix} ${prefix_general}DaughterCaptures/AllignedToMother/D1/")
    print("")
    print("# 选择基因")
    print("python select_genes.py --n-top=${n_genes} \"${prefix}cnts.tsv\" \"${prefix}gene-names.txt\"")
    print("")
    print("# 可视化spots")
    print("python plot_spots.py ${prefix} grayHE_flag=True")
    print("python plot_spots_integrated.py ${prefix} grayHE_flag=True ${dist_ST}")
    print("")
    print("# 提取组织学特征")
    print("python extract_features.py ${prefix} --device=${device}")
    print("")
    print("# 生成组织掩膜")
    print("python get_mask.py ${prefix}embeddings-hist.pickle ${prefix}")
    print("python refine_mask.py --prefix=${prefix}")
    print("python plot_embeddings.py ${prefix}embeddings-hist.pickle ${prefix} --mask=${prefix}mask-small.png")
    print("")
    print("# 训练模型并预测基因表达")
    print("python impute_integrated.py ${prefix} --epochs=1000 --device=${device} --n-states=5 --dist=${dist_ST}")
    print("python refine_gene.py ${prefix} \"conserve_index.pickle\"")
    print("")
    print("# 可视化结果")
    print("python plot_imputed_iSCALE.py ${prefix}")
    print("python merge_imputed.py ${prefix} 1")
    print("")
    print("# 聚类分析")
    print("python cluster_iSCALE.py \\")
    print("    --n-clusters=${n_clusters} \\")
    print("    --filter-size=2 \\")
    print("    --min-cluster-size=20 \\")
    print("    --mask=${prefix}filterRGB/mask-small-refined.png \\")
    print("    --refinedImage=${prefix}filterRGB/conserve_index.pickle \\")
    print("    ${prefix}embeddings-gene.pickle \\")
    print("    ${prefix}iSCALE_output/clusters-gene_${n_clusters}/")
    print("")
    print("# 评估性能")
    print("python evaluate_fit.py ${prefix}")
    print("")
    print("echo \"✅ iSCALE流程完成！\"")
    print("EOF")
    
    print("\n# 3. 运行iSCALE (后台运行)")
    print("chmod +x run_my_xenium.sh")
    print("nohup bash run_my_xenium.sh > iscale_xenium.log 2>&1 &")
    
    print("\n# 4. 查看日志")
    print("tail -f iscale_xenium.log")
    
    print("\n⚠️  重要提示:")
    print("  • 数据已经是单capture，在AllignedToMother中，无需额外对齐")
    print("  • H&E是Xenium原始分辨率(0.2125µm/pixel)，会自动缩放到0.5µm/pixel")
    print("  • 建议使用GPU，训练速度更快")
    print(f"  • 有{cnts.shape[1]}个基因，流程会预测所有基因的表达")
    print("  • 预计运行时间: 数小时到十几小时（取决于硬件）")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ 未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

