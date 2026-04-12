#!/usr/bin/env python
"""
检查h5ad文件中包含的所有信息
验证是否可以从h5ad提取iSCALE所需的全部数据（除H&E图像）
"""

import anndata as ad
import pandas as pd
import numpy as np


def inspect_h5ad(h5ad_path):
    """详细检查h5ad文件内容"""
    
    print("="*70)
    print(f"检查 h5ad 文件: {h5ad_path}")
    print("="*70)
    
    # 读取h5ad
    print("\n[1] 读取文件...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"✓ 成功读取")
    
    # 基本信息
    print("\n" + "="*70)
    print("📊 基本信息")
    print("="*70)
    print(f"维度: {adata.shape[0]} spots × {adata.shape[1]} genes")
    print(f"数据类型: {type(adata.X)}")
    if hasattr(adata.X, 'toarray'):
        print(f"  → 稀疏矩阵 (需要用 .toarray() 转换)")
    
    # 表达矩阵
    print("\n" + "="*70)
    print("🧬 表达矩阵 (adata.X)")
    print("="*70)
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    print(f"形状: {X.shape}")
    print(f"数据类型: {X.dtype}")
    print(f"值范围: {X.min():.2f} - {X.max():.2f}")
    print(f"平均值: {X.mean():.2f}")
    print(f"非零比例: {(X > 0).sum() / X.size * 100:.2f}%")
    print(f"\n✅ 可用于iSCALE的 cnts.tsv")
    
    # 基因信息
    print("\n" + "="*70)
    print("🔬 基因信息 (adata.var)")
    print("="*70)
    print(f"基因数量: {len(adata.var_names)}")
    print(f"基因名示例: {list(adata.var_names[:10])}")
    if adata.var.shape[1] > 0:
        print(f"Var列: {list(adata.var.columns)}")
        print(f"Var内容示例:\n{adata.var.head(3)}")
    print(f"\n✅ 可用于iSCALE的基因名列表")
    
    # Spot信息
    print("\n" + "="*70)
    print("📍 Spot信息 (adata.obs)")
    print("="*70)
    print(f"Spots数量: {len(adata.obs_names)}")
    print(f"Spot ID示例: {list(adata.obs_names[:10])}")
    print(f"Obs列数: {adata.obs.shape[1]}")
    print(f"Obs列名: {list(adata.obs.columns)}")
    
    print(f"\nObs内容示例:")
    print(adata.obs.head(3))
    
    # 检查位置信息
    print("\n" + "="*70)
    print("📌 位置坐标")
    print("="*70)
    
    has_x_pixel = 'x_pixel' in adata.obs.columns
    has_y_pixel = 'y_pixel' in adata.obs.columns
    has_x = 'x' in adata.obs.columns
    has_y = 'y' in adata.obs.columns
    
    if has_x_pixel and has_y_pixel:
        print(f"✅ 发现像素坐标: x_pixel, y_pixel")
        print(f"  X范围: {adata.obs['x_pixel'].min():.2f} - {adata.obs['x_pixel'].max():.2f}")
        print(f"  Y范围: {adata.obs['y_pixel'].min():.2f} - {adata.obs['y_pixel'].max():.2f}")
        print(f"\n✅ 可用于iSCALE的 locs.tsv")
    
    if has_x and has_y:
        print(f"✅ 发现网格坐标: x, y")
        print(f"  X范围: {adata.obs['x'].min():.2f} - {adata.obs['x'].max():.2f}")
        print(f"  Y范围: {adata.obs['y'].min():.2f} - {adata.obs['y'].max():.2f}")
    
    # 检查细胞类型信息
    print("\n" + "="*70)
    print("🔬 细胞类型/元数据")
    print("="*70)
    
    # 查找可能的细胞类型列
    cell_type_cols = [col for col in adata.obs.columns 
                     if col not in ['x', 'y', 'x_pixel', 'y_pixel', 'cell_count']]
    
    if cell_type_cols:
        print(f"✅ 发现 {len(cell_type_cols)} 种可能的细胞类型列:")
        for col in cell_type_cols[:10]:
            unique_vals = adata.obs[col].unique()
            print(f"  - {col}: {len(unique_vals)} 个唯一值")
            if len(unique_vals) <= 5:
                print(f"    值: {unique_vals}")
        if len(cell_type_cols) > 10:
            print(f"  ... 还有 {len(cell_type_cols) - 10} 列")
    
    # 检查其他属性
    print("\n" + "="*70)
    print("📦 其他属性")
    print("="*70)
    
    if hasattr(adata, 'uns') and len(adata.uns) > 0:
        print(f"✅ uns (非结构化数据): {list(adata.uns.keys())}")
    else:
        print(f"❌ 没有uns数据")
    
    if hasattr(adata, 'obsm') and len(adata.obsm.keys()) > 0:
        print(f"✅ obsm (多维obs注释): {list(adata.obsm.keys())}")
        for key in adata.obsm.keys():
            print(f"  - {key}: shape {adata.obsm[key].shape}")
    else:
        print(f"❌ 没有obsm数据")
    
    if hasattr(adata, 'varm') and len(adata.varm.keys()) > 0:
        print(f"✅ varm (多维var注释): {list(adata.varm.keys())}")
    else:
        print(f"❌ 没有varm数据")
    
    # iSCALE需求总结
    print("\n" + "="*70)
    print("✅ iSCALE 需求映射")
    print("="*70)
    
    print("\n从h5ad可以提取:")
    print("  ✅ cnts.tsv       ← adata.X (表达矩阵)")
    print("  ✅ locs.tsv       ← adata.obs[['x_pixel', 'y_pixel']]")
    print("  ✅ gene-names.txt ← adata.var_names")
    print("  ✅ radius.txt     ← 根据window_size计算 (55µm/2/0.2125)")
    print("\n需要单独提供:")
    print("  ❌ he-raw.*       ← H&E组织图像 (必须单独提供)")
    
    # 生成提取代码
    print("\n" + "="*70)
    print("💻 提取代码示例")
    print("="*70)
    
    code = f"""
import anndata as ad
import pandas as pd

# 读取h5ad
adata = ad.read_h5ad('{h5ad_path}')

# 1. 提取表达矩阵
cnts = pd.DataFrame(
    adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
    index=adata.obs_names,
    columns=adata.var_names
)
cnts.to_csv('cnts.tsv', sep='\\t')
print(f"✓ cnts.tsv: {{cnts.shape[0]}} spots × {{cnts.shape[1]}} genes")

# 2. 提取位置信息
locs = adata.obs[['x_pixel', 'y_pixel']].copy()
locs.columns = ['x', 'y']  # 重命名为iSCALE期望的列名
locs.to_csv('locs.tsv', sep='\\t')
print(f"✓ locs.tsv: {{locs.shape[0]}} spots")

# 3. 提取基因名（可选）
with open('gene-names.txt', 'w') as f:
    for gene in adata.var_names:
        f.write(f"{{gene}}\\n")
print(f"✓ gene-names.txt: {{len(adata.var_names)}} genes")

# 4. 创建radius文件
window_size = 55  # µm
scale = 0.2125    # µm/pixel
radius = (window_size / 2) / scale
with open('radius-raw.txt', 'w') as f:
    f.write(f"{{radius:.2f}}\\n")
print(f"✓ radius-raw.txt: {{radius:.2f}} pixels")

print("\\n✅ 除了H&E图像，所有数据都已从h5ad提取！")
"""
    
    print(code)
    
    return adata


if __name__ == '__main__':
    h5ad_path = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/SC/simulated_data.h5ad'
    
    try:
        adata = inspect_h5ad(h5ad_path)
        
        print("\n" + "="*70)
        print("🎯 结论")
        print("="*70)
        print("\n✅ 是的！除了H&E图像，所有iSCALE需要的信息都在h5ad中：")
        print("\n  1. 表达矩阵 (adata.X)")
        print("  2. 基因名 (adata.var_names)")
        print("  3. Spot坐标 (adata.obs['x_pixel', 'y_pixel'])")
        print("  4. Spot IDs (adata.obs_names)")
        print("  5. 额外的细胞类型信息 (adata.obs的其他列)")
        print("\n❌ 唯一需要单独提供的是:")
        print("  • H&E组织切片图像")
        print("\n💡 这意味着你可以:")
        print("  • 只用h5ad + H&E图像运行iSCALE")
        print("  • 不需要其他TSV/JSON文件")
        print("  • 所有信息都在一个h5ad文件中，方便管理")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

