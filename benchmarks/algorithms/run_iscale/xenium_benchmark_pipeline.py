#!/usr/bin/env python
"""
Xenium Benchmarking Pipeline for iSCALE
模拟论文中的Xenium benchmark实验

步骤：
1. 读取Xenium pseudo-Visium数据（从simulateVisium_new.py生成）
2. 将整张切片切分成多个3.2mm × 3.2mm的daughter captures
3. 保存为iSCALE输入格式（训练用）
4. 保存完整的Xenium数据作为ground truth（评估用）
5. 生成benchmark评估脚本

论文设置：
- 母图：整张Xenium slide
- 子图：5个3.2mm × 3.2mm区域（D1-D5）
- Ground truth：完整Xenium数据
"""

import os
import sys
import pandas as pd
import numpy as np
import anndata as ad
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


class XeniumBenchmarkPipeline:
    
    def __init__(self, h5ad_path, he_path, output_base,
                 capture_size_mm=3.2, pixel_scale=0.2125, 
                 window_size=55, min_spots=200):
        """
        初始化pipeline
        
        参数：
        - h5ad_path: simulated_data.h5ad路径
        - he_path: H&E图像路径
        - output_base: 输出目录
        - capture_size_mm: capture大小（mm）
        - pixel_scale: 像素尺度（µm/pixel）
        - window_size: spot直径（µm）
        - min_spots: 每个capture最少spots数
        """
        self.h5ad_path = h5ad_path
        self.he_path = he_path
        self.output_base = output_base
        self.capture_size_mm = capture_size_mm
        self.pixel_scale = pixel_scale
        self.window_size = window_size
        self.min_spots = min_spots
        
        # 创建输出目录
        self.train_dir = os.path.join(output_base, 'iscale_training_data')
        self.truth_dir = os.path.join(output_base, 'ground_truth')
        self.eval_dir = os.path.join(output_base, 'evaluation')
        
        for d in [self.train_dir, self.truth_dir, self.eval_dir]:
            os.makedirs(d, exist_ok=True)
    
    
    def print_header(self, text):
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70 + "\n")
    
    
    def run(self):
        """运行完整pipeline"""
        
        self.print_header("Xenium Benchmarking Pipeline for iSCALE")
        
        # Step 1: 读取数据
        self.print_header("Step 1: 读取Xenium数据")
        adata = self.load_xenium_data()
        
        # Step 2: 切分captures
        self.print_header("Step 2: 切分Daughter Captures (3.2mm × 3.2mm)")
        captures = self.split_into_captures(adata)
        
        # Step 3: 保存训练数据
        self.print_header("Step 3: 保存iSCALE训练数据")
        self.save_training_data(captures)
        
        # Step 4: 保存ground truth
        self.print_header("Step 4: 保存Ground Truth")
        self.save_ground_truth(adata)
        
        # Step 5: 生成评估脚本
        self.print_header("Step 5: 生成评估脚本")
        self.generate_evaluation_scripts(adata, captures)
        
        # Step 6: 生成可视化
        self.print_header("Step 6: 生成可视化")
        self.visualize_captures(adata, captures)
        
        # 总结
        self.print_summary(adata, captures)
    
    
    def load_xenium_data(self):
        """读取Xenium pseudo-Visium数据"""
        print(f"读取: {self.h5ad_path}")
        adata = ad.read_h5ad(self.h5ad_path)
        
        print(f"  ✓ 数据维度: {adata.shape[0]:,} spots × {adata.shape[1]} genes")
        
        # 检查必需的列
        if 'x_pixel' not in adata.obs.columns or 'y_pixel' not in adata.obs.columns:
            raise ValueError("h5ad.obs中缺少x_pixel或y_pixel列")
        
        # 计算数据范围
        x_span = adata.obs['x_pixel'].max() - adata.obs['x_pixel'].min()
        y_span = adata.obs['y_pixel'].max() - adata.obs['y_pixel'].min()
        x_mm = x_span * self.pixel_scale / 1000
        y_mm = y_span * self.pixel_scale / 1000
        
        print(f"  ✓ 物理尺寸: {x_mm:.2f}mm × {y_mm:.2f}mm")
        
        return adata
    
    
    def split_into_captures(self, adata):
        """将数据切分成多个captures"""
        
        locs = adata.obs[['x_pixel', 'y_pixel']].copy()
        
        # 计算网格
        capture_size_pixels = self.capture_size_mm * 1000 / self.pixel_scale
        
        x_min, x_max = locs['x_pixel'].min(), locs['x_pixel'].max()
        y_min, y_max = locs['y_pixel'].min(), locs['y_pixel'].max()
        
        n_x = int((x_max - x_min) / capture_size_pixels)
        n_y = int((y_max - y_min) / capture_size_pixels)
        
        print(f"可以切分成: {n_x} × {n_y} = {n_x * n_y} 个captures")
        
        if n_x * n_y == 0:
            print(f"⚠ 警告: 组织太小，无法切分成{self.capture_size_mm}mm的captures")
            print(f"  建议: 使用完整数据或减小capture_size_mm")
            sys.exit(1)
        
        # 切分数据
        captures = {}
        capture_id = 1
        
        for i in range(n_x):
            for j in range(n_y):
                # 计算边界
                cap_x_min = x_min + i * capture_size_pixels
                cap_x_max = cap_x_min + capture_size_pixels
                cap_y_min = y_min + j * capture_size_pixels
                cap_y_max = cap_y_min + capture_size_pixels
                
                # 选择spots
                mask = (
                    (locs['x_pixel'] >= cap_x_min) & (locs['x_pixel'] < cap_x_max) &
                    (locs['y_pixel'] >= cap_y_min) & (locs['y_pixel'] < cap_y_max)
                )
                
                n_spots = mask.sum()
                
                if n_spots < self.min_spots:
                    print(f"  ⊗ D{capture_id} ({i},{j}): {n_spots} spots < {self.min_spots}, 跳过")
                    continue
                
                # 提取数据
                adata_cap = adata[mask].copy()
                
                capture_name = f"D{capture_id}"
                captures[capture_name] = {
                    'adata': adata_cap,
                    'grid_pos': (i, j),
                    'bbox': (cap_x_min, cap_x_max, cap_y_min, cap_y_max),
                    'n_spots': n_spots
                }
                
                print(f"  ✓ {capture_name} ({i},{j}): {n_spots:,} spots, "
                      f"[{cap_x_min:.0f},{cap_x_max:.0f}] × [{cap_y_min:.0f},{cap_y_max:.0f}]")
                
                capture_id += 1
        
        print(f"\n总共生成 {len(captures)} 个有效的captures")
        
        return captures
    
    
    def save_training_data(self, captures):
        """保存iSCALE训练数据"""
        
        mother_dir = os.path.join(self.train_dir, 'MotherImage')
        os.makedirs(mother_dir, exist_ok=True)
        
        # 保存每个capture
        for name, cap in captures.items():
            cap_dir = os.path.join(self.train_dir, f'DaughterCaptures/AllignedToMother/{name}')
            os.makedirs(cap_dir, exist_ok=True)
            
            adata = cap['adata']
            
            # 提取表达矩阵
            if hasattr(adata.X, 'toarray'):
                X = adata.X.toarray()
            else:
                X = adata.X
            
            cnts = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
            locs = adata.obs[['x_pixel', 'y_pixel']].copy()
            locs.columns = ['x', 'y']
            
            # 保存
            cnts.to_csv(os.path.join(cap_dir, 'cnts.tsv'), sep='\t')
            locs.to_csv(os.path.join(cap_dir, 'locs.tsv'), sep='\t')
            
            print(f"  ✓ {name}: {cap_dir}")
        
        # 复制H&E图像
        he_dst = os.path.join(mother_dir, 'he-raw.png')
        shutil.copy2(self.he_path, he_dst)
        print(f"  ✓ H&E图像: {he_dst}")
        
        # 创建radius文件
        radius_raw = (self.window_size / 2) / self.pixel_scale
        with open(os.path.join(mother_dir, 'radius-raw.txt'), 'w') as f:
            f.write(f"{radius_raw:.2f}\n")
        print(f"  ✓ radius-raw.txt: {radius_raw:.2f} pixels")
    
    
    def save_ground_truth(self, adata):
        """保存完整的Xenium数据作为ground truth"""
        
        # 保存完整的h5ad
        truth_h5ad = os.path.join(self.truth_dir, 'xenium_ground_truth.h5ad')
        adata.write(truth_h5ad)
        print(f"  ✓ Ground truth h5ad: {truth_h5ad}")
        
        # 保存表达矩阵
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X
        
        cnts = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
        locs = adata.obs[['x_pixel', 'y_pixel']].copy()
        locs.columns = ['x', 'y']
        
        cnts.to_csv(os.path.join(self.truth_dir, 'cnts_truth.tsv'), sep='\t')
        locs.to_csv(os.path.join(self.truth_dir, 'locs_truth.tsv'), sep='\t')
        print(f"  ✓ Truth TSV files saved")
        
        # 保存统计信息
        stats = {
            'n_spots': adata.shape[0],
            'n_genes': adata.shape[1],
            'genes': adata.var_names.tolist(),
            'pixel_scale': self.pixel_scale,
            'window_size': self.window_size
        }
        
        with open(os.path.join(self.truth_dir, 'metadata.pickle'), 'wb') as f:
            pickle.dump(stats, f)
        print(f"  ✓ Metadata saved")
    
    
    def generate_evaluation_scripts(self, adata, captures):
        """生成评估脚本"""
        
        # Python评估脚本
        eval_script = f"""#!/usr/bin/env python
'''
iSCALE Xenium Benchmark 评估脚本

比较iSCALE预测结果与Xenium ground truth
'''

import pandas as pd
import numpy as np
import pickle
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 路径
TRUTH_DIR = '{self.truth_dir}'
ISCALE_OUTPUT = 'path/to/iscale/output/MotherImage/iSCALE_output/super_res_gene_expression/cnts-super-refined/'

def load_ground_truth():
    '''加载ground truth'''
    cnts = pd.read_csv(f'{{TRUTH_DIR}}/cnts_truth.tsv', sep='\\t', index_col=0)
    locs = pd.read_csv(f'{{TRUTH_DIR}}/locs_truth.tsv', sep='\\t', index_col=0)
    
    with open(f'{{TRUTH_DIR}}/metadata.pickle', 'rb') as f:
        metadata = pickle.load(f)
    
    return cnts, locs, metadata


def load_iscale_predictions():
    '''加载iSCALE预测结果'''
    import os
    predictions = {{}}
    
    for gene_file in os.listdir(ISCALE_OUTPUT):
        if gene_file.endswith('.pickle'):
            gene = gene_file.replace('.pickle', '')
            with open(os.path.join(ISCALE_OUTPUT, gene_file), 'rb') as f:
                predictions[gene] = pickle.load(f)
    
    return predictions


def evaluate_gene(truth_values, pred_matrix, locs, pixel_scale):
    '''评估单个基因的预测性能'''
    
    # 将truth的spot坐标映射到预测矩阵的像素
    x_coords = (locs['x'] / pixel_scale).astype(int)
    y_coords = (locs['y'] / pixel_scale).astype(int)
    
    # 提取对应位置的预测值
    pred_values = []
    for x, y in zip(x_coords, y_coords):
        if 0 <= y < pred_matrix.shape[0] and 0 <= x < pred_matrix.shape[1]:
            pred_values.append(pred_matrix[y, x])
        else:
            pred_values.append(np.nan)
    
    pred_values = np.array(pred_values)
    
    # 过滤NaN
    valid = ~np.isnan(pred_values) & ~np.isnan(truth_values)
    truth_valid = truth_values[valid]
    pred_valid = pred_values[valid]
    
    if len(truth_valid) < 10:
        return None
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(truth_valid, pred_valid))
    pearson_r, pearson_p = pearsonr(truth_valid, pred_valid)
    spearman_r, spearman_p = spearmanr(truth_valid, pred_valid)
    
    return {{
        'rmse': rmse,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_spots': len(truth_valid)
    }}


def main():
    print("加载数据...")
    cnts_truth, locs_truth, metadata = load_ground_truth()
    predictions = load_iscale_predictions()
    
    print(f"Ground truth: {{cnts_truth.shape[0]}} spots × {{cnts_truth.shape[1]}} genes")
    print(f"Predictions: {{len(predictions)}} genes")
    
    # 评估每个基因
    results = {{}}
    for gene in cnts_truth.columns:
        if gene not in predictions:
            print(f"跳过 {{gene}}: 没有预测")
            continue
        
        metrics = evaluate_gene(
            cnts_truth[gene].values,
            predictions[gene],
            locs_truth,
            metadata['pixel_scale']
        )
        
        if metrics:
            results[gene] = metrics
            print(f"{{gene}}: Pearson={{metrics['pearson_r']:.3f}}, "
                  f"RMSE={{metrics['rmse']:.3f}}")
    
    # 汇总统计
    print("\\n" + "="*50)
    print("总体性能:")
    print("="*50)
    
    all_pearson = [r['pearson_r'] for r in results.values()]
    all_rmse = [r['rmse'] for r in results.values()]
    
    print(f"平均 Pearson R: {{np.mean(all_pearson):.3f}} ± {{np.std(all_pearson):.3f}}")
    print(f"平均 RMSE: {{np.mean(all_rmse):.3f}} ± {{np.std(all_rmse):.3f}}")
    print(f"评估基因数: {{len(results)}}")
    
    # 保存结果
    results_df = pd.DataFrame(results).T
    results_df.to_csv('evaluation_results.csv')
    print(f"\\n结果已保存到: evaluation_results.csv")


if __name__ == '__main__':
    main()
"""
        
        script_path = os.path.join(self.eval_dir, 'evaluate_iscale_predictions.py')
        with open(script_path, 'w') as f:
            f.write(eval_script)
        
        print(f"  ✓ 评估脚本: {script_path}")
        
        # 生成README
        readme = f"""# iSCALE Xenium Benchmark 评估

## 数据信息

- **Ground Truth**: {adata.shape[0]:,} spots × {adata.shape[1]} genes
- **Training Captures**: {len(captures)} 个 3.2mm×3.2mm 区域
- **物理尺寸**: {(adata.obs['x_pixel'].max() - adata.obs['x_pixel'].min()) * self.pixel_scale / 1000:.2f}mm × {(adata.obs['y_pixel'].max() - adata.obs['y_pixel'].min()) * self.pixel_scale / 1000:.2f}mm

## 运行iSCALE

```bash
cd /path/to/iSCALE

# 修改运行脚本
nano run_iscale_xenium.sh
# 设置: INPUT_BASE="{self.train_dir}"

# 运行
bash run_iscale_xenium.sh
```

## 评估预测结果

```bash
cd {self.eval_dir}

# 编辑评估脚本，设置ISCALE_OUTPUT路径
nano evaluate_iscale_predictions.py

# 运行评估
python evaluate_iscale_predictions.py
```

## 目录结构

```
{self.output_base}/
├── iscale_training_data/        # iSCALE训练数据
│   ├── DaughterCaptures/
│   │   └── AllignedToMother/
│   │       ├── D1/
│   │       ├── D2/
│   │       └── ...
│   └── MotherImage/
│       ├── he-raw.png
│       └── radius-raw.txt
│
├── ground_truth/                # Xenium ground truth
│   ├── xenium_ground_truth.h5ad
│   ├── cnts_truth.tsv
│   ├── locs_truth.tsv
│   └── metadata.pickle
│
└── evaluation/                  # 评估脚本和结果
    ├── evaluate_iscale_predictions.py
    ├── captures_grid.png
    └── README.md
```
"""
        
        with open(os.path.join(self.eval_dir, 'README.md'), 'w') as f:
            f.write(readme)
        
        print(f"  ✓ README: {self.eval_dir}/README.md")
    
    
    def visualize_captures(self, adata, captures):
        """可视化capture网格"""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        locs = adata.obs[['x_pixel', 'y_pixel']]
        
        # 绘制所有spots
        ax.scatter(locs['x_pixel'], locs['y_pixel'], 
                  c='lightgray', s=1, alpha=0.5, label='All spots')
        
        # 绘制每个capture
        colors = plt.cm.tab10(np.linspace(0, 1, len(captures)))
        
        for i, (name, cap) in enumerate(captures.items()):
            x_min, x_max, y_min, y_max = cap['bbox']
            
            # 矩形框
            rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                linewidth=2, edgecolor=colors[i], 
                                facecolor=colors[i], alpha=0.2)
            ax.add_patch(rect)
            
            # 标签
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            ax.text(cx, cy, f'{name}\\n{cap["n_spots"]} spots', 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.set_title('Daughter Captures Grid Layout (3.2mm × 3.2mm)', 
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        output_path = os.path.join(self.eval_dir, 'captures_grid.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 网格可视化: {output_path}")
    
    
    def print_summary(self, adata, captures):
        """打印总结"""
        
        self.print_header("✅ Benchmark Pipeline 完成！")
        
        print("📊 数据统计:")
        print(f"  • Ground Truth: {adata.shape[0]:,} spots × {adata.shape[1]} genes")
        print(f"  • Training Captures: {len(captures)} 个")
        for name, cap in captures.items():
            print(f"    - {name}: {cap['n_spots']:,} spots")
        
        total_training_spots = sum(cap['n_spots'] for cap in captures.values())
        coverage = total_training_spots / adata.shape[0] * 100
        print(f"  • 训练数据覆盖率: {coverage:.1f}%")
        
        print(f"\\n📂 输出目录:")
        print(f"  • 训练数据: {self.train_dir}")
        print(f"  • Ground Truth: {self.truth_dir}")
        print(f"  • 评估脚本: {self.eval_dir}")
        
        print(f"\\n🎯 下一步:")
        print(f"1. 运行iSCALE训练:")
        print(f"   cd /path/to/iSCALE")
        print(f"   # 修改输入路径为: {self.train_dir}")
        print(f"   bash run_iscale_xenium.sh")
        print(f"")
        print(f"2. 评估预测结果:")
        print(f"   cd {self.eval_dir}")
        print(f"   python evaluate_iscale_predictions.py")
        print(f"")
        print(f"3. 查看可视化:")
        print(f"   {self.eval_dir}/captures_grid.png")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Xenium Benchmarking Pipeline for iSCALE",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--h5ad', type=str, required=True,
                       help='simulated_data.h5ad路径')
    parser.add_argument('--he_image', type=str, required=True,
                       help='H&E图像路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--capture_size', type=float, default=3.2,
                       help='Capture大小（mm），默认3.2')
    parser.add_argument('--pixel_scale', type=float, default=0.2125,
                       help='像素尺度（µm/pixel），默认0.2125')
    parser.add_argument('--window_size', type=float, default=55,
                       help='Spot直径（µm），默认55')
    parser.add_argument('--min_spots', type=int, default=200,
                       help='每个capture最少spots数，默认200')
    
    args = parser.parse_args()
    
    # 运行pipeline
    pipeline = XeniumBenchmarkPipeline(
        h5ad_path=args.h5ad,
        he_path=args.he_image,
        output_base=args.output,
        capture_size_mm=args.capture_size,
        pixel_scale=args.pixel_scale,
        window_size=args.window_size,
        min_spots=args.min_spots
    )
    
    pipeline.run()


if __name__ == '__main__':
    main()

