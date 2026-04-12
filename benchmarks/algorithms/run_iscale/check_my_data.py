#!/usr/bin/env python
"""
检查你的Xenium pseudo-Visium数据是否适合iSCALE
数据路径：
- H&E: /data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/HE/align_he.png
- ST数据: /data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/SC/
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import json

Image.MAX_IMAGE_PIXELS = None


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


def print_pass(text):
    print(f"{Color.GREEN}✓ {text}{Color.END}")


def print_fail(text):
    print(f"{Color.RED}✗ {text}{Color.END}")


def print_warn(text):
    print(f"{Color.YELLOW}⚠ {text}{Color.END}")


def print_info(text):
    print(f"{Color.BLUE}ℹ {text}{Color.END}")


def check_he_image(he_path):
    """检查H&E图像"""
    print_header("检查 H&E 图像")
    
    if not os.path.exists(he_path):
        print_fail(f"文件不存在: {he_path}")
        return False
    
    print_pass(f"文件存在: {he_path}")
    
    # 获取文件大小
    size_mb = os.path.getsize(he_path) / (1024 * 1024)
    print_info(f"文件大小: {size_mb:.2f} MB")
    
    # 读取图像
    try:
        img = Image.open(he_path)
        width, height = img.size
        mode = img.mode
        
        print_pass(f"图像尺寸: {width} × {height} pixels")
        print_info(f"颜色模式: {mode}")
        
        # 检查颜色模式
        if mode == 'RGB':
            print_pass("颜色模式正确 (RGB)")
        elif mode == 'RGBA':
            print_warn("颜色模式为RGBA，建议转换为RGB")
        else:
            print_warn(f"颜色模式为{mode}，可能需要转换为RGB")
        
        # 估算分辨率
        print_info("\n📏 估算图像分辨率:")
        print_info("假设组织物理尺寸（根据图像推算）:")
        
        for pixel_size in [0.2125, 0.5, 1.0]:
            w_mm = width * pixel_size / 1000
            h_mm = height * pixel_size / 1000
            print_info(f"  {pixel_size:.4f} µm/pixel → {w_mm:.2f}mm × {h_mm:.2f}mm")
        
        # 文件名判断
        filename = os.path.basename(he_path)
        if 'align' in filename.lower():
            print_info("\n💡 文件名包含'align'，这是对齐后的图像")
        
        print_info(f"\n🎯 推荐:")
        print_info(f"  - 如果这是Xenium原始分辨率(0.2125µm/pixel)，命名为 he-raw.png")
        print_info(f"  - 如果已缩放到0.5µm/pixel，命名为 he-scaled.png")
        
        return True, (width, height)
        
    except Exception as e:
        print_fail(f"读取图像失败: {e}")
        return False, None


def check_st_data(st_dir):
    """检查ST数据目录"""
    print_header("检查 ST 数据目录")
    
    if not os.path.exists(st_dir):
        print_fail(f"目录不存在: {st_dir}")
        return False
    
    print_pass(f"目录存在: {st_dir}")
    
    # 列出所有文件
    files = os.listdir(st_dir)
    print_info(f"\n发现 {len(files)} 个文件:")
    for f in sorted(files):
        file_path = os.path.join(st_dir, f)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            size_str = f"{size / (1024*1024):.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"
            print_info(f"  - {f} ({size_str})")
    
    return True


def check_counts_matrix(st_dir):
    """检查基因表达矩阵"""
    print_header("检查基因表达矩阵 (combined_cell_counts.tsv)")
    
    # 先尝试combined_cell_counts.tsv
    counts_file = os.path.join(st_dir, "combined_cell_counts.tsv")
    
    if not os.path.exists(counts_file):
        print_warn("combined_cell_counts.tsv 不是表达矩阵，检查其他文件...")
        
        # 尝试其他可能的文件
        alternatives = ["simulated_data.h5ad", "combined_spatial_count.tsv"]
        for alt in alternatives:
            alt_path = os.path.join(st_dir, alt)
            if os.path.exists(alt_path):
                print_info(f"发现 {alt}")
    
    # 检查h5ad文件
    h5ad_file = os.path.join(st_dir, "simulated_data.h5ad")
    if os.path.exists(h5ad_file):
        print_pass(f"发现 simulated_data.h5ad")
        try:
            import anndata as ad
            adata = ad.read_h5ad(h5ad_file)
            print_pass(f"成功读取 AnnData 对象")
            print_info(f"维度: {adata.shape[0]} spots × {adata.shape[1]} genes")
            print_info(f"Obs列: {list(adata.obs.columns)}")
            print_info(f"前5个基因: {', '.join(adata.var_names[:5].tolist())}")
            
            # 提取表达矩阵
            print_info("\n💡 可以从h5ad提取表达矩阵:")
            print_info("  import anndata as ad")
            print_info("  adata = ad.read_h5ad('simulated_data.h5ad')")
            print_info("  cnts = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)")
            print_info("  cnts.to_csv('combined_counts_matrix.tsv', sep='\\t')")
            
            return True, adata.shape[0], adata.shape[1], adata.obs_names.tolist()
            
        except ImportError:
            print_warn("需要安装anndata: pip install anndata")
            return False, 0, 0, None
        except Exception as e:
            print_fail(f"读取h5ad失败: {e}")
            return False, 0, 0, None
    
    return False, 0, 0, None


def check_locations(st_dir, spot_ids=None):
    """检查位置文件"""
    print_header("检查位置文件 (combined_spot_locations.tsv)")
    
    locs_file = os.path.join(st_dir, "combined_spot_locations.tsv")
    
    if not os.path.exists(locs_file):
        print_fail(f"文件不存在: {locs_file}")
        return False
    
    print_pass(f"文件存在: {locs_file}")
    
    try:
        locs = pd.read_csv(locs_file, sep='\t', index_col=0)
        print_pass(f"成功读取位置文件")
        print_info(f"Spots数量: {locs.shape[0]}")
        print_info(f"列名: {list(locs.columns)}")
        print_info(f"前3行:\n{locs.head(3)}")
        
        # 检查必需的列
        has_x_pixel = 'x_pixel' in locs.columns
        has_y_pixel = 'y_pixel' in locs.columns
        has_x = 'x' in locs.columns
        has_y = 'y' in locs.columns
        
        if has_x_pixel and has_y_pixel:
            print_pass("发现 x_pixel, y_pixel 列（像素坐标）")
            
            x_min, x_max = locs['x_pixel'].min(), locs['x_pixel'].max()
            y_min, y_max = locs['y_pixel'].min(), locs['y_pixel'].max()
            
            print_info(f"X坐标范围: {x_min:.2f} - {x_max:.2f} ({x_max - x_min:.2f} pixels)")
            print_info(f"Y坐标范围: {y_min:.2f} - {y_max:.2f} ({y_max - y_min:.2f} pixels)")
            
            # 估算物理尺寸
            for scale in [0.2125, 0.5]:
                w_mm = (x_max - x_min) * scale / 1000
                h_mm = (y_max - y_min) * scale / 1000
                print_info(f"物理尺寸 ({scale}µm/pixel): {w_mm:.2f}mm × {h_mm:.2f}mm")
        
        elif has_x and has_y:
            print_pass("发现 x, y 列")
            print_info(f"X范围: {locs['x'].min()} - {locs['x'].max()}")
            print_info(f"Y范围: {locs['y'].min()} - {locs['y'].max()}")
            print_warn("x, y 是网格坐标，需要转换为像素坐标")
        
        else:
            print_fail("缺少坐标列！需要 x_pixel, y_pixel 或 x, y")
            return False
        
        # 检查是否需要转换格式
        if has_x_pixel and has_y_pixel:
            print_info("\n🔧 需要转换为iSCALE格式:")
            print_info("  locs_iscale = locs[['x_pixel', 'y_pixel']].copy()")
            print_info("  locs_iscale.columns = ['x', 'y']")
            print_info("  locs_iscale.to_csv('locs.tsv', sep='\\t')")
        
        # 检查spot ID匹配
        if spot_ids is not None:
            locs_ids = set(locs.index.astype(str))
            cnts_ids = set([str(x) for x in spot_ids])
            
            if locs_ids == cnts_ids:
                print_pass("Spot IDs与表达矩阵完全匹配")
            else:
                missing = cnts_ids - locs_ids
                extra = locs_ids - cnts_ids
                if missing:
                    print_fail(f"位置文件中缺少 {len(missing)} 个spots")
                if extra:
                    print_warn(f"位置文件中多出 {len(extra)} 个spots")
        
        return True, locs
        
    except Exception as e:
        print_fail(f"读取位置文件失败: {e}")
        return False, None


def check_spot_info(st_dir):
    """检查spot信息文件"""
    print_header("检查 Spot 信息文件 (combined_spot_info.json)")
    
    info_file = os.path.join(st_dir, "combined_spot_info.json")
    
    if not os.path.exists(info_file):
        print_warn(f"文件不存在（可选文件）: {info_file}")
        return True
    
    print_pass(f"文件存在: {info_file}")
    
    try:
        with open(info_file, 'r') as f:
            spot_info = json.load(f)
        
        print_pass(f"成功读取JSON文件")
        print_info(f"包含 {len(spot_info)} 个spots的信息")
        
        # 查看第一个spot的信息
        first_spot = list(spot_info.keys())[0]
        print_info(f"\n示例 spot {first_spot}:")
        print_info(f"  Keys: {list(spot_info[first_spot].keys())}")
        
        if 'cell_type' in spot_info[first_spot]:
            all_types = set()
            for spot in spot_info.values():
                if 'cell_type' in spot:
                    all_types.update(spot['cell_type'])
            print_info(f"发现 {len(all_types)} 种细胞类型:")
            for ct in sorted(all_types)[:10]:
                print_info(f"    - {ct}")
            if len(all_types) > 10:
                print_info(f"    ... 还有{len(all_types)-10}种")
        
        return True
        
    except Exception as e:
        print_fail(f"读取JSON文件失败: {e}")
        return False


def calculate_radius(st_dir, window_size=55, scale=0.2125):
    """计算spot半径"""
    print_header("计算 Spot 半径")
    
    print_info(f"参数:")
    print_info(f"  Window size: {window_size} µm")
    print_info(f"  Scale: {scale} µm/pixel (Xenium标准)")
    
    # 计算半径
    radius_um = window_size / 2  # 微米
    radius_pixel = radius_um / scale  # 像素
    
    print_info(f"\n计算结果:")
    print_info(f"  Spot直径: {window_size} µm")
    print_info(f"  Spot半径: {radius_um} µm = {radius_pixel:.2f} pixels")
    
    # 如果图像要缩放到0.5 µm/pixel
    target_scale = 0.5
    radius_pixel_scaled = radius_um / target_scale
    
    print_info(f"\n如果图像缩放到 {target_scale} µm/pixel:")
    print_info(f"  Spot半径: {radius_pixel_scaled:.2f} pixels")
    
    return radius_pixel, radius_pixel_scaled


def generate_conversion_script(st_dir, he_path, output_dir, window_size=55, scale=0.2125):
    """生成转换脚本"""
    print_header("生成数据转换命令")
    
    print_info("运行以下命令将数据转换为iSCALE格式:\n")
    
    script = f"""
# 方法1: 使用Python脚本转换
import pandas as pd
import anndata as ad
import shutil
import os

# === 1. 提取并转换表达矩阵 ===
adata = ad.read_h5ad('{st_dir}/simulated_data.h5ad')
cnts = pd.DataFrame(
    adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
    index=adata.obs_names, 
    columns=adata.var_names
)

# === 2. 转换位置文件 ===
locs = pd.read_csv('{st_dir}/combined_spot_locations.tsv', sep='\\t', index_col=0)
locs_iscale = locs[['x_pixel', 'y_pixel']].copy()
locs_iscale.columns = ['x', 'y']

# === 3. 创建iSCALE目录结构 ===
output_base = '{output_dir}'
capture_dir = os.path.join(output_base, 'DaughterCaptures/AllignedToMother/D1')
mother_dir = os.path.join(output_base, 'MotherImage')
os.makedirs(capture_dir, exist_ok=True)
os.makedirs(mother_dir, exist_ok=True)

# === 4. 保存文件 ===
cnts.to_csv(os.path.join(capture_dir, 'cnts.tsv'), sep='\\t')
locs_iscale.to_csv(os.path.join(capture_dir, 'locs.tsv'), sep='\\t')

# === 5. 复制H&E图像 ===
shutil.copy2('{he_path}', os.path.join(mother_dir, 'he-raw.png'))

# === 6. 创建radius文件 ===
radius = {window_size / 2 / scale:.2f}
with open(os.path.join(mother_dir, 'radius-raw.txt'), 'w') as f:
    f.write(f"{{radius:.2f}}\\n")

print("✓ 转换完成！")
print(f"输出目录: {{output_base}}")
"""
    
    print(f"{Color.BOLD}Python脚本内容:{Color.END}")
    print(script)
    
    # 保存脚本
    script_file = "convert_to_iscale.py"
    with open(script_file, 'w') as f:
        f.write(script)
    
    print_pass(f"\n脚本已保存到: {script_file}")
    print_info(f"运行: python {script_file}")


def main():
    print_header("检查 Xenium Pseudo-Visium 数据")
    
    # 你的数据路径
    HE_PATH = "/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/HE/align_he.png"
    ST_DIR = "/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/SC"
    OUTPUT_DIR = "/data1/linxin/Benchmark/iSCALE/iSCALE/Data/my_xenium_data"
    
    all_pass = True
    
    # 1. 检查H&E图像
    result = check_he_image(HE_PATH)
    if not result[0]:
        all_pass = False
    
    # 2. 检查ST数据目录
    if not check_st_data(ST_DIR):
        all_pass = False
    
    # 3. 检查表达矩阵
    result = check_counts_matrix(ST_DIR)
    spot_ids = result[3] if result[0] else None
    if not result[0]:
        all_pass = False
    
    # 4. 检查位置文件
    result = check_locations(ST_DIR, spot_ids)
    if not result[0]:
        all_pass = False
    
    # 5. 检查spot信息
    check_spot_info(ST_DIR)
    
    # 6. 计算半径
    calculate_radius(ST_DIR)
    
    # 7. 生成转换脚本
    generate_conversion_script(ST_DIR, HE_PATH, OUTPUT_DIR)
    
    # 总结
    print_header("检查总结")
    
    if all_pass:
        print_pass("✓ 数据检查通过！")
        print_info("\n📝 下一步:")
        print_info("1. 运行生成的 convert_to_iscale.py 转换数据")
        print_info("2. 运行 iSCALE 流程")
    else:
        print_warn("⚠ 部分检查未通过，请查看上述详情")
    
    print(f"\n{Color.BOLD}数据路径:{Color.END}")
    print(f"  H&E图像: {HE_PATH}")
    print(f"  ST数据: {ST_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

