#!/usr/bin/env python
"""
将TSV格式的marker文件转换为iSCALE需要的CSV格式
"""

import pandas as pd
import sys

def convert_tsv_to_csv(tsv_file, csv_file=None):
    """
    将TSV格式的marker文件转换为CSV格式
    
    参数:
    - tsv_file: 输入的TSV文件路径
    - csv_file: 输出的CSV文件路径（如果为None，自动生成）
    """
    
    print(f"读取TSV文件: {tsv_file}")
    
    # 读取TSV文件（无表头）
    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['gene', 'label'])
    
    print(f"  ✓ 读取成功: {len(df)} 行")
    print(f"  ✓ 基因数量: {df['gene'].nunique()} 个唯一基因")
    print(f"  ✓ 细胞类型: {df['label'].nunique()} 种")
    
    # 显示细胞类型统计
    print("\n细胞类型统计:")
    type_counts = df['label'].value_counts()
    for cell_type, count in type_counts.items():
        print(f"  • {cell_type}: {count} 个marker基因")
    
    # 生成输出文件名
    if csv_file is None:
        csv_file = tsv_file.replace('.tsv', '.csv')
    
    # 保存为CSV格式
    df.to_csv(csv_file, index=False)
    print(f"\n✓ 转换完成!")
    print(f"  输出文件: {csv_file}")
    print(f"\n文件格式:")
    print(f"  列名: gene,label")
    print(f"  格式: CSV (逗号分隔)")
    
    # 显示前几行示例
    print(f"\n前5行示例:")
    print(df.head().to_string(index=False))
    
    return csv_file


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python convert_markers_to_csv.py <input.tsv> [output.csv]")
        print("\n示例:")
        print("  python convert_markers_to_csv.py brca_markers.tsv")
        print("  python convert_markers_to_csv.py brca_markers.tsv markers.csv")
        sys.exit(1)
    
    tsv_file = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_tsv_to_csv(tsv_file, csv_file)

