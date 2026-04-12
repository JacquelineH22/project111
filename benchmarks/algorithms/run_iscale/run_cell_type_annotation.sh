#!/bin/bash
################################################################################
# iSCALE 细胞类型注释脚本
# 
# 使用方法:
#   chmod +x run_cell_type_annotation.sh
#   bash run_cell_type_annotation.sh
################################################################################

set -e

# ==================== 参数配置 ====================

# 输出目录（iSCALE运行后的输出目录）
OUTPUT_BASE="/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/xenium_benchmark_output"
prefix_mother="${OUTPUT_BASE}/MotherImage/"

# Marker文件路径
MARKER_FILE="${prefix_mother}markers.csv"

# 注释输出目录
ANNOTATION_OUTPUT="${prefix_mother}iSCALE_output/annotations/"

# ==================== 检查文件 ====================

echo "=================================================="
echo "iSCALE 细胞类型注释"
echo "=================================================="
echo ""

# 检查marker文件
if [ ! -f "$MARKER_FILE" ]; then
    echo "✗ 错误: 找不到marker文件: $MARKER_FILE"
    echo ""
    echo "请确保:"
    echo "  1. 已将 markers.csv 放在 ${prefix_mother} 目录下"
    echo "  2. 或者运行转换脚本:"
    echo "     python convert_markers_to_csv.py brca_markers.tsv ${MARKER_FILE}"
    exit 1
fi

echo "✓ Marker文件: $MARKER_FILE"

# 检查必要的输出文件
if [ ! -f "${prefix_mother}gene-names.txt" ]; then
    echo "✗ 错误: 找不到 gene-names.txt"
    echo "  请先运行完整的iSCALE流程"
    exit 1
fi

echo "✓ 基因列表文件存在"

# 检查预测结果
if [ ! -d "${prefix_mother}iSCALE_output/super_res_gene_expression/cnts-super-refined/" ]; then
    echo "✗ 错误: 找不到基因表达预测结果"
    echo "  请先运行完整的iSCALE流程（包括impute_integrated.py）"
    exit 1
fi

echo "✓ 基因表达预测结果存在"

# ==================== 运行注释 ====================

echo ""
echo "开始运行细胞类型注释..."
echo "  输入: $MARKER_FILE"
echo "  输出: $ANNOTATION_OUTPUT"
echo ""

python pixannot_percentile.py \
    "${prefix_mother}" \
    "${MARKER_FILE}" \
    "${ANNOTATION_OUTPUT}"

echo ""
echo "=================================================="
echo "✓ 细胞类型注释完成！"
echo "=================================================="
echo ""
echo "📂 输出目录: ${ANNOTATION_OUTPUT}"
echo ""
echo "📊 主要输出文件:"
echo "  • ${ANNOTATION_OUTPUT}labels.png                    (注释标签图)"
echo "  • ${ANNOTATION_OUTPUT}confidence.png                (置信度图)"
echo "  • ${ANNOTATION_OUTPUT}scores/                       (各细胞类型的得分图)"
echo "  • ${ANNOTATION_OUTPUT}threshold001/                 (阈值0.01的注释结果)"
echo "  • ${ANNOTATION_OUTPUT}threshold005/                 (阈值0.05的注释结果)"
echo "  • ${ANNOTATION_OUTPUT}threshold010/                 (阈值0.1的注释结果)"
echo "  • ${ANNOTATION_OUTPUT}masks/                        (各细胞类型的掩膜)"
echo ""
echo "🎉 注释结果已保存到: ${ANNOTATION_OUTPUT}"

