#!/bin/bash
################################################################################
# iSCALE运行脚本 - Xenium Pseudo-Visium数据（修复版）
# 
# 修复内容：
# 1. 确保使用所有Daughter Captures（D1, D2, ...）
# 2. 正确生成radius.txt（整数格式）
# 3. 自动检测可用的captures
#
# 输入数据: benchmark_data/xenium_benchmark_input/iscale_training_data/
# 输出数据: benchmark_data/xenium_benchmark_output/
#
# 使用方法:
#   chmod +x run_iscale_xenium.sh
#   CUDA_VISIBLE_DEVICES=2 nohup bash run_iscale_xenium.sh > iscale_run.log 2>&1 &
#   tail -f iscale_run.log
################################################################################

set -e  # 遇到错误立即退出

# ==================== 参数配置 ====================

# 数据目录（修改为你的实际路径）
INPUT_BASE="/data1/hounaiqiao/wzr/Simulated_Xenium/brca_rep1/w55/iscale_input"
OUTPUT_BASE="/data1/hounaiqiao/wzr/Simulated_Xenium/brca_rep1/w55/iscale_output"

# iSCALE参数
DEVICE="cuda"           # "cuda" 或 "cpu"
N_GENES=313             # 基因数量（你的数据有313个基因）
N_CLUSTERS=15           # 聚类数量
DIST_ST=100             # ST样本间的平滑参数
N_STATES=5              # 训练状态数量（集成预测）
EPOCHS=1000             # 训练轮数

# Xenium参数
PIXEL_SIZE_RAW=0.2125   # 原始像素大小 (µm/pixel)
PIXEL_SIZE=0.5          # 目标像素大小 (µm/pixel)

# 线程设置
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

# ==================== 函数定义 ====================

print_section() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
    echo ""
}

print_step() {
    echo ""
    echo "[$1] $2"
    echo "----------------------------------------"
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "✗ 错误: 找不到文件 $1"
        exit 1
    fi
    echo "✓ 文件存在: $1"
}

check_dir() {
    if [ ! -d "$1" ]; then
        echo "✗ 错误: 找不到目录 $1"
        exit 1
    fi
    echo "✓ 目录存在: $1"
}

# ==================== 检查环境 ====================

print_section "检查运行环境"

echo "当前工作目录: $(pwd)"
echo "Python: $(which python)"
echo "Python版本: $(python --version)"

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "iSCALE_env" ]]; then
    echo "⚠ 警告: 当前不在iSCALE_env环境中"
    echo "  当前环境: $CONDA_DEFAULT_ENV"
    echo "  建议运行: conda activate iSCALE_env"
fi

# 检查输入目录和文件
echo ""
echo "检查输入数据..."
check_dir "$INPUT_BASE"
check_dir "$INPUT_BASE/MotherImage"
check_dir "$INPUT_BASE/DaughterCaptures/AllignedToMother/D1"
check_file "$INPUT_BASE/DaughterCaptures/AllignedToMother/D1/cnts.tsv"
check_file "$INPUT_BASE/DaughterCaptures/AllignedToMother/D1/locs.tsv"

# 查找H&E图像
HE_FILE=$(find "$INPUT_BASE/MotherImage" -name "he-raw.*" -o -name "he-scaled.*" | head -1)
if [ -z "$HE_FILE" ]; then
    echo "✗ 错误: 找不到H&E图像文件"
    exit 1
fi
echo "✓ H&E图像: $HE_FILE"

# 创建输出目录
echo ""
echo "创建输出目录..."
mkdir -p "$OUTPUT_BASE/MotherImage"
echo "✓ 输出目录: $OUTPUT_BASE"

# ==================== 开始处理 ====================

print_section "开始运行 iSCALE 流程"

START_TIME=$(date +%s)
echo "开始时间: $(date)"

# 设置路径变量
prefix_mother="${OUTPUT_BASE}/MotherImage/"
input_mother="${INPUT_BASE}/MotherImage/"
input_capture="${INPUT_BASE}/DaughterCaptures/AllignedToMother/D1/"

# ==================== Step 1: 复制和预处理图像 ====================

print_step "1/10" "图像预处理"

# 复制H&E图像到输出目录
echo "复制H&E图像..."
cp "$INPUT_BASE/MotherImage/he-raw."* "${OUTPUT_BASE}/MotherImage/" 2>/dev/null || true

# 复制radius文件
if [ -f "$INPUT_BASE/MotherImage/radius-raw.txt" ]; then
    cp "$INPUT_BASE/MotherImage/radius-raw.txt" "${OUTPUT_BASE}/MotherImage/"
    echo "✓ 复制radius-raw.txt"
fi

# 预处理图像（缩放和padding）
echo "预处理H&E图像..."
python preprocess.py \
    --prefix="${prefix_mother}" \
    --image \
    --outputDir="${prefix_mother}"

# 生成radius.txt（缩放后的半径，必须是整数）
echo "生成radius.txt..."
if [ -f "${OUTPUT_BASE}/MotherImage/radius-raw.txt" ]; then
    # 从radius-raw.txt读取原始半径
    RADIUS_RAW=$(cat "${OUTPUT_BASE}/MotherImage/radius-raw.txt")
    # 计算缩放后的半径: radius_scaled = radius_raw * (pixel_size_raw / pixel_size)
    # 注意：必须输出整数，plot_spots.py用int()读取
    RADIUS_SCALED=$(python -c "print(int(${RADIUS_RAW} * ${PIXEL_SIZE_RAW} / ${PIXEL_SIZE} + 0.5))")
    echo "${RADIUS_SCALED}" > "${OUTPUT_BASE}/MotherImage/radius.txt"
    echo "✓ 生成radius.txt: ${RADIUS_SCALED} pixels (原始: ${RADIUS_RAW} pixels)"
else
    # 如果没有radius-raw.txt，直接计算
    RADIUS_SCALED=$(python -c "print(int((55/2) / ${PIXEL_SIZE} + 0.5))")
    echo "${RADIUS_SCALED}" > "${OUTPUT_BASE}/MotherImage/radius.txt"
    echo "✓ 生成radius.txt: ${RADIUS_SCALED} pixels"
fi

echo "✓ 图像预处理完成"

# ==================== Step 2: 整合所有Daughter Captures ====================

print_step "2/10" "整合所有Daughter Captures数据"

# 自动检测所有可用的captures
echo "检测可用的Daughter Captures:"
CAPTURE_DIRS=(${INPUT_BASE}/DaughterCaptures/AllignedToMother/D*)
CAPTURE_PATHS=()

for cap_dir in "${CAPTURE_DIRS[@]}"; do
    if [ -d "$cap_dir" ]; then
        cap_name=$(basename $cap_dir)
        n_spots=$(($(wc -l < ${cap_dir}/locs.tsv) - 1))
        echo "  ✓ ${cap_name}: ${n_spots} spots"
        CAPTURE_PATHS+=("${cap_dir}/")
    fi
done

echo ""
echo "整合 ${#CAPTURE_PATHS[@]} 个captures..."

# 使用stitch命令整合所有captures
python stitch_locs_cnts_relativeToM.py \
    "${prefix_mother}" \
    "${CAPTURE_PATHS[@]}"

# 验证整合结果
n_total_spots=$(($(wc -l < ${prefix_mother}locs.tsv) - 1))
echo "✓ 整合后总spots数: ${n_total_spots}"

# ==================== Step 3: 选择基因 ====================

print_step "3/10" "选择高变异基因"

python select_genes.py \
    --n-top=${N_GENES} \
    "${prefix_mother}cnts.tsv" \
    "${prefix_mother}gene-names.txt"

echo "✓ 选择了 ${N_GENES} 个基因"

# ==================== Step 4: 可视化Spots ====================

print_step "4/10" "可视化Spot级别表达"

echo "绘制spots..."
python plot_spots.py \
    "${prefix_mother}" \
    grayHE_flag=True

echo "绘制整合后的spots..."
python plot_spots_integrated.py \
    "${prefix_mother}" \
    grayHE_flag=True \
    ${DIST_ST}

echo "✓ Spots可视化完成"

# ==================== Step 5: 提取组织学特征 ====================

print_step "5/10" "提取组织学特征（H&E）"

echo "使用HIPT模型提取特征..."
python extract_features.py \
    "${prefix_mother}" \
    --device=${DEVICE}

echo "✓ 特征提取完成"

# ==================== Step 6: 生成组织掩膜 ====================

print_step "6/10" "生成组织掩膜"

echo "自动检测组织区域..."
python get_mask.py \
    "${prefix_mother}embeddings-hist.pickle" \
    "${prefix_mother}"

echo "优化掩膜..."
python refine_mask.py \
    --prefix="${prefix_mother}"

echo "可视化embeddings..."
python plot_embeddings.py \
    "${prefix_mother}embeddings-hist.pickle" \
    "${prefix_mother}" \
    --mask="${prefix_mother}mask-small.png"

echo "✓ 组织掩膜生成完成"

# ==================== Step 7: 训练模型并预测 ====================

print_step "7/10" "训练基因表达预测模型"

echo "训练模型并预测超分辨率基因表达..."
echo "  训练轮数: ${EPOCHS}"
echo "  状态数量: ${N_STATES}"
echo "  设备: ${DEVICE}"

python impute_integrated.py \
    "${prefix_mother}" \
    --epochs=${EPOCHS} \
    --device=${DEVICE} \
    --n-states=${N_STATES} \
    --dist=${DIST_ST}

echo "优化基因表达预测..."
python refine_gene.py \
    "${prefix_mother}" \
    "conserve_index.pickle"

echo "✓ 模型训练和预测完成"

# ==================== Step 8: 可视化结果 ====================

print_step "8/10" "可视化预测结果"

echo "绘制预测的基因表达图..."
python plot_imputed_iSCALE.py \
    "${prefix_mother}"

echo "合并预测结果..."
python merge_imputed.py \
    "${prefix_mother}" \
    1

echo "✓ 结果可视化完成"

# ==================== Step 9: 聚类分析 ====================

print_step "9/10" "基于基因表达的聚类分析"

echo "运行聚类算法..."
echo "  聚类数量: ${N_CLUSTERS}"

python cluster_iSCALE.py \
    --n-clusters=${N_CLUSTERS} \
    --filter-size=2 \
    --min-cluster-size=20 \
    --mask="${prefix_mother}filterRGB/mask-small-refined.png" \
    --refinedImage="${prefix_mother}filterRGB/conserve_index.pickle" \
    "${prefix_mother}embeddings-gene.pickle" \
    "${prefix_mother}iSCALE_output/clusters-gene_${N_CLUSTERS}/"

echo "✓ 聚类分析完成"

# ==================== Step 10: 评估性能 ====================

print_step "10/10" "评估模型性能"

echo "计算训练集RMSE和Pearson相关..."
python evaluate_fit.py \
    "${prefix_mother}"

echo "✓ 性能评估完成"

# ==================== 完成 ====================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

print_section "✅ iSCALE 流程完成！"

echo "结束时间: $(date)"
echo "运行时长: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""
echo "📂 输出目录: ${OUTPUT_BASE}"
echo ""
echo "📊 主要输出文件:"
echo "  • ${prefix_mother}he.tiff                                     (预处理后的H&E图像)"
echo "  • ${prefix_mother}embeddings-hist.pickle                      (组织学特征)"
echo "  • ${prefix_mother}iSCALE_output/super_res_gene_expression/    (超分辨率基因表达)"
echo "  • ${prefix_mother}iSCALE_output/super_res_ST_plots/           (可视化结果)"
echo "  • ${prefix_mother}iSCALE_output/clusters-gene_${N_CLUSTERS}/  (聚类结果)"
echo ""
echo "📝 查看结果:"
echo "  cd ${prefix_mother}iSCALE_output"
echo "  ls -lh super_res_ST_plots/cnts-super-plots-refined/"
echo ""
echo "🎉 所有结果已保存到: ${OUTPUT_BASE}"

