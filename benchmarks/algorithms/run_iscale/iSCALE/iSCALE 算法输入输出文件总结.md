 iSCALE 算法输入输出文件总结

 一、必需输入文件

 1. 母图像（MotherImage）文件夹

 1.1 原始组织学图像
 文件名: `heraw.` (支持多种格式)
 支持格式: `.tiff`, `.tif`, `.svs`, `.ome.tif`, `.ome.tiff`, `.jpg`, `.png`, `.ndpi`, `.scn`, `.mrxs`
 内容: 原始大尺寸组织切片图像（H&E染色）
 要求:
   图像必须覆盖整个组织区域
   图像质量应足够高以支持特征提取
   建议使用TIFF格式以保持图像质量
 说明: 
   算法会通过 `preprocess.py` 处理此图像生成最终的 `he.tiff` 文件
   如果图像已经缩放到目标分辨率，可以命名为 `hescaled.` 并跳过缩放步骤

 1.2 Spot半径文件
 文件名: `radiusraw.txt`
 格式: 纯文本文件，包含单个数值
 内容: 在原始图像中，每个spot的半径（单位：像素数）
 计算方式（对于Visium数据）:
  ```
  radius_raw = spot_diameter_fullres  0.5
  ```
  其中 `spot_diameter_fullres` 来自Visium的 `scalefactors_json.json` 文件
 示例内容:
  ```
  55
  ```
 说明: 
   如果图像经过缩放，可以使用 `rescale_locs.py` 自动生成缩放后的 `radius.txt`
   `radius.txt` 是缩放后的spot半径，用于与预处理后的图像对齐

 2. 子捕获（DaughterCaptures）文件夹

iSCALE 需要多个空间转录组学（ST）子捕获数据，这些数据将被整合到母图像上。

 2.1 未对齐的原始数据（UnallignedToMother）

目录结构:
```
DaughterCaptures/
└── UnallignedToMother/
    ├── D1/
    │   ├── cnts.tsv            基因表达计数矩阵
    │   ├── locs.tsv            spot坐标
    │   └── he.                H&E图像
    ├── D2/
    └── ...
```

 2.1.1 基因表达计数矩阵
 文件名: `cnts.tsv`（每个子捕获文件夹中）
 格式: TSV（TabSeparated Values）文本文件
 结构:
   第1行: 基因名称（列名）
   第2行及以后: 每个spot的基因表达计数
   第1列: Spot ID（必须与 `locs.tsv` 中的Spot ID匹配）
   第2列及以后: 每个基因的表达计数值
 矩阵方向: 基因 × spots（行=基因，列=spots）
 示例格式:
  ```
  spot_id    GENE1    GENE2    GENE3    ...
  spot_001   10       25       5        ...
  spot_002   15       30       8        ...
  ...
  ```
 要求:
   Spot ID必须与对应 `locs.tsv` 中的Spot ID完全匹配
   基因表达值应为非负整数或浮点数
   支持多种ST平台：Visium, Visium HD, Xenium, CosMx等

 2.1.2 Spot位置坐标文件
 文件名: `locs.tsv`（每个子捕获文件夹中）
 格式: TSV（TabSeparated Values）文本文件
 结构:
   第1行: 表头（Header），必须包含 `spot`, `x`, `y` 列
   第2行及以后: 每个spot的坐标信息
   第1列: Spot ID（必须与 `cnts.tsv` 中的Spot ID匹配）
   第2列: x坐标（水平轴）
   第3列: y坐标（垂直轴）
 坐标系统要求:
   坐标必须与对应子捕获的H&E图像在同一空间坐标系中
   坐标值应为像素单位
 示例格式:
  ```
  spot      x      y
  spot_001  100    200
  spot_002  150    250
  ...
  ```

 2.1.3 子捕获H&E图像
 文件名: `he.`（每个子捕获文件夹中）
 格式: 支持多种图像格式（JPG, PNG, TIFF等）
 内容: 与子捕获ST数据对应的H&E染色图像
 要求:
   图像必须与 `locs.tsv` 中的坐标在同一空间坐标系中
   用于与母图像进行对齐（registration）

 2.2 对齐后的数据（AllignedToMother）

目录结构:
```
DaughterCaptures/
└── AllignedToMother/
    ├── D1/
    │   ├── cnts.tsv            对齐后的基因表达计数矩阵
    │   └── locs.tsv            对齐后的spot坐标（相对于母图像）
    ├── D2/
    └── ...
```

说明:
 此文件夹中的数据需要通过半自动对齐工具（`Alignment_scripts/AlignmentMethod.ipynb`）生成
 对齐后的坐标是相对于母图像的坐标系
 对齐后的数据将被 `stitch_locs_cnts_relativeToM.py` 整合到母图像上

 2.2.1 对齐后的基因表达矩阵
 文件名: `cnts.tsv`（每个对齐后的子捕获文件夹中）
 格式: 与未对齐版本相同
 内容: 对齐后的基因表达计数矩阵
 要求: Spot ID和基因表达值保持不变，但坐标已更新

 2.2.2 对齐后的坐标文件
 文件名: `locs.tsv`（每个对齐后的子捕获文件夹中）
 格式: 与未对齐版本相同
 内容: 对齐后的spot坐标，现在相对于母图像坐标系
 要求: 
   坐标必须与母图像（`heraw.`）在同一空间坐标系中
   所有子捕获的spots将被整合到同一个坐标系中

 二、可选输入文件

 1. 用户定义的基因列表（可选）
 文件名: `genenames.txt`（位于MotherImage文件夹）
 格式: 纯文本文件，每行一个基因名称
 内容: 用户希望预测的基因列表
 要求:
   基因名称必须与 `cnts.tsv` 中的列名匹配
   每行一个基因名称，无其他格式要求
 说明:
   如果不提供，算法会使用 `select_genes.py` 自动选择高变异基因
   默认选择指定数量的变异度最高的基因（通过 `ntop` 参数设置，默认100）
   对于Visium等大基因面板平台，建议使用更高的值（如3000）
 示例内容:
  ```
  GENE1
  GENE2
  GENE3
  ...
  ```

 2. 标记基因文件（可选）
 文件名: `markers.csv`（位于MotherImage文件夹）
 格式: CSV文件
 内容: 用于自动注释的标记基因列表
 结构:
  ```
  gene,label
  MKI67,Tumor
  KRT20,Mucosa
  ...
  ```
 说明: 
   如果提供此文件，算法会使用 `pixannot_percentile.py` 进行细胞类型/区域注释
   注释结果保存在 `iSCALE_output/annotations/` 文件夹中

 3. 缩放后的图像（可选）
 文件名: `hescaled.`（位于MotherImage文件夹）
 格式: 支持多种图像格式
 内容: 已经缩放到目标分辨率的H&E图像
 说明:
   如果提供此文件，可以跳过 `rescale_img.py` 步骤
   目标分辨率由 `pixel_size` 参数指定（默认0.5微米/像素）

 4. 缩放后的spot半径（可选）
 文件名: `radius.txt`（位于MotherImage文件夹）
 格式: 纯文本文件，包含单个数值
 内容: 在缩放后的图像中，每个spot的半径（单位：像素数）
 说明:
   如果未提供，可以使用 `rescale_locs.py` 从 `radiusraw.txt` 自动生成
   必须与预处理后的图像分辨率匹配

 三、文件位置要求

所有输入文件应按照以下目录结构组织：

示例目录结构:
```
<project_name>/
│
├── DaughterCaptures/
│   ├── UnallignedToMother/         原始ST数据（必需）
│   │   ├── D1/
│   │   │   ├── cnts.tsv
│   │   │   ├── locs.tsv
│   │   │   └── he.
│   │   ├── D2/
│   │   └── ...
│   │
│   └── AllignedToMother/           对齐后的数据（必需，对齐后生成）
│       ├── D1/
│       │   ├── cnts.tsv
│       │   └── locs.tsv
│       ├── D2/
│       └── ...
│
└── MotherImage/
    ├── heraw.                    原始H&E图像（必需）
    ├── hescaled.                 缩放后的H&E图像（可选）
    ├── radiusraw.txt              原始spot半径（必需）
    ├── radius.txt                  缩放后的spot半径（可选）
    ├── genenames.txt              用户定义的基因列表（可选）
    └── markers.csv                 标记基因文件（可选）
```

路径参数:
 `prefix_general`: 项目根目录路径（必须包含 `DaughterCaptures` 和 `MotherImage` 子文件夹）
 `prefix`: 通常设置为 `${prefix_general}MotherImage/`

 四、数据格式详细说明

 TSV文件格式要求
 分隔符: Tab字符（`\t`）
 编码: UTF8
 行尾: Unix格式（`\n`）或Windows格式（`\r\n`）均可
 缺失值: 建议使用 `0` 或空字符串表示缺失值
 矩阵方向: 
   `cnts.tsv`: 基因 × spots（行=基因，列=spots）

 图像文件格式要求
 支持格式: 
   母图像: `.tiff`, `.tif`, `.svs`, `.ome.tif`, `.ome.tiff`, `.jpg`, `.png`, `.ndpi`, `.scn`, `.mrxs`
   子捕获图像: JPG, PNG, TIFF等常见格式
 颜色空间: RGB（3通道）
 数据类型: 8位无符号整数（0255）或更高位深
 建议: 使用TIFF格式以保持图像质量

 文本文件格式要求
 编码: UTF8 或 ASCII
 行尾: Unix格式（`\n`）或Windows格式（`\r\n`）均可
 数值格式: 浮点数或整数，使用点号（`.`）作为小数点

 CSV文件格式要求
 分隔符: 逗号（`,`）
 编码: UTF8
 表头: 第一行必须包含列名（`gene,label`）


 iSCALE 算法输出文件总结

 一、中间处理文件

 1. 图像预处理文件
 `heraw.tif` (2.1GB): 原始组织切片图像
 `hescaled.tiff` (862MB): 缩放后的图像
 `he.tiff` (861MB): 预处理后的最终图像
 `masksmall.png` (13KB): 自动检测的组织掩码（二值图像）
 `mask.png` (434KB): 完整尺寸的组织掩码

 2. 特征提取文件
 `embeddingshist.pickle` (3.4GB): 组织学特征嵌入
   从HIPT模型提取的多尺度组织学特征
   包含 cls（256×256级别）和 sub（16×16级别）两个层次的特征
   用于后续的基因表达预测

 `embeddingsgene.pickle` (1.5GB): 基因表达特征嵌入
   从训练好的神经网络模型提取的中间层特征
   基于基因表达模式提取的特征表示
   可用于聚类和空间模式分析

 3. 坐标和参数文件
 `radius.txt`: 预处理后的spot半径（像素）
 `locs.tsv`: 预处理后的spot坐标（4,620个spots）
 `genenames.txt`: 用于预测的基因列表（100个高变异基因）

 4. 掩膜优化文件 (`filterRGB/`)
 `masksmallrefined.png`: 优化后的掩膜（小尺寸）
 `maskrefined.png`: 优化后的掩膜（完整尺寸）
 `conserve_index.pickle`: 保留的组织区域索引

 二、主要输出结果

 1. 高分辨率基因表达预测 (`iSCALE_output/super_res_gene_expression/`)

 1.1 初步预测结果 (`cntssuper/`)
 位置: `iSCALE_output/super_res_gene_expression/cntssuper/`
 格式: 每个基因一个 `.pickle` 文件（100个基因）
 内容: 每个基因的超分辨率表达矩阵
   从spot级别提升到像素级别
   每个pickle文件包含一个2D numpy数组（float32类型），表示该基因在组织中的空间表达模式
   矩阵维度：1104 × 1408 像素
   每个文件大小：约6MB

 1.2 优化后的预测结果 (`cntssuperrefined/`)
 位置: `iSCALE_output/super_res_gene_expression/cntssuperrefined/`
 格式: 每个基因一个 `.pickle` 文件（100个基因）
 内容: 经过优化处理的超分辨率基因表达预测
   维度与 `cntssuper/` 相同：1104 × 1408 像素
   通过 refine_gene.py 进行优化，提高预测质量

 1.3 合并后的预测结果 (`cntssupermerged/`)
 位置: `iSCALE_output/super_res_gene_expression/cntssupermerged/`
 格式: 按因子组织的 `.pickle` 文件（factor0001.pickle 等）
 内容: 整合多个状态的预测结果
   使用5个训练状态的中位数进行集成预测
   提高预测的稳定性和准确性

 2. 可视化结果 (`iSCALE_output/super_res_ST_plots/`)

 2.1 初步可视化 (`cntssuperplots/`)
 位置: `iSCALE_output/super_res_ST_plots/cntssuperplots/`
 格式: PNG图像文件
 内容: 每个基因的空间表达热图（基于初步预测）

 2.2 优化后的可视化 (`cntssuperplotsrefined/`)
 位置: `iSCALE_output/super_res_ST_plots/cntssuperplotsrefined/`
 格式: PNG图像文件（100个基因）
 内容: 基于优化后预测的空间表达热图
   使用turbo色彩映射展示基因表达强度
   表达值高的区域显示为暖色（红/黄），低的显示为冷色（蓝/紫）
   结合组织掩码，只显示组织区域

 2.3 带掩膜的可视化 (`cntssuperplotsrefinedmask/`)
 位置: `iSCALE_output/super_res_ST_plots/cntssuperplotsrefinedmask/`
 格式: PNG图像文件（100个基因）
 内容: 应用了组织掩膜的基因表达可视化
   更好地突出组织区域的表达模式
   去除背景噪音

 3. 组织学嵌入可视化 (`iSCALE_output/HE_embeddings_plots/`)
 位置: `iSCALE_output/HE_embeddings_plots/`
 格式: PNG图像文件
 内容: 组织学特征嵌入的可视化
   `raw/cls/`: 256×256级别（cls）的原始嵌入可视化（100个文件）
   `raw/sub/`: 16×16级别（sub）的原始嵌入可视化（100个文件）
   `raw/rgb/`: RGB特征的嵌入可视化（3个文件）
   `dimreduced/cls/`: 降维后的cls嵌入可视化（100个文件）
   `dimreduced/sub/`: 降维后的sub嵌入可视化（100个文件）
   总计约591MB的可视化文件

 4. 基因表达聚类结果 (`iSCALE_output/clustersgene_15/`)
 位置: `iSCALE_output/clustersgene_15/`
 聚类数量: 15个不同的空间区域/组织类型
 内容:
   `labels.pickle` (12MB): 聚类标签矩阵（1104 × 1408，与预测结果匹配）
   `labels.png` (321KB): 聚类结果的可视化图像
   `masks/`: 每个聚类区域的掩码文件（15个PNG文件）
 说明: 基于基因表达特征（embeddingsgene.pickle）进行Kmeans聚类
   识别具有相似表达模式的空间区域
   可用于识别不同的组织区域或细胞类型

 5. 模型评估结果 (`iSCALE_output/cntssupereval/`)
 位置: `iSCALE_output/cntssupereval/`
 内容: 模型性能评估指标
   预测准确性的统计信息
   模型拟合质量的评估结果

 三、模型训练相关

 训练状态 (`states/`)
 位置: `states/00/`, `states/01/`, ..., `states/04/`
 说明: 5个模型训练状态（集成预测用）
 每个状态包含:
   `history.pickle`: 训练历史记录（损失函数变化）
   `history.png`: 训练曲线可视化
   `iSCALE_output/trainingdataplots/`: 训练数据可视化
     `x0000.png`: 输入特征示例
     `y0000.png`: 目标基因表达示例
 训练参数:
   训练轮数（epochs）: 1000
   状态数量（n_states）: 5
   集成方法: 使用5个模型的中位数作为最终预测

 四、辅助文件

 基因列表
 `genenames.txt`: 用于预测的基因列表（100个高变异基因）
   通过 select_genes.py 从原始表达矩阵中筛选
   选择变异系数最高的前100个基因

 坐标文件
 `locs.tsv`: 预处理后的spot坐标
   包含4,620个spots的位置信息
   经过坐标缩放，与预处理后的图像对齐

 参数文件
 `radius.txt`: spot半径（像素）
   经过缩放处理，与预处理后的图像分辨率匹配

 五、输出统计信息（基于实际数据）

 超分辨率预测结果
 预测基因数量: 100个基因
 每个基因预测矩阵维度: 1104 × 1408 像素
 空间分辨率: 从4,620个spots提升到约155万像素点（~336倍提升）
 文件格式: 每个基因一个`.pickle`文件，包含numpy数组（float32）
 预测结果目录:
   `cntssuper/`: 594MB（100个基因文件）
   `cntssuperrefined/`: 594MB（100个基因文件）
   `cntssupermerged/`: 593MB（合并后的因子文件）

 聚类结果
 聚类数量: 15个不同的空间区域/组织类型
 聚类标签维度: 1104 × 1408（与预测结果匹配）
 聚类方法: 基于基因表达特征的Kmeans聚类
 输出文件: 16MB

 模型训练状态
 训练状态数量: 5个（states/00 到 states/04）
 集成预测: 使用5个模型的中位数作为最终预测，提高稳定性
 每个状态的输出: 约15MB（包含训练历史和可视化）

 可视化文件
 基因表达可视化: 约198MB（100个基因 × 多种格式）
 组织学嵌入可视化: 约591MB（303个PNG文件）
 总可视化文件: 约789MB
