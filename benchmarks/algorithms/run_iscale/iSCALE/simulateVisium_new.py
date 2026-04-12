import os
import time
import json
import argparse

import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sns

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from PIL import Image
Image.MAX_IMAGE_PIXELS = None 


def Simulated(feature_matrix, cell_type, spatial_loc, CoordinateXlable, CoordinateYlable, window, outdir):
    '''
    Simulate Psudo Visium Data from Xenium

    simulate psudo-visium data from xenium
    create adata and save h5ad file

    Outputs:
        combined_spot_info.json: spot information, including cell centroid, cell id, cell_type
        combined_cell_counts.tsv: number of cells in each spot
        combined_spot_locations.tsv: location information of each spot
        combined_counts_matrix.tsv: psudo-visium counts matrix
        combined_spot_clusters.tsv: cell type counts in each spot
        simulated_data.h5ad: anndata object

    Args:
        feature_matrix: anndata, Xenium cell-feature matrix
        cell_type: pandas DataFrame, cell_id and cell_type
        spatial_loc: list, spatial location of each cell
        CoordinateXlabel: str
        CoordinateYlabel: str
        window: int, size of each psudo-spot
        outdir: str, dir to save outputs
    
    Returns:
        None

    '''

    if os.path.exists(outdir):
        print ('The output file is in %s' % outdir)
    else:
        os.mkdir(outdir)

    spatial_rna = feature_matrix.X.toarray()

    # 获取spot内细胞信息
    combined_spot = []
    combined_spot_loc = []
    spot_centroid_loc = []

    c = 0  # 计spot数
    for x in np.arange(spatial_loc[CoordinateXlable].min() // window, spatial_loc[CoordinateXlable].max() // window + 1):  # 遍历x轴
        for y in np.arange(spatial_loc[CoordinateYlable].min() // window, spatial_loc[CoordinateYlable].max() // window + 1):  # 遍历y轴
            tmp_loc = spatial_loc[(x * window <= spatial_loc[CoordinateXlable]) &  # 获取范围内的细胞
                                  (spatial_loc[CoordinateXlable] < (x + 1) * window) &
                                  (y * window <= spatial_loc[CoordinateYlable]) &
                                  (spatial_loc[CoordinateYlable] < (y + 1) * window)]
            if len(tmp_loc) > 0:
                centroid_loc = list(zip(tmp_loc[CoordinateXlable], tmp_loc[CoordinateYlable]))  # 打包坐标
                spot_centroid_loc.append(centroid_loc)  # 细胞坐标
                combined_spot_loc.append([x, y])  # 孔坐标
                # combined_spot.append((tmp_loc.index + 1).to_list())  #TODO 为什么+1 --> 因为之前在combined_Spot是想写cell_id，而不是index
                combined_spot.append((tmp_loc.index).to_list())
                c += 1
    # 生成spot和对应细胞编号和对应细胞坐标和对应细胞类型的字典
    spot_info = {i: {'centroid': spot_centroid_loc[i], 'cell_id': combined_spot[i]} for i in range(len(combined_spot))}

    # 获取细胞类型
    # cell_type_dict = cell_type['cell_type'].to_dict()
    for idx in spot_info.keys():
        types = []
        for ids in spot_info[idx]['cell_id']:
            celltype = cell_type.loc[ids,'cell_type']
            types.append(celltype)
        spot_info[idx]['cell_type'] = types
    
    # 保存为json
    with open('%s/combined_spot_info.json' % outdir, 'w') as json_file:
        json.dump(spot_info, json_file, indent=4)

    # 每个spot内细胞数
    combined_cell_counts = pd.DataFrame([len(s) for s in combined_spot], columns=['cell_count'])
    combined_cell_counts.to_csv('%s/combined_cell_counts.tsv' % outdir, sep='\t')
    print ('The simulated spot has cell number ranging from %s to %s' % (str(combined_cell_counts.min()[0]), str(combined_cell_counts.max()[0])))
    
    # 保存spot坐标
    combined_spot_loc = pd.DataFrame(combined_spot_loc, columns=['x', 'y'])
    combined_spot_loc['x_pixel'] = (combined_spot_loc['x'] * window + round(window / 2, 2)) / scale  # 质心坐标
    combined_spot_loc['y_pixel'] = (combined_spot_loc['y'] * window + round(window / 2, 2)) / scale
    combined_spot_loc.to_csv('%s/combined_spot_locations.tsv' % outdir, sep='\t')

    # spot内基因表达
    combined_spot_exp = []
    for s in combined_spot:
        # s = [x - 1 for x in s]  #TODO 为什么-1
        s = [x for x in s]
        combined_spot_exp.append(spatial_rna[s,:].sum(axis=0))
    combined_spot_exp = pd.DataFrame(combined_spot_exp, columns=feature_matrix.var.index)
    combined_spot_exp.to_csv('%s/combined_counts_matrix.tsv' % outdir, sep='\t')

    combined_spot_clusters = pd.DataFrame(np.zeros((len(combined_spot_loc.index), len(cell_type['cell_type'].unique()))), columns=cell_type['cell_type'].unique())
    for i, c in enumerate(combined_spot):
        # c = [x - 1 for x in c]  #TODO 为什么-1
        c = [x for x in c]
        for clt in cell_type.loc[c, 'cell_type']:
            combined_spot_clusters.loc[i, clt] += 1
    combined_spot_clusters.to_csv('%s/combined_spot_clusters.tsv' % outdir, sep='\t')
    
    # 将结果保存成h5ad
    # 创建 AnnData 对象
    adata = ad.AnnData(X=combined_spot_exp)

    # 添加细胞元数据
    adata.obs = combined_spot_loc
    adata.var = feature_matrix.var
    adata.write('%s/simulated_data.h5ad' % outdir)
    print ('%s simulated spots' % str(combined_spot_clusters.shape[0]))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='input xenium data and transform into visium.')
    parser.add_argument('--data_dir', '-d', type=str)
    parser.add_argument('--type_file', '-t', type=str)
    parser.add_argument('--HE_aligned', '-i', type=str)
    parser.add_argument('--is_scale', action='store_true')
    parser.add_argument('--window_size', '-w', type=int, default=55)
    parser.add_argument('--output_dir', '-o', type=str)
    args = parser.parse_args()

    data_dir = args.data_dir
    type_file = args.type_file
    he_aligned = args.HE_aligned
    window_size = args.window_size
    output_dir = args.output_dir

    # 读取数据
    print('Load Data...')
    feature_matrix = sc.read_10x_h5('%s/cell_feature_matrix.h5' % data_dir)
    cells = pd.read_parquet('%s/cells.parquet' % data_dir)

    cell_type = pd.read_csv(type_file)
    cell_type = cell_type.rename(columns={'group': 'cell_type'})

    HE_img = Image.open(he_aligned)
    width, height = HE_img.size
    scale = 0.2125  # 微米转pixel

    # 模拟Visium
    print('Start Simulating...')
    time_start = time.time()

    min_x = cells['x_centroid'].min()
    max_x = cells['x_centroid'].max()
    min_y = cells['y_centroid'].min()
    max_y = cells['y_centroid'].max()

    # 取cell_type和cells的细胞交集
    aligned = cells.merge(cell_type[['cell_id','cell_type']], on='cell_id', how='inner')
    cells_intersect = aligned['cell_id'].tolist()
    
    cell_type = cell_type[cell_type['cell_id'].isin(cells_intersect)]
    cells = cells[cells['cell_id'].isin(cells_intersect)].reset_index()
    feature_matrix = feature_matrix[cells_intersect]
    # 添加细胞类别信息
    cell_type = cell_type.set_index('cell_id').reindex(cells['cell_id']).reset_index()  # 与cells对应，并去除索引

    spatial_loc = cells
    # 筛选 x 和 y 坐标小于 (width, height) 的行
    if args.is_scale:
        filtered_spatial_loc = spatial_loc[(spatial_loc['x_centroid'] < width * 2 * scale) & (spatial_loc['y_centroid'] < height * 2 * scale)]  # * 2 恢复至未放缩状态
    else:
        filtered_spatial_loc = spatial_loc[(spatial_loc['x_centroid'] < width * scale) & (spatial_loc['y_centroid'] < height * scale)]  # * 2 恢复至未放缩状态

    CoordinateX = 'x_centroid'
    CoordinateY = 'y_centroid'

    Simulated(feature_matrix, cell_type, filtered_spatial_loc, CoordinateX, CoordinateY, window_size, output_dir) 
    
    time_end = time.time()
    print('Done: %f s' % (time_end - time_start))

