import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing as mp
from tqdm import tqdm

import json
import os
import random

from utils import *


if __name__ == '__main__':

    positions_file = './example/tissue_positions.csv'
    spot_combine_file = './example/deconvolution_results.csv'
    proportion = False

    scalefactor_file = './example/scalefactors_json.json'

    nuclei_seg_json_file = './example/CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_tissue_image.json'
    type_dict_file = './example/type_dict.json'

    spot_infer_save_file = './example/outs/spot_infer.json'
    cell_info_save_file = './example/outs/cell_info.json'


    with open(scalefactor_file, "r") as f:
        scalefactor = json.load(f)

    id_type = {0:'Neo', 1: 'Inflam', 2:'Conn', 3:'Dead', 4:'Epi'}

    with open(type_dict_file, 'r') as json_f:
        type_dict = json.load(json_f)
    inverse_type_dict = {subtype: supertype for supertype, subtypes in type_dict.items() for subtype in subtypes}

    cell_info = {}
    with open(nuclei_seg_json_file, 'r') as f:
        data = json.load(f)
        cell_dict = data['nuc']
        for cell_id in cell_dict:
            x, y = cell_dict[cell_id]['centroid']
            type_prob = cell_dict[cell_id]['type_prob']
            predict_type = type_prob[1:].index(max(type_prob[1:]))
            predict_type = id_type.get(predict_type, 'wrong')
            cell_info[cell_id]={
                    'centroid': [y, x],
                    'type_prob': type_prob,
                    'type': predict_type,
                    'assign':'unassigned type',
                    'random_guess':'unassigned type',
                    'gt':'unassigned type'
            }
    print(f"Number of Cells Segmented in H&E Image: {len(cell_info)}.")
    from collections import Counter
    types = [value['type'] for value in cell_info.values()]
    type_counts = Counter(types)
    print(f"Number of Each Cell Type: {type_counts}")

    spatial_loc = pd.read_csv(positions_file, index_col=0, header=0)
    spatial_loc.columns = ['in_tissue', 'x', 'y', 'pixel_x', 'pixel_y']

    r = scalefactor['spot_diameter_fullres'] / 2

    spatial_interval = spatial_loc.copy()
    spatial_interval = spatial_interval[spatial_interval["x"] % 2 == 0]
    spatial_interval_new = spatial_interval.copy()  
    spatial_interval_new["y"] += 1 
    x, y = spatial_loc[['x','y']].values[0]
    w = spatial_loc[(spatial_loc['y'] == y+1) & (spatial_loc['x'] == x+1)]['pixel_y'].values[0] - spatial_loc['pixel_y'].values[0]
    h = spatial_loc[(spatial_loc['x'] == x+2) & (spatial_loc['y'] == y)]['pixel_x'].values[0] - spatial_loc['pixel_x'].values[0]
    spatial_interval_new['pixel_y'] = (spatial_interval_new['pixel_y'] + w).astype(int)
    spatial_interval = pd.concat([spatial_interval, spatial_interval_new], ignore_index=True)
    spatial_interval['index_str'] = spatial_interval.apply(lambda row: f"{int(row['x'])}x{int(row['y'])}", axis=1)
    spatial_interval.set_index('index_str', inplace=True)

    if os.path.exists(spot_infer_save_file):
        with open(spot_infer_save_file, 'r') as f:
            spot_infer = json.load(f) 
    else:
        spot_infer = {}
        spot_interval = {}
        spot_size = None
        param_list = [(spot, spot_size, spatial_loc, cell_info, r) for spot in spatial_loc.index]
    
        with mp.Pool(processes=3) as pool:
            results = list(pool.starmap(process_spot, param_list), total=len(spatial_loc.index))

        cell_inspot = []
        for spot, data in results:
            spot_infer[spot] = data
            cell_inspot.extend(data['cell_id'])
        
        cell_interval = {k: v for k, v in cell_info.items() if k not in cell_inspot}
        spot_size = [w, h]
        param_list = [(spot, spot_size, spatial_interval, cell_interval, r) for spot in spatial_interval.index]
        with mp.Pool(processes=10) as pool:
            results = list(pool.starmap(process_spot, param_list), total=len(spatial_interval.index))

        for spot, data in results:
            spot_interval[spot] = data
        
        spot_infer.update(spot_interval)
        
        with open(spot_infer_save_file, 'w') as json_f:
            json.dump(spot_infer, json_f, indent=4)

    cell_compose = pd.read_csv(spot_combine_file, index_col=0, header=0)
    if proportion:
        cell_prop = cell_compose
    else:
        cell_prop = cell_compose.div(cell_compose.sum(axis=1), axis=0)


    new_cell_compose = cell_prop.copy()
    for spot in cell_prop.index:
        cell_count = len(spot_infer[spot]['cell_id'])
        if sum(cell_prop.loc[spot]) != 0:
            new_cell_compose.loc[spot] = prop2count(cell_prop.loc[spot,:], cell_count)
        else:
            new_cell_compose.loc[spot] = 0

    type_num = len(type_dict.keys())
    import itertools
    type_list = list(type_dict.keys())
    o = [[int(i == j) for i in range(type_num)] for j in range(type_num)]
    new_cell_compose['cell_count'] = new_cell_compose.sum(axis=1)
    random.seed(724)
    def infer(spot):
        major_type_dict = {key: 0 for key in type_dict.keys()}
        minor_type_dict = {key:0 for key in new_cell_compose.columns[:-1]}
        minor_type_total = sum(type_dict.values(),[])

        for minor_type in new_cell_compose.columns[:-1]:
            assert minor_type in minor_type_total, f'Deconvolution type {minor_type} is not in the type dictionary.'
            minor_cell_count = int(new_cell_compose.loc[spot, minor_type])
            minor_type_dict[minor_type] += minor_cell_count 
            major_type = inverse_type_dict[minor_type]
            major_type_dict[major_type] += minor_cell_count 

        minor_type_spotlist = []
        for minor, num in minor_type_dict.items():
            minor_type_spotlist = minor_type_spotlist + [minor] * num

        counts = list(major_type_dict.values())
        P_list = []
        spot_cell_id = spot_infer[spot]['cell_id']

        if spot_cell_id == []:
            return {}

        for cell_id in spot_cell_id:
            P = cell_info[cell_id]['type_prob']
            new_P = P[1:4]
            new_P.append(P[5])  
            P_sum = sum(new_P)
            scaled_P = [prob/P_sum for prob in new_P]
            P_list.append(scaled_P)

        spot_score, assign_result = cell_assign(counts, P_list, o, type_num, type_list)

        # assign minor type
        spot_major = list(set(assign_result))
        cell_assign_dict = {}   # cell_id -> assign
        for major in spot_major:
            indices = [index for index, value in enumerate(assign_result) if value == major]
            num = len(indices)
            minor_list = [minor for minor in minor_type_spotlist if minor in type_dict[major]]
            random.shuffle(minor_list)
            major_cell_id = [spot_cell_id[i] for i in indices]
            for idx, cell_id in enumerate(major_cell_id):
                cell_assign_dict[cell_id] = {'assign': minor_list[idx]}

        random.shuffle(minor_type_spotlist)
        for idx, cell_id in enumerate(spot_cell_id):
            if cell_id not in cell_assign_dict:
                cell_assign_dict[cell_id] = {}
            cell_assign_dict[cell_id]['random_guess'] = minor_type_spotlist[idx]

        return cell_assign_dict

    spots = new_cell_compose.index
    with mp.Pool(processes=10) as pool:
        results = list(tqdm(pool.imap(infer, spots), total=len(spots)))
    for res_dict in results:
        for cell_id, assign_info in res_dict.items():
            for k,v in assign_info.items():
                cell_info[cell_id][k] = v

    for interval in spatial_interval.index:
        x, y = spatial_interval.loc[interval,["x","y"]].astype(int)
        u_bound = spatial_interval['x'].max()
        b_bound = spatial_interval['y'].max()
        if y // 2 == 0:
            neighbors = [
                (x, y),
                (min(x+2, u_bound), y),
                (min(x+1, u_bound), min(y+1, b_bound))
            ]
        else:
            neighbors = [
                (min(x+1, u_bound), y),
                (x, min(y+1, b_bound)),
                (min(x+2, u_bound), min(y+1, b_bound))
            ]
        valid_neighbors = []
        for neighbor in neighbors:
            (x, y) = (neighbor[0], neighbor[1])
            valid_neighbor = spatial_loc[(spatial_loc["x"] == x) & (spatial_loc["y"] == y)].index.tolist()
            valid_neighbors.extend(valid_neighbor)

        near_cells=list()
        for neighbor in valid_neighbors:
            near_cells.extend(spot_infer[neighbor]['cell_id'])
        softmax_prob =np.array([cell_info[cell]['type_prob'] for cell in near_cells])

        if len(softmax_prob) > 0:
            for cell in spot_infer[interval]['cell_id']:
                cell_prob = cell_info[cell]['type_prob']
                similarities = cosine_similarity([cell_prob], softmax_prob)
                top_indices = np.argsort(similarities[0])[-min(5, len(similarities[0])):][::-1]
                assign_types = [cell_info[near_cells[idx]]['assign'] for idx in top_indices]
                random_types = [cell_info[near_cells[idx]]['random_guess'] for idx in top_indices]
                assign_counter = Counter(assign_types)
                random_counter = Counter(random_types)
                cell_info[cell]['assign'] = assign_counter.most_common(1)[0][0]
                cell_info[cell]['random_guess'] = random_counter.most_common(1)[0][0]

        else:
            for cell in spot_infer[interval]['cell_id']:
                major_type = cell_info[cell]['type']
                if major_type == "Dead":
                    assign_type = "Dead"
                else:
                    if len(type_dict[major_type]) > 0:
                        assign_type = random.choice(type_dict[major_type])
                cell_info[cell]['assign'] = assign_type
                cell_info[cell]['random_guess'] = assign_type

    with open(cell_info_save_file, 'w') as f:
        json.dump(cell_info, f, indent=4)
    
    print("Done!")
