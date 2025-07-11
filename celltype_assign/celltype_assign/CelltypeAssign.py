import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from collections import Counter
import random
import json
import os
from pathlib import Path

from .utils import prop2count, assign, find_most_similar_sample, qc
from .set_config import CelltypeAssignConfig


class CelltypeAssign:
    def __init__(
            self,
            config: CelltypeAssignConfig,
        ):
        config.validate()
        self.config = config

        for key, value in config.__dict__.items():
            setattr(self, key, value)
    
        self.nuclei_seg_info = None
        self.nuclei_seg_types = None
        self.spatial_positions = None
        self.r = None
        self.cell_composition = None
        self.type_mapping_dict = None

        self.cell_info = None
        self.spatial_intervals = None
        self.w = None
        self.h = None
        self.spatial_info = None


    def preprocess(self):
        """Load data, create cell info and spatial info, and process cell composition.
        """
        print('Loading data...')
        self.nuclei_seg_info, self.nuclei_seg_types, self.spatial_positions, \
            self.r, self.cell_composition, self.type_mapping_dict = self._load_data()

        print('Creating cell info...')
        self.cell_info = self._create_cell_info()
        types = [value['type'] for value in self.cell_info.values()]
        type_counts = dict(Counter(types))
        print('    Number of nuclei segmented: %s' % len(self.cell_info))
        print('    Number of each nuclei type: %s' % type_counts)

        print('Defining spatial intervals...')
        if self.with_interval:
            self.spatial_intervals, self.w, self.h = self._define_spatial_intervals()

        print('Creating spatial info, using %d cores...' % self.n_processes)
        self.spatial_info = self._create_spatial_info()

        print('Processing cell composition...')
        self.cell_composition = self._process_cell_composition()

        print('Ready to assign cell types!')

        return


    def celltype_assign(self):
        """Assign cell types to cells in spots and intervals.
        """
        print('Assigning cell types...')

        random.seed(707)

        # process type mappind dict
        inversed_type_mapping_dict = {
            subtype: supertype for supertype, subtypes in self.type_mapping_dict.items() \
                for subtype in subtypes
        }

        major_type_list = list(self.type_mapping_dict.keys())
        n_major_type = len(major_type_list)

        o = [[int(i == j) for i in range(n_major_type)] for j in range(n_major_type)]

        self.cell_composition['cell_count'] = self.cell_composition.sum(axis=1)

        # spots
        print('    Processing spots, using %d cores...' % self.n_processes)
        if self.cbi_threshold is not None:
            print('        ### Using CBI to refine cell type assignment, CBI threshold: %.1f ###' % self.cbi_threshold)

        param_list = [(s, self.spatial_info, self.cell_info, self.cell_composition,
                       self.type_mapping_dict, inversed_type_mapping_dict, self.nuclei_seg_types,
                       o, n_major_type, major_type_list, self.cbi_threshold, self.metric, self.top_k) for s in self.cell_composition.index]
        with mp.Pool(processes=self.n_processes) as pool:
            results = pool.starmap(self._infer_spot, param_list)

        # write into cell_info
        for res_dict in results:
            for cell_id, assign_info in res_dict.items():
                for k, v in assign_info.items():
                    self.cell_info[cell_id][k] = v

        # intervals
        if self.with_interval:
            print('    Processing intervals, using %d cores...' % self.n_processes)
            param_list = [(interval, self.spatial_intervals, self.spatial_positions, 
                      self.spatial_info, self.cell_info, self.nuclei_seg_types, self.spot_shape,
                      self.type_mapping_dict, self.metric, self.top_k)
                      for interval in self.spatial_intervals.index]

            with mp.Pool(processes=self.n_processes) as pool:
                results = pool.starmap(self._process_interval, param_list)

            # write into cell_info
            for interval_results in results:
                for cell_id, assign_info in interval_results.items():
                    self.cell_info[cell_id].update(assign_info)

        # save
        Path(self.cell_info_json_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cell_info_json_file, 'w') as f:
            json.dump(self.cell_info, f, indent=4)

        print('Cell types assigned!')

        return


    def _load_data(self):
        with open(self.nuclei_seg_json_file, 'r') as f:
            nuclei_seg_info = json.load(f)
            nuclei_seg_info = nuclei_seg_info['nuc']

        with open(self.nuclei_seg_types_json_file, 'r') as f:
            nuclei_seg_types = json.load(f)
        nuclei_seg_types = {int(k): v for k, v in nuclei_seg_types.items()}

        if self.header:
            spatial_positions = pd.read_csv(self.spatial_positions_file, index_col=0, header=0)
        else:
            spatial_positions = pd.read_csv(self.spatial_positions_file, index_col=0, header=None)
        spatial_positions.columns = ['in_tissue', 'y', 'x', 'pixel_y', 'pixel_x']
        spatial_positions = spatial_positions.loc[spatial_positions.loc[:, 'in_tissue'] == 1]

        with open(self.scalefactors_json_file, 'r') as f:
            scalefactors = json.load(f)
        r = scalefactors['spot_diameter_fullres'] / 2

        cell_composition = pd.read_csv(self.deconvolution_file, index_col=0, header=0)
        # intersect between cell_composition and spatial_positions
        spot_keep = spatial_positions.index.intersection(cell_composition.index)
        spatial_positions = spatial_positions.loc[spot_keep]
        cell_composition = cell_composition.loc[spot_keep, :]

        type_mapping_dict = {}
        with open(self.type_mapping_json_file, 'r') as f:
            type_mapping_dict_raw = json.load(f)
        for v in nuclei_seg_types.values():  # order type_mapping_dict by nuclei_seg_types
            if v in type_mapping_dict_raw.keys():
                type_mapping_dict[v] = type_mapping_dict_raw[v]

        return nuclei_seg_info, nuclei_seg_types, spatial_positions, r, cell_composition, type_mapping_dict


    def _create_cell_info(self):
        cell_info = {}
        max_x = self.spatial_positions['pixel_x'].max() + self.r
        max_y = self.spatial_positions['pixel_y'].max() + self.r
        min_x = self.spatial_positions['pixel_x'].min() - self.r
        min_y = self.spatial_positions['pixel_y'].min() - self.r

        for i in self.nuclei_seg_info:
            x, y = self.nuclei_seg_info[i]['centroid']
            x_scale = x / self.cell_seg_scale
            y_scale = y / self.cell_seg_scale

            # filter out cells that are out of the ST region
            if x_scale > max_x or y_scale > max_y or x_scale < min_x or y_scale < min_y:
                continue

            type_prob = self.nuclei_seg_info[i]['type_prob']
            predict_type = type_prob[1:].index(max(type_prob[1:]))  # 0 is background
            predict_type = self.nuclei_seg_types.get(predict_type, 'wrong')
            cell_info[i]={
                'centroid': [x_scale, y_scale],
                'type_prob': type_prob,
                'type': predict_type,
                'assign':'unassigned type',
                'random_guess':'unassigned type',
                'gt':'unassigned type'
            }

        return cell_info


    def _define_spatial_intervals(self):
        spatial_intervals = self.spatial_positions.copy()
        if self.spot_shape == 'circle':
            spatial_intervals = spatial_intervals.loc[spatial_intervals['x'] % 2 == 0]
            spatial_intervals_new = spatial_intervals.copy()

            spatial_intervals_new.loc[:, 'y'] += 1
            i0 = 100
            x0, y0 = self.spatial_positions.loc[self.spatial_positions.index[i0], ['x', 'y']]
            w = (self.spatial_positions.loc[(self.spatial_positions.loc[:, 'y'] == y0 + 1) & (self.spatial_positions.loc[:, 'x'] == x0 + 1), 'pixel_y'] - \
                self.spatial_positions.loc[self.spatial_positions.index[i0], 'pixel_y'])
            h = (self.spatial_positions.loc[(self.spatial_positions.loc[:, 'x'] == x0 + 2) & (self.spatial_positions.loc[:, 'y'] == y0), 'pixel_x'] - \
                self.spatial_positions.loc[self.spatial_positions.index[i0], 'pixel_x'])
            w, h = w.values[0], h.values[0]
            spatial_intervals_new.loc[:, 'pixel_y'] = (spatial_intervals_new.loc[:, 'pixel_y'] + w).astype(int)

            spatial_intervals = pd.concat([spatial_intervals, spatial_intervals_new], ignore_index=True)
        
        elif self.spot_shape == 'square':
            i0 = 100
            x0, y0 = self.spatial_positions.loc[self.spatial_positions.index[i0], ['x', 'y']]
            h = self.spatial_positions[(self.spatial_positions['y'] == y0+1) & (self.spatial_positions['x'] == x0)]['pixel_y'].values[0] - self.spatial_positions['pixel_y'].values[0]
            w = self.spatial_positions[(self.spatial_positions['x'] == x0+1) & (self.spatial_positions['y'] == y0   )]['pixel_x'].values[0] - self.spatial_positions['pixel_x'].values[0]

        spatial_intervals['index_str'] = spatial_intervals.apply(lambda row: '%dx%d' % (row['x'], row['y']), axis=1)
        spatial_intervals = spatial_intervals.set_index('index_str')

        return spatial_intervals, w, h


    @staticmethod
    def _process_spatial(args):
        s = args['s']
        spot_shape = args['spot_shape']
        spatial_positions = args['spatial_positions']
        cell_info = args['cell_info']

        r = args.get('r')
        interval_size = args.get('interval_size')

        if spot_shape == "interval":
            x_min = min(spatial_positions.loc[s, 'pixel_x'], spatial_positions.loc[s, 'pixel_x'] + interval_size[1])
            x_max = max(spatial_positions.loc[s, 'pixel_x'], spatial_positions.loc[s, 'pixel_x'] + interval_size[1])
            y_min = min(spatial_positions.loc[s, 'pixel_y'], spatial_positions.loc[s, 'pixel_y'] + interval_size[0])
            y_max = max(spatial_positions.loc[s, 'pixel_y'], spatial_positions.loc[s, 'pixel_y'] + interval_size[0])
        # spots
        elif spot_shape == 'square':
            x_min = min(spatial_positions.loc[s, 'pixel_x'] - r, spatial_positions.loc[s, 'pixel_x'] + r)
            x_max = max(spatial_positions.loc[s, 'pixel_x'] - r, spatial_positions.loc[s, 'pixel_x'] + r)
            y_min = min(spatial_positions.loc[s, 'pixel_y'] - r, spatial_positions.loc[s, 'pixel_y'] + r)
            y_max = max(spatial_positions.loc[s, 'pixel_y'] - r, spatial_positions.loc[s, 'pixel_y'] + r)
        elif spot_shape == 'circle':
            x_center = spatial_positions.loc[s, 'pixel_x']
            y_center = spatial_positions.loc[s, 'pixel_y']
        else:
            raise ValueError('Invalid spot shape: %s' % spot_shape)

        # infer if cells in spot
        cell_ids = []
        centroids = []

        # numpy is faster
        all_cell_ids = np.array(list(cell_info.keys()))
        all_centroids = np.array([cell['centroid'] for cell in cell_info.values()])

        # filter cells in spot
        if spot_shape == 'circle':
            # calculate distances
            distances = np.sqrt((all_centroids[:, 0] - x_center) ** 2 + (all_centroids[:, 1] - y_center) ** 2)
            idx = distances < r
        else:
            idx = (all_centroids[:, 0] >= x_min) & (all_centroids[:, 0] <= x_max) & \
                (all_centroids[:, 1] >= y_min) & (all_centroids[:, 1] <= y_max)

        filtered_cell_ids = all_cell_ids[idx]
        filtered_centroids = all_centroids[idx]

        cell_ids.extend(filtered_cell_ids)
        centroids.extend(filtered_centroids.tolist())

        return s, \
            {'cell_id': cell_ids, 'centroid': centroids}


    def _create_spatial_info(self):
        # add cells to spatial
        if os.path.exists(self.spatial_info_json_file) and self.load_from_file:  # load from file
            print('    Loading spatial info from file...')
            with open(self.spatial_info_json_file, 'r') as f:
                spatial_info = json.load(f)
        else:
            print('    Mapping cells into space...')
            spatial_info = {}
            spatial_info_interval = {}

            # define cells in spots
            # prepare parameters
            param_list = []
            for s in self.spatial_positions.index:
                param_list.append({
                    's': s,
                    'spot_shape': self.spot_shape,
                    'spatial_positions': self.spatial_positions,
                    'cell_info': self.cell_info,
                    'r': self.r,  
                })

            # use multiprocess
            with mp.Pool(processes=self.n_processes) as pool:
                results = pool.map(self._process_spatial, param_list)
            cells_in_spots = []
            for s, data in results:
                spatial_info[s] = data
                cells_in_spots.extend(data['cell_id'])

            # define cells in intervals (only if intervals exist)
            if hasattr(self, 'with_interval') and self.with_interval:
                cells_in_intervals = {k: v for k, v in self.cell_info.items() if k not in cells_in_spots}
                interval_size = [self.h, self.w]
                param_list = []
                for s in self.spatial_intervals.index:
                    param_list.append({
                        's': s,
                        'spot_shape': "interval",
                        'spatial_positions': self.spatial_intervals,
                        'cell_info': cells_in_intervals,
                        'interval_size': interval_size
                    })

                with mp.Pool(processes=self.n_processes) as pool:
                    results = pool.map(self._process_spatial, param_list)
                for s, data in results:
                    spatial_info_interval[s] = data

                spatial_info.update(spatial_info_interval)

            # write into file
            Path(self.spatial_info_json_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.spatial_info_json_file, 'w') as f:
                json.dump(spatial_info, f, indent=4)

        return spatial_info
    
    def _select_top_m_celltypes(self, cell_proportion):
        
        processed_proportion = cell_proportion.copy()
        
        for spot_idx in processed_proportion.index:
            spot_proportions = processed_proportion.loc[spot_idx]
            top_m_celltypes = spot_proportions.nlargest(self.top_m_count).index
            
            for celltype in processed_proportion.columns:
                if celltype not in top_m_celltypes:
                    processed_proportion.loc[spot_idx, celltype] = 0
            total_proportion = processed_proportion.loc[spot_idx].sum()
            if total_proportion > 0:
                processed_proportion.loc[spot_idx] = processed_proportion.loc[spot_idx] / total_proportion
            
        return processed_proportion


    def _process_cell_composition(self):
        # to proportion
        if self.is_proportion:
            cell_proportion = self.cell_composition
        else:
            cell_proportion = self.cell_composition.div(self.cell_composition.sum(axis=1), axis=0)
        
        # Select top m cell types
        if self.cell_composition_mode == 'top':
            print(f'    Using top {self.top_m_count} cell types per spot')
            cell_proportion = self._select_top_m_celltypes(cell_proportion)
        else:
            print('    Using all cell types (full mode)')

        # to counts
        cell_composition_new = cell_proportion.copy()
        for s in cell_proportion.index:
            cell_count = len(self.spatial_info[s]['cell_id'])
            if sum(cell_proportion.loc[s]) != 0:
                cell_composition_new.loc[s] = prop2count(cell_proportion.loc[s, :], cell_count)
            else:
                cell_composition_new.loc[s] = 0

        return cell_composition_new


    @staticmethod
    def _replace_unwanted_type(cell_info_item, nuclei_seg_types, type_mapping_dict):
        type_prob = cell_info_item['type_prob'][1:]  # remove background

        prob_with_indices = [(prob, idx) for idx, prob in enumerate(type_prob)]
        prob_with_indices.sort(reverse=True, key=lambda x: x[0])

        for prob, type_idx in prob_with_indices:
            if type_idx in nuclei_seg_types:
                major_type = nuclei_seg_types[type_idx]
                if major_type != 'Dead' and major_type in type_mapping_dict:
                    return random.choice(type_mapping_dict[major_type])

        for major_type, minor_types in type_mapping_dict.items():
            if major_type != 'Dead' and minor_types:
                return random.choice(minor_types)

        return 'Dead'


    @staticmethod
    def _infer_spot(s, spatial_info, cell_info, cell_composition,
                    type_mapping_dict, inversed_type_mapping_dict, nuclei_seg_types,
                    o, n_major_type, major_type_list, cbi_threshold, metric, top_k):
        spatial_cell_id = spatial_info[s]['cell_id']
        if spatial_cell_id == []:
            return {}

        # step 1: get info
        major_type_dict = {key: 0 for key in type_mapping_dict.keys()}  # major type -> count
        minor_type_dict = {key:0 for key in cell_composition.columns[:-1]}  # minor type -> count
        minor_type_total = sum(type_mapping_dict.values(),[])

        # number of cells in each type
        for minor_type in minor_type_dict.keys():
            assert minor_type in minor_type_total, \
                'Deconvolution type `%s` is not in the type mapping dictionary.' % minor_type
            minor_cell_count = int(cell_composition.loc[s, minor_type])
            minor_type_dict[minor_type] += minor_cell_count 
            major_type = inversed_type_mapping_dict[minor_type]
            major_type_dict[major_type] += minor_cell_count   # major type counts according to minor type

        minor_type_spatial_list = []  # all minor types
        for minor, num in minor_type_dict.items():
            minor_type_spatial_list = minor_type_spatial_list + [minor] * num

        p_list = []
        # remove unwanted nuclei classification types
        unwanted_id = [int(i) for i in nuclei_seg_types.keys() if nuclei_seg_types[i] not in major_type_dict.keys()]
        # scale new probabilities
        for cell_id in spatial_cell_id:
            p = cell_info[cell_id]['type_prob'][1:]  # remove background
            p_new = [p[i] for i in range(len(p)) if i not in unwanted_id]
            p_sum = sum(p_new)
            p_scaled = [prob / p_sum for prob in p_new]
            p_list.append(p_scaled)

        major_cell_counts = list(major_type_dict.values())

        # step 2: assign celltypes
        # dynamic programming
        best_score, assign_result = assign(major_cell_counts, p_list, o, n_major_type, major_type_list)

        # assign minor types
        spatial_major = list(set(assign_result))
        cell_assign_dict = {}   # cell_id -> assign
        for major in spatial_major:
            indices = [index for index, value in enumerate(assign_result) if value == major]
            num = len(indices)
            minor_list = [minor for minor in minor_type_spatial_list if minor in type_mapping_dict[major]]
            random.shuffle(minor_list)
            major_cell_id = [spatial_cell_id[i] for i in indices]
            for idx, cell_id in enumerate(major_cell_id):
                cell_assign_dict[cell_id] = {'assign': minor_list[idx]}

        # do qc and reassign cell types
        if cbi_threshold is not None:
            he_score = sum(max(p) for p in p_list)
            score_difference = (he_score - best_score) / (best_score + 1e-4)

            if score_difference > cbi_threshold:
                he_pred = [cell_info[cell_id]['type'] for cell_id in spatial_cell_id]
                true_indexes = [i for i, (a, b) in enumerate(zip(he_pred, assign_result)) if a == b]
                updated_assign_result = [i for i in assign_result]
                updated_assign_result, changed_indexes = qc(updated_assign_result, he_pred, p_list, he_score, best_score, cbi_threshold)

                # reassign cell types
                for diff_idx in changed_indexes:
                    softmax_prob = p_list[diff_idx]
                    major_type = updated_assign_result[diff_idx]
                    true_major_type = [updated_assign_result[i] for i in true_indexes]
                    tmp_idx = [index for index, value in enumerate(true_major_type) if value == major_type]
                    true_same_major_indexes = [true_indexes[i] for i in tmp_idx]
                    true_p_list = [p_list[i] for i in true_same_major_indexes]

                    if len(true_p_list) > 0:
                        top_indices = find_most_similar_sample([softmax_prob], true_p_list, k=top_k, metric=metric)
                        assign_types = [cell_assign_dict[spatial_cell_id[true_same_major_indexes[idx]]]['assign'] for idx in top_indices]
                        assign_counter = Counter(assign_types)
                        # update cell info
                        cell_assign_dict[spatial_cell_id[diff_idx]]['assign'] = assign_counter.most_common(1)[0][0]
                    else:
                        if major_type == 'Dead':
                            cell_id = spatial_cell_id[diff_idx]
                            assign_type = CelltypeAssign._replace_unwanted_type(cell_info[cell_id], nuclei_seg_types, type_mapping_dict)
                        elif major_type == 'Epi' and 'Epi' not in type_mapping_dict.keys() or \
                            major_type == 'Epi' and len(type_mapping_dict['Epi']) == 0:
                            assign_type = random.choice(type_mapping_dict['Neo'])
                        else:
                            assign_type = random.choice(type_mapping_dict[major_type])
                        cell_assign_dict[spatial_cell_id[diff_idx]]['assign'] = assign_type

        # random guess
        random.shuffle(minor_type_spatial_list)
        for idx, cell_id in enumerate(spatial_cell_id):
            if cell_id not in cell_assign_dict:
                cell_assign_dict[cell_id] = {}
            cell_assign_dict[cell_id]['random_guess'] = minor_type_spatial_list[idx]

        return cell_assign_dict


    @staticmethod  
    def _process_interval(interval, spatial_intervals, spatial_positions, 
                          spatial_info, cell_info, nuclei_seg_types,
                          spot_shape, type_mapping_dict, metric, top_k):
        # calculate neighbor positions
        x, y = spatial_intervals.loc[interval, ['x', 'y']].astype(int)
        u_bound = spatial_intervals['x'].max()
        b_bound = spatial_intervals['y'].max()

        if spot_shape == 'circle':
            if y % 2 == 0:
                neighbors = [(x, y), (min(x + 2, u_bound), y), (min(x + 1, u_bound), min(y + 1, b_bound))]
            else:
                neighbors = [(min(x + 1, u_bound), y), (x, min(y + 1, b_bound)), (min(x + 2, u_bound), min(y + 1, b_bound))]
        elif spot_shape == 'square':
            neighbors = [(x, min(y, u_bound)), (min(x+1, b_bound), y), 
                         (x, min(y+1, u_bound)), (min(x+1, b_bound), min(y+1, u_bound))]

        # confirm valid neighbors
        spatial_pos_coords = spatial_positions[['x', 'y']].values
        valid_neighbors = []

        for nx, ny in neighbors:
            mask = (spatial_pos_coords[:, 0] == nx) & (spatial_pos_coords[:, 1] == ny)
            valid_spots = spatial_positions.index[mask].tolist()
            valid_neighbors.extend(valid_spots)

        # obtain neighbor cells from valid neighbor spots
        near_cells = []
        for neighbor in valid_neighbors:
            if neighbor in spatial_info:
                near_cells.extend(spatial_info[neighbor]['cell_id'])

        cell_assign_dict = {}
        interval_cells = spatial_info[interval]['cell_id']

        if near_cells and interval_cells:
            near_cells_probs = np.array([cell_info[cell]['type_prob'] for cell in near_cells])

            for cell in interval_cells:
                cell_prob = cell_info[cell]['type_prob']
                top_indexes = find_most_similar_sample(cell_prob, near_cells_probs, k=top_k, metric=metric)

                assign_types = [cell_info[near_cells[idx]]['assign'] for idx in top_indexes]
                random_types = [cell_info[near_cells[idx]]['random_guess'] for idx in top_indexes]

                assign_counter = Counter(assign_types)
                random_counter = Counter(random_types)

                cell_assign_dict[cell] = {
                    'assign': assign_counter.most_common(1)[0][0],
                    'random_guess': random_counter.most_common(1)[0][0]
                }
        else:
            # no neighbor
            for cell in interval_cells:
                major_type = cell_info[cell]['type']
                if major_type == 'Dead':
                    assign_type = CelltypeAssign._replace_unwanted_type(cell_info[cell], nuclei_seg_types, type_mapping_dict)
                elif major_type == 'Epi' and ('Epi' not in type_mapping_dict or not type_mapping_dict['Epi']):
                    assign_type = random.choice(type_mapping_dict['Neo'])
                else:
                    assign_type = random.choice(type_mapping_dict[major_type])

                cell_assign_dict[cell] = {
                    'assign': assign_type,
                    'random_guess': assign_type
                }

        return cell_assign_dict
    