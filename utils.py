import numpy as np
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity

import itertools


def process_spot(spot, spot_size, spatial_loc, cell_info, *args):

    if spot_size:
        x_min = min(spatial_loc.loc[spot, 'pixel_x'], spatial_loc.loc[spot, 'pixel_x'] + spot_size[1])
        x_max = max(spatial_loc.loc[spot, 'pixel_x'], spatial_loc.loc[spot, 'pixel_x'] + spot_size[1])
        y_min = min(spatial_loc.loc[spot, 'pixel_y'], spatial_loc.loc[spot, 'pixel_y'] + spot_size[0])
        y_max = max(spatial_loc.loc[spot, 'pixel_y'], spatial_loc.loc[spot, 'pixel_y'] + spot_size[0])
    else: 
        r = args[0]
        x_center = spatial_loc.loc[spot, 'pixel_x']
        y_center = spatial_loc.loc[spot, 'pixel_y']

    spot_cell_id = []
    assigns = []
    centroids = []
    guesses = []
    model_preds = []

    # use numpy array for faster processing
    cell_ids = np.array(list(cell_info.keys()))
    centroid_array = np.array([cell['centroid'] for cell in cell_info.values()])
    types = np.array([cell['type'] for cell in cell_info.values()])
    assigns_arr = np.array([cell['assign'] for cell in cell_info.values()])
    guesses_arr = np.array([cell['random_guess'] for cell in cell_info.values()])

    # select cells within the spot
    if spot_size:
        mask = (centroid_array[:, 0] >= x_min) & (centroid_array[:, 0] <= x_max) & \
            (centroid_array[:, 1] >= y_min) & (centroid_array[:, 1] <= y_max)
    else:
        distances = np.sqrt((centroid_array[:, 0] - x_center) ** 2 + (centroid_array[:, 1] - y_center) ** 2)
        mask = distances < r

    # filter the cells
    filtered_cell_ids = cell_ids[mask]
    filtered_centroids = centroid_array[mask]
    filtered_types = types[mask]
    filtered_assigns = assigns_arr[mask]
    filtered_guesses = guesses_arr[mask]

    spot_cell_id.extend(filtered_cell_ids)
    centroids.extend(filtered_centroids.tolist())
    assigns.extend(filtered_assigns.tolist())
    guesses.extend(filtered_guesses.tolist())
    model_preds.extend(filtered_types.tolist())
    
    return spot, {'cell_id': spot_cell_id, 'centroid': centroids, 'assign': assigns, 'guess': guesses, 'model_pred': model_preds}


def updatemax(f, d, start_state, try_direction, p, o, type_num):
    to_state = tuple(start_state[i] + o[try_direction][i] for i in range(type_num))
    if to_state not in f:
        return
    if f[start_state] + p > f[to_state]:
        f[to_state] = f[start_state] + p
        d[to_state] = try_direction


def cell_assign(counts, P, o, type_num, type_list):
    N = sum(counts)
    assert len(P) == N
    f = {}
    d = {}
    
    state_space = itertools.product(*[range(c + 1) for c in counts])
    for state in state_space:
        f[state] = 0
        d[state] = -1

    for n in range(N):
        for state in list(f.keys()):
            if sum(state) != n:
                continue
            for try_direction in range(type_num):
                updatemax(f, d, state, try_direction, P[n][try_direction], o, type_num)

    state = tuple(counts)
    result = []
    while True:
        result.insert(0, d[state])
        state = tuple(state[i] - o[d[state]][i] for i in range(type_num))
        if state == (0,) * type_num:
            break

    return f[tuple(counts)], [type_list[i] for i in result]


def prop2count(proportion_list, total, filter_thres = 0.1):
    raw_values = np.array(proportion_list) * total
    integer_values = np.floor(raw_values).astype(int)
    difference = (total - np.sum(integer_values)).astype(int)
    remainders = raw_values - integer_values
    indices = np.argsort(remainders)[::-1]
    
    for i in range(difference):
        integer_values[indices[i]] += 1
    
    return integer_values.tolist()


def compute_cross_entropy(p, q):
    p = np.array(p)
    q = np.array(q)
    return -np.sum(p * np.log(q + 1e-10))


def find_most_similar_sample(unknown_sample, known_samples, k=5, metric="ce"):

    distances = []
    for i, known_sample in enumerate(known_samples):
        if metric == "kl":
            distance = np.sum(kl_div(unknown_sample, known_sample))
        elif metric == "js":
            distance = jensenshannon(unknown_sample, known_sample)
        elif metric == "cosine":
            distance = 1 - cosine_similarity([unknown_sample], [known_sample])[0][0]
        elif metric == "ce":
            distance = compute_cross_entropy(unknown_sample, known_sample)
        else:
            raise ValueError("Unsupported metric. Choose from 'kl', 'js', 'ce' or 'cosine'.")
        distances.append((i, distance))

    distances.sort(key=lambda x: x[1])
    top_k_indices = [idx for idx, _ in distances[:k]]
    return top_k_indices
