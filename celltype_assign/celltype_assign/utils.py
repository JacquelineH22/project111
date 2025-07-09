import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon

import itertools


def prop2count(proportion_list, total):
    # proportion * total -> integer
    raw_values = np.array(proportion_list) * total
    integer_values = np.floor(raw_values).astype(int)
    
    # calculate the differences
    extra = (total - np.sum(integer_values)).astype(int)
    remainders = raw_values - integer_values
    indices = np.argsort(remainders)[::-1]  # in descending order

    # rearrange the extra cells
    for i in range(extra):
        integer_values[indices[i]] += 1

    return integer_values.tolist()


def updatemax(f, d, start_state, try_direction, p, o, n_major_type):
    to_state = tuple(start_state[i] + o[try_direction][i] for i in range(n_major_type))
    if to_state not in f:
        return
    if f[start_state] + p > f[to_state]:
        f[to_state] = f[start_state] + p  # update score
        d[to_state] = try_direction

    return


def assign(major_cell_counts, p_list, o, n_major_type, major_type_list):
    n_cell = sum(major_cell_counts)
    # init state space
    f = {}  # scores in each state
    d = {}  # directions to next state
    state_space = itertools.product(*[range(c + 1) for c in major_cell_counts])
    for state in state_space:
        f[state] = 0
        d[state] = -1

    # dynamic programming
    for n in range(n_cell):  # n-th cell
        for state in list(f.keys()):
            if sum(state) != n:  # sum of cells assigned
                continue
            for try_direction in range(n_major_type):
                updatemax(f, d, state, try_direction, p_list[n][try_direction], o, n_major_type)

    # backtracking
    state = tuple(major_cell_counts)
    result = []
    while True:
        result.insert(0, d[state])
        state = tuple(state[i] - o[d[state]][i] for i in range(n_major_type))
        if state == (0,) * n_major_type:
            break

    return f[tuple(major_cell_counts)], [major_type_list[i] for i in result]


def qc(updated_assign_result, he_pred, p_list, he_score, best_score, cbi_threshold):
    # cells had different types with he prediction
    diff_indexes = [i for i, (a, b) in enumerate(zip(he_pred, updated_assign_result)) if a != b]
    diff_p_list = [p_list[i] for i in diff_indexes]

    # he prediction probabilities and its index
    max_p_and_index = [(i, max(sublist)) for i, sublist in enumerate(diff_p_list)]
    max_p_and_index = sorted(max_p_and_index, key=lambda x: x[1], reverse=True)

    sort_idx = 0
    changed_indexes = []
    # modify assignment results until the score difference is less than threshold
    while ((he_score - best_score) / best_score) > cbi_threshold:
        if sort_idx < len(diff_indexes):
            best_score += max_p_and_index[sort_idx][1]
            diff_idx = max_p_and_index[sort_idx][0]
            cell_id = diff_indexes[diff_idx]
            changed_indexes.append(cell_id)
            updated_assign_result[cell_id] = he_pred[cell_id]
            sort_idx += 1
        else:
            break

    return updated_assign_result, changed_indexes


def compute_cross_entropy(p, q):
    p = np.array(p)
    q = np.array(q)

    return -np.sum(p * np.log(q + 1e-10))


def find_most_similar_sample(unknown_sample, known_samples, k=5, metric='ce'):
    if metric == 'cosine':
        similarities = cosine_similarity([unknown_sample], known_samples)
        top_k_indices = np.argsort(similarities[0])[-min(k, len(similarities[0])):][::-1]
    elif metric in ['kl', 'js', 'ce']:
        distances = []
        for i, known_sample in enumerate(known_samples):
            if metric == 'kl':
                distance = np.sum(kl_div(unknown_sample, known_sample))
            elif metric == 'js':
                distance = jensenshannon(unknown_sample, known_sample)
            elif metric == 'cosine':
                distance = 1 - cosine_similarity([unknown_sample], [known_sample])[0][0]
            elif metric == 'ce':
                distance = compute_cross_entropy(unknown_sample, known_sample)
            distances.append((i, distance))
        # choose top k most similar samples
        distances.sort(key=lambda x: x[1])
        top_k_indices = [idx for idx, _ in distances[:k]]
    else:
        raise ValueError("Unsupported metric. Choose from 'kl', 'js', 'ce' or 'cosine'.")

    return top_k_indices
