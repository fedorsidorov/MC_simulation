import importlib
from collections import deque

import numpy as np
from tqdm import tqdm

import constants as const

const = importlib.reload(const)


# %%
def get_initial_n_surface_facets(local_chain_length):
    n_surface_facets = np.zeros(np.shape(local_chain_length))

    for i in range(np.shape(local_chain_length)[0]):
        for j in range(np.shape(local_chain_length)[1]):
            n_surface_facets[i, j, 0] = 1

    return n_surface_facets


def get_local_chain_lengths(sum_lens_matrix, n_chains_matrix):
    local_chain_length_avg = np.average(sum_lens_matrix, axis=1) / np.average(n_chains_matrix, axis=1)
    local_chain_length = np.zeros(np.shape(sum_lens_matrix))

    for i in range(np.shape(sum_lens_matrix)[0]):
        for j in range(np.shape(sum_lens_matrix)[1]):
            for k in range(np.shape(sum_lens_matrix)[2]):

                if n_chains_matrix[i, j, k] == 0:
                    local_chain_length[i, j, k] = local_chain_length_avg[i, k]
                else:
                    local_chain_length[i, j, k] = sum_lens_matrix[i, j, k] / n_chains_matrix[i, j, k]

    return local_chain_length


def get_development_rates(sum_lens_matrix, n_chains_matrix, S0, alpha, beta):
    local_chain_length_avg = np.average(sum_lens_matrix, axis=1) / np.average(n_chains_matrix, axis=1)
    development_rates = np.zeros(np.shape(sum_lens_matrix))

    for i in range(np.shape(sum_lens_matrix)[0]):
        for j in range(np.shape(sum_lens_matrix)[1]):
            for k in range(np.shape(sum_lens_matrix)[2]):

                if n_chains_matrix[i, j, k] == 0:
                    now_local_chain_length = local_chain_length_avg[i, k]
                else:
                    now_local_chain_length = sum_lens_matrix[i, j, k] / n_chains_matrix[i, j, k]

                development_rates[i, j, k] = S0 + beta / ((now_local_chain_length * const.u_MMA) ** alpha)

    return development_rates


def get_development_time_factor(n_surface_facets):
    return np.sqrt(n_surface_facets)


def update_n_surface_facets(development_times, n_surface_facets):
    for i in range(np.shape(development_times)[0]):
        for j in range(np.shape(development_times)[1]):
            for k in range(np.shape(development_times)[2]):

                if development_times[i, j, k] == 0:
                    n_surface_facets[i, j, k] = 0
                    continue

                now_n_surface_facets = 0

                if i - 1 >= 0 and development_times[i - 1, j, k] == 0:
                    now_n_surface_facets += 1
                if i + 1 < np.shape(development_times)[0] and development_times[i + 1, j, k] == 0:
                    now_n_surface_facets += 1

                if j - 1 >= 0 and development_times[i, j - 1, k] == 0:
                    now_n_surface_facets += 1
                if j + 1 < np.shape(development_times)[1] and development_times[i, j + 1, k] == 0:
                    now_n_surface_facets += 1

                if k == 0:
                    now_n_surface_facets += 1
                if k - 1 >= 0 and development_times[i, j, k - 1] == 0:
                    now_n_surface_facets += 1
                if k + 1 < np.shape(development_times)[2] and development_times[i, j, k + 1] == 0:
                    now_n_surface_facets += 1

                n_surface_facets[i, j, k] = now_n_surface_facets


def transfer_overkill(development_times, n_surface_facets, i, j, k, overkill):  # overkill is negative
    neighbour_inds = deque()

    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):

                if np.abs(di) == np.abs(dj) == 1 or np.abs(dj) == np.abs(dk) == 1 or np.abs(di) == np.abs(dk) == 1:
                    continue
                if not 0 <= i + di < np.shape(development_times)[0]:
                    continue
                if not 0 <= j + dj < np.shape(development_times)[1]:
                    continue
                if not 0 <= k + dk < np.shape(development_times)[2]:
                    continue
                if development_times[i + di, j + dj, k + dk] > 0:
                    neighbour_inds.append([i + di, j + dj, k + dk])

    if len(neighbour_inds) == 0:
        print('can\'t share overkill')

    for neigh_inds in neighbour_inds:
        neigh_i, neigh_j, neigh_k = neigh_inds
        neigh_development_time_factor = get_development_time_factor(n_surface_facets[neigh_i, neigh_j, neigh_k])
        development_times[neigh_i, neigh_j, neigh_k] += overkill / len(neighbour_inds) * neigh_development_time_factor


def share_all_overkills(development_times, n_surface_facets):
    while True:
        negative_inds = np.where(development_times < 0)
        if len(negative_inds[0]) == 0:
            return

        for neg_inds in np.array(negative_inds).transpose():
            neg_i, neg_j, neg_k = neg_inds
            overkill = development_times[neg_i, neg_j, neg_k]
            development_times[neg_i, neg_j, neg_k] = 0
            transfer_overkill(development_times, n_surface_facets, neg_i, neg_j, neg_k, overkill)


def make_develop_step(development_times, n_surface_facets, delta_t, j_range=range(0, 50)):
    for k in range(np.shape(development_times)[2]):
        for i in range(np.shape(development_times)[0]):
            # for j in range(np.shape(development_times)[1]):
            for j in j_range:

                negative_inds = np.array(np.where(development_times < 0))
                if len(negative_inds[0]) != 0:
                    print('negative times exist')

                now_development_time = development_times[i, j, k]
                now_n_surface_facets = n_surface_facets[i, j, k]

                if now_n_surface_facets == 0 or now_development_time == 0:
                    continue

                effective_delta_t = delta_t * get_development_time_factor(now_n_surface_facets)
                new_development_time = now_development_time - effective_delta_t
                development_times[i, j, k] = new_development_time
                share_all_overkills(development_times, n_surface_facets)
                update_n_surface_facets(development_times, n_surface_facets)
