import importlib

import numpy as np
from tqdm import tqdm

import constants as const
import indexes
import mapping_aktary as mapping
from functions import mapping_functions as mf

mapping = importlib.reload(mapping)
const = importlib.reload(const)
indexes = importlib.reload(indexes)
mf = importlib.reload(mf)


# %%
def get_initial_n_surface_facets(local_chain_length_avg):
    n_surface_facets = np.zeros(np.shape(local_chain_length_avg))

    for i in range(np.shape(local_chain_length_avg)[0]):
        n_surface_facets[i, 0] = 1

    return n_surface_facets


def get_development_rates(local_chain_length_avg):
    # atoda1979.pdf
    S0 = 51  # A/min
    beta = 3.59e+8  # A/min
    alpha = 1.42

    # greeneich1975.pdf MIBK:IPA 1:1
    # S0 = 0
    # beta = 6.645e+6
    # alpha = 1.188

    # greeneich1975.pdf MIBK:IPA 1:3
    # S0 = 0
    # beta = 9.332e+14  # 22.8 C
    # beta = 1.046e+16  # 32.8 C
    # alpha = 3.86

    development_rates = np.zeros(np.shape(local_chain_length_avg))

    for i in range(np.shape(local_chain_length_avg)[0]):
        for j in range(np.shape(local_chain_length_avg)[1]):
            development_rates[i, j] = S0 + beta / ((local_chain_length_avg[i, j] * const.u_MMA) ** alpha)

    return development_rates


def get_development_time_factor(n_surface_facets):
    return np.sqrt(n_surface_facets)


def update_n_surface_facets(development_times, n_surface_facets):
    for i in range(np.shape(development_times)[0]):
        for j in range(np.shape(development_times)[1]):

            if development_times[i, j] == 0:
                n_surface_facets[i, j] = 0
                continue

            now_n_surface_facets = 0

            if i - 1 >= 0 and development_times[i - 1, j] == 0:
                now_n_surface_facets += 1
            if i + 1 < np.shape(development_times)[0] and development_times[i + 1, j] == 0:
                now_n_surface_facets += 1

            if j == 0:
                now_n_surface_facets += 1
            if j - 1 >= 0 and development_times[i, j - 1] == 0:
                now_n_surface_facets += 1
            if j + 1 < np.shape(development_times)[1] and development_times[i, j + 1] == 0:
                now_n_surface_facets += 1

            n_surface_facets[i, j] = now_n_surface_facets


def transfer_overkill(development_times, i, j, overkill):  # overkill is negative
    neighbour_inds = []

    for di in range(-1, 2):
        for dj in range(-1, 2):

            if np.abs(di) == np.abs(dj):
                continue
            if not 0 <= i + di < np.shape(development_times)[0]:
                continue
            if not 0 <= j + dj < np.shape(development_times)[1]:
                continue
            if development_times[i + di, j + dj] > 0:
                neighbour_inds.append([i + di, j + dj])

    if len(neighbour_inds) == 0:
        print('can\'t share overkill')

    for neigh_inds in neighbour_inds:
        neigh_i, neigh_j = neigh_inds
        development_times[neigh_i, neigh_j] += overkill / len(neighbour_inds)

    # return development_times


def share_all_overkills(development_times):
    while True:
        negative_inds = np.where(development_times < 0)
        if len(negative_inds[0]) == 0:
            return

        for neg_inds in np.array(negative_inds).transpose():
            i, j = neg_inds
            overkill = development_times[i, j]
            development_times[i, j] = 0
            transfer_overkill(development_times, i, j, overkill)


def make_develop_step(development_times, n_surface_facets, delta_t):

    for j in range(np.shape(development_times)[1]):
        for i in range(np.shape(development_times)[0]):

            negative_inds = np.array(np.where(development_times < 0))
            if len(negative_inds[0]) != 0:
                print('negative times exist')

            now_development_time = development_times[i, j]
            now_n_surface_facets = n_surface_facets[i, j]

            if now_n_surface_facets == 0 or now_development_time == 0:
                continue

            effective_delta_t = delta_t * get_development_time_factor(now_n_surface_facets)
            new_development_time = now_development_time - effective_delta_t
            development_times[i, j] = new_development_time
            share_all_overkills(development_times)
            update_n_surface_facets(development_times, n_surface_facets)
