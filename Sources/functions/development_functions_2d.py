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
    S0 = 51  # A/min
    beta = 3.59e+8  # A/min
    alpha = 1.42
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


def share_overkill(development_times, i, j, overkill):
    neighbour_inds = []

    for di in range(-1, 2):
        for dj in range(-1, 2):

            if di == 0 and dj == 0:
                continue
            if not 0 <= i + di < np.shape(development_times)[0]:
                continue
            if not 0 <= j + dj < np.shape(development_times)[1]:
                continue

            if development_times[i + di, j + dj] > 0:
                neighbour_inds.append([i + di, j + dj])

    for inds in neighbour_inds:
        development_times[inds] -= overkill / len(neighbour_inds)


def make_develop_step(development_times, n_surface_facets, delta_t):
    progress_bar = tqdm(total=np.shape(development_times)[0], position=0)

    # for i in range(np.shape(development_times)[0]):
    #     for j in range(np.shape(development_times)[1]):

    for j in range(np.shape(development_times)[1]):
        for i in range(np.shape(development_times)[0]):

            now_development_time = development_times[i, j]
            now_n_surface_facets = n_surface_facets[i, j]

            if now_n_surface_facets == 0 or now_development_time == 0:
                continue

            if now_development_time < 0:
                new_development_time = 0
                overkill = - now_development_time
                share_overkill(development_times, i, j, overkill)

                development_times[i, j] = new_development_time
                update_n_surface_facets(development_times, n_surface_facets)
                continue

            effective_delta_t = delta_t * get_development_time_factor(now_n_surface_facets)
            new_development_time = now_development_time - effective_delta_t

            if new_development_time < 0:

                new_development_time = 0
                overkill = effective_delta_t - now_development_time
                share_overkill(development_times, i, j, overkill)

            development_times[i, j] = new_development_time
            update_n_surface_facets(development_times, n_surface_facets)

        progress_bar.update()
