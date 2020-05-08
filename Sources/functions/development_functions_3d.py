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
def get_initial_n_surface_facets(sum_lens_matrix):
    n_surface_facets = np.zeros(np.shape(sum_lens_matrix))

    for i in range(np.shape(sum_lens_matrix)[0]):
        for j in range(np.shape(sum_lens_matrix)[1]):
            n_surface_facets[i, j, 0] = 1

    return n_surface_facets


def get_development_rates(local_chain_length_avg, y_shape):
    S0 = 51  # A/min
    beta = 3.59e+8  # A/min
    alpha = 1.42
    development_rates = np.zeros((np.shape(local_chain_length_avg)[0], y_shape, np.shape(local_chain_length_avg)[1]))

    for i in range(np.shape(local_chain_length_avg)[0]):
        for j in range(y_shape):
            for k in range(np.shape(local_chain_length_avg)[1]):
                development_rates[i, j, k] = S0 + beta / ((local_chain_length_avg[i, k] * const.u_MMA) ** alpha)  # i, k

    return development_rates


def get_development_time_factor(n_surface_facets):
    return np.sqrt(n_surface_facets)


def update_n_surface_facets(development_times, n_surface_facets):
    for i in range(np.shape(development_times)[0]):
        for j in range(np.shape(development_times)[0]):
            for k in range(np.shape(development_times)[0]):

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


def make_develop_step(development_times, n_surface_facets, delta_t):
    progress_bar = tqdm(total=np.shape(development_times)[0], position=0)

    for i in range(np.shape(development_times)[0]):
        # for j in range(np.shape(development_times)[1]):
        for j in range(1):

            progress_bar.update()

            for k in range(np.shape(development_times)[2]):

                # progress_bar.update()

                # if i == 25:
                #     print(i)

                now_development_time = development_times[i, j, k]
                now_n_surface_facets = n_surface_facets[i, j, k]

                if now_n_surface_facets == 0 or now_development_time == 0:
                    continue

                if now_development_time < 0:
                    new_development_time = 0
                    overkill = - now_development_time
                    neighbour_inds = []

                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            for dk in range(-1, 2):

                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                if development_times[i + di, j + dj, k + dk] > 0:
                                    neighbour_inds.append([i + di, j + dj, k + dk])

                    for inds in neighbour_inds:
                        development_times[inds] -= overkill / len(neighbour_inds)

                    development_times[i, j, k] = new_development_time
                    update_n_surface_facets(development_times, n_surface_facets)
                    continue

                effective_delta_t = delta_t * get_development_time_factor(now_n_surface_facets)
                new_development_time = now_development_time - effective_delta_t

                if new_development_time < 0:

                    new_development_time = 0
                    overkill = effective_delta_t - now_development_time
                    neighbour_inds = []

                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            for dk in range(-1, 2):

                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                if development_times[i + di, j + dj, k + dk] > 0:
                                    neighbour_inds.append([i + di, j + dj, k + dk])

                    for inds in neighbour_inds:
                        development_times[inds] -= overkill / len(neighbour_inds)

                development_times[i, j, k] = new_development_time
                update_n_surface_facets(development_times, n_surface_facets)

        # progress_bar.update()
