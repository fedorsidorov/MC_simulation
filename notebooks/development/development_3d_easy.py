import importlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import constants as const
import indexes
import mapping_aktary as mapping
from functions import development_functions_2d as df
from functions import mapping_functions as mf

mapping = importlib.reload(mapping)
const = importlib.reload(const)
indexes = importlib.reload(indexes)
mf = importlib.reload(mf)
df = importlib.reload(df)

# %% 2nm histograms
sum_lens_matrix = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/development/sum_lens_matrix_series_2.npy')
n_chains_matrix = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/development/n_chains_matrix_series_2.npy')

sum_lens_matrix_avg = np.average(sum_lens_matrix, axis=1)
n_chains_matrix_avg = np.average(n_chains_matrix, axis=1)

local_chain_length_avg = np.average(sum_lens_matrix, axis=1) / np.average(n_chains_matrix, axis=1)
local_chain_length = np.zeros(np.shape(sum_lens_matrix))

for i in range(np.shape(sum_lens_matrix)[0]):
    for j in range(np.shape(sum_lens_matrix)[1]):
        for k in range(np.shape(sum_lens_matrix)[2]):

            if n_chains_matrix[i, j, k] == 0:
                local_chain_length[i, j, k] = local_chain_length_avg[i, k]
            else:
                local_chain_length[i, j, k] = sum_lens_matrix[i, j, k] / n_chains_matrix[i, j, k]

# plt.imshow(local_chain_length[:, 0, :].transpose())
# plt.show()

development_rates = np.zeros(np.shape(sum_lens_matrix))
development_times = np.zeros(np.shape(sum_lens_matrix))
n_surface_facets = np.zeros(np.shape(sum_lens_matrix))

for j in range(np.shape(sum_lens_matrix)[1]):
    development_rates[:, j, :] = df.get_development_rates(local_chain_length[:, j, :])
    development_times[:, j, :] = mapping.step_2nm * 10 / development_rates[:, j, :]
    n_surface_facets[:, j, :] = df.get_initial_n_surface_facets(local_chain_length[:, j, :])

# plt.imshow(development_times[:, 4, :].transpose())
# plt.show()

# %%
delta_t = 1 / 60 / 2
n_steps = 10


for i in range(n_steps):

    print('step #' + str(i), '\n')
    progress_bar = tqdm(total=np.shape(sum_lens_matrix)[1], position=0)

    for j in range(np.shape(sum_lens_matrix)[1]):
        df.make_develop_step(development_times[:, j, :], n_surface_facets[:, j, :], delta_t)
        progress_bar.update()

    progress_bar.close()

# %%
plt.figure(dpi=300)
# plt.imshow(development_times.transpose())
plt.imshow(np.average(n_surface_facets, axis=1).transpose())
plt.colorbar()
plt.show()
