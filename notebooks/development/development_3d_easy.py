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

# 4nm histograms
sum_lens_matrix = np.load('data/chains/combine_chains/development/sum_lens_matrix_series_1_4nm_1500.npy')
n_chains_matrix = np.load('data/chains/combine_chains/development/n_chains_matrix_series_1_4nm_1500.npy')

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

# greeneich1975.pdf MIBK:IPA 1:3
S0, alpha, beta = 0, 3.86, 9.332e+14  # 22.8 C, Han

for j in range(np.shape(sum_lens_matrix)[1]):
    development_rates[:, j, :] = df.get_development_rates(local_chain_length[:, j, :], S0, alpha, beta)
    development_times[:, j, :] = mapping.step_2nm * 10 / development_rates[:, j, :]
    n_surface_facets[:, j, :] = df.get_initial_n_surface_facets(local_chain_length[:, j, :])

# plt.imshow(development_times[:, 4, :].transpose())
# plt.show()

# %%
n_seconds = 10
factor = 100 * 2

delta_t = 1 / 60 / factor
n_steps = n_seconds * factor

progress_bar = tqdm(total=np.shape(sum_lens_matrix)[1], position=0)

for j in range(np.shape(sum_lens_matrix)[1]):

    for n in range(n_steps):
        df.make_develop_step(development_times[:, j, :], n_surface_facets[:, j, :], delta_t)

    progress_bar.update()

np.save('data/chains/combine_chains/development/n_surface_facets_series_1_4nm_1500_j.npy', n_surface_facets)

# %%
plt.figure(dpi=300)
# plt.imshow(development_times.transpose())
plt.imshow(n_surface_facets[:, 24, :].transpose())
plt.colorbar()
plt.show()

# %%
#
