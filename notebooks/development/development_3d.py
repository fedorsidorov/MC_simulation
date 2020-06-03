import importlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import mapping_aktary as mapping
from functions import development_functions_3d as df

mapping = importlib.reload(mapping)
df = importlib.reload(df)

# 4nm histograms
sum_lens_matrix = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/combine_chains/development/sum_lens_matrix_series_1_4nm_1500.npy')
n_chains_matrix = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/combine_chains/development/n_chains_matrix_series_1_4nm_1500.npy')

lens_avg = np.average(sum_lens_matrix, axis=1)
chains_avg = np.average(n_chains_matrix, axis=1)

bin_size = mapping.step_4nm * 10  # A
S0, alpha, beta = 0, 3.86, 9.332e+14  # 22.8 C, Han

local_chain_length = df.get_local_chain_lengths(sum_lens_matrix, n_chains_matrix)
development_rates = df.get_development_rates(sum_lens_matrix, n_chains_matrix, S0, alpha, beta)
development_times = bin_size / development_rates
n_surface_facets = df.get_initial_n_surface_facets(development_times)

# plt.imshow(np.log(np.average(local_chain_length, axis=1)).transpose())
# plt.show()

# %%
n_seconds = 10
factor = 10

delta_t = 1 / 60 / factor
n_steps = n_seconds * factor

progress_bar = tqdm(total=n_steps, position=0)

for i in range(n_steps):
    df.make_develop_step(development_times, n_surface_facets, delta_t, j_range=range(10, 15))
    progress_bar.update()

np.save('data/chains/combine_chains/development/n_surface_facets_1500_10s.npy', n_surface_facets)

# %%
plt.figure(dpi=300)
# plt.imshow(development_times.transpose())
plt.imshow(n_surface_facets[:, 14, :].transpose())
plt.colorbar()
plt.show()

# %%
np.save('data/chains/combine_chains/development/n_surface_facets_4nm_10s_easy.npy', n_surface_facets)
