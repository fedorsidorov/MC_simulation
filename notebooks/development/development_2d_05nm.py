import importlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import mapping_aktary as mapping
from functions import development_functions_2d as df

mapping = importlib.reload(mapping)
df = importlib.reload(df)

# %% 0.5nm histograms
sum_lens_matrix = np.load('data/chains/Aktary/development/sum_lens_matrix_series_1_05nm_1500.npy')
n_chains_matrix = np.load('data/chains/Aktary/development/n_chains_matrix_series_1_05nm_1500.npy')

n_chains_matrix_avg = np.average(n_chains_matrix, axis=1)
sum_lens_matrix_avg = np.average(sum_lens_matrix, axis=1)

local_chain_length_avg = sum_lens_matrix_avg / n_chains_matrix_avg

# plt.imshow(sum_lens_matrix_avg.transpose())
plt.imshow(local_chain_length_avg.transpose())
# plt.imshow(np.log10(local_chain_length_avg).transpose())
plt.colorbar()
plt.show()

# %%
bin_size = mapping.step_05nm * 10  # A

# greeneich1975.pdf MIBK:IPA 1:3
S0, alpha, beta = 0, 3.86, 9.332e+14  # 22.8 C, Han

development_rates = df.get_development_rates(local_chain_length_avg, S0, alpha, beta)
development_times = bin_size / development_rates
n_surface_facets = df.get_initial_n_surface_facets(local_chain_length_avg)

# plt.imshow(development_times.transpose())
plt.imshow(np.log(development_times).transpose())
plt.show()

# %%
n_seconds = 10
factor = 100

delta_t = 1 / 60 / factor
n_steps = n_seconds * factor

progress_bar = tqdm(total=n_steps, position=0)

for i in range(n_steps):
    df.make_develop_step(development_times, n_surface_facets, delta_t)
    progress_bar.update()

np.save('data/chains/Aktary/development/n_surface_facets_series_1_1500_05nm.npy', n_surface_facets)

# %%
plt.figure(dpi=300)
plt.imshow(n_surface_facets.transpose())
plt.colorbar()
plt.show()

# %%
# np.save('data/chains/Aktary/development/n_surface_facets.npy', n_surface_facets)
