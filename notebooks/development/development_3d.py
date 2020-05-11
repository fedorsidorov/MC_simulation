import importlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import constants as const
import indexes
import mapping_aktary as mapping
from functions import development_functions_3d as df
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

# sum_lens_matrix = np.load(
#     '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/development/sum_lens_matrix_series_2_4nm.npy')
# n_chains_matrix = np.load(
#     '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/development/n_chains_matrix_series_2_4nm.npy')

bin_size = mapping.step_2nm * 10  # A
# bin_size = mapping.step_4nm * 10  # A

development_rates = df.get_development_rates(sum_lens_matrix, n_chains_matrix)
development_times = bin_size / development_rates
n_surface_facets = df.get_initial_n_surface_facets(development_times)

# plt.imshow(np.log(np.average(development_times, axis=1)).transpose())
# plt.show()

# %%
# delta_t = 1 / 60 / 10
delta_t = 1 / 60
n_steps = 15

# progress_bar = tqdm(total=n_steps, position=0)

for i in range(n_steps):
    print(i)
    df.make_develop_step(development_times, n_surface_facets, delta_t)
    # progress_bar.update()

# progress_bar.close()

np.save('data/chains/Aktary/development/n_surface_facets_15s.npy', n_surface_facets)

# %%
plt.figure(dpi=300)
# plt.imshow(development_times.transpose())
plt.imshow(n_surface_facets[:, 5, :].transpose())
plt.colorbar()
plt.show()

# %%
# np.save('data/chains/Aktary/development/n_surface_facets_4nm_15s.npy', n_surface_facets)
