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
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/development/sum_lens_matrix.npy')
n_chains_matrix = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/development/n_chains_matrix.npy')

n_chains_matrix_avg = np.average(n_chains_matrix, axis=1)
sum_lens_matrix_avg = np.average(sum_lens_matrix, axis=1)
local_chain_length_avg = sum_lens_matrix_avg / n_chains_matrix_avg

bin_size = mapping.step_4nm * 10  # A

development_rates = df.get_development_rates(local_chain_length_avg)
development_times = bin_size / development_rates
n_surface_facets = df.get_initial_n_surface_facets(local_chain_length_avg)

# %%
delta_t = 1 / 10
n_steps = 60

progress_bar = tqdm(total=n_steps, position=0)

for i in range(n_steps):
    df.make_develop_step(development_times, n_surface_facets, delta_t)
    progress_bar.update()


# %%
plt.figure(dpi=300)
plt.imshow(development_times.transpose())
plt.colorbar()
plt.show()
