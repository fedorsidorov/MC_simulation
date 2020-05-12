import importlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import mapping_aktary as mapping
from functions import development_functions_2d as df

mapping = importlib.reload(mapping)
df = importlib.reload(df)

# 2nm histograms
sum_lens_matrix = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/development/sum_lens_matrix_series_1_4nm_1500.npy')
n_chains_matrix = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/development/n_chains_matrix_series_1_4nm_1500.npy')

n_chains_matrix_avg = np.average(n_chains_matrix, axis=1)
sum_lens_matrix_avg = np.average(sum_lens_matrix, axis=1)

local_chain_length_avg = sum_lens_matrix_avg / n_chains_matrix_avg

# plt.imshow(local_chain_length_avg.transpose())
# plt.show()

bin_size = mapping.step_4nm * 10  # A

# atoda1979.pdf
# S0, alpha, beta = 51, 1.42, 3.59e+8  # A/min

# greeneich1975.pdf MIBK
# S0, alpha, beta = 84.0, 1.5, 3.14e+8  # 22.8 C
# S0, alpha, beta = 241.9, 1.5, 5.669e+8  # 34.1 C
# S0, alpha, beta = 464.0, 1.5, 1.435e+9  # 35.6 C

# greeneich1975.pdf MIBK:IPA 1:1
# S0, alpha, beta = 0, 2.0, 6.7e+9  # < 5e+3
# S0, alpha, beta = 0, 1.188, 6.645e+6  # > 5e+3

# greeneich1975.pdf MIBK:IPA 1:3
S0, alpha, beta = 0, 3.86, 9.332e+14  # 22.8 C, Han
# S0, alpha, beta = 0, 3.86, 1.046e+16  # 32.8 C

development_rates = df.get_development_rates(local_chain_length_avg, S0, alpha, beta)
development_times = bin_size / development_rates
n_surface_facets = df.get_initial_n_surface_facets(local_chain_length_avg)

# plt.imshow(np.log(development_times).transpose())
# plt.show()

# %%
n_seconds = 10
factor = 100

delta_t = 1 / 60 / factor
n_steps = n_seconds * factor

progress_bar = tqdm(total=n_steps, position=0)

for i in range(n_steps):
    df.make_develop_step(development_times, n_surface_facets, delta_t)
    progress_bar.update()

progress_bar.close()


#
plt.figure(dpi=300)
# plt.imshow(development_times.transpose())
plt.imshow(n_surface_facets.transpose())
plt.colorbar()
plt.show()

# %%
# np.save('data/chains/Aktary/development/n_surface_facets.npy', n_surface_facets)
