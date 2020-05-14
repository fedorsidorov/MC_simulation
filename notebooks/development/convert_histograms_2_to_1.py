import importlib

import matplotlib.pyplot as plt
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

# %% 2nm histograms
sum_lens_matrix_2nm = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/development/sum_lens_matrix_series_1_2nm_1500.npy')
n_chains_matrix_2nm = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/development/n_chains_matrix_series_1_2nm_1500.npy')

hist_2nm = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/best_sh_sn_chains/best_hist_2nm.npy')

# %% 4nm histograms
sum_lens_matrix_1nm = np.zeros(mapping.hist_1nm_shape)
n_chains_matrix_1nm = np.zeros(mapping.hist_1nm_shape)

resist_shape_2nm = mapping.hist_2nm_shape

progress_bar = tqdm(total=resist_shape_2nm[0], position=0)

for x_ind in range(resist_shape_2nm[0]):
    for y_ind in range(resist_shape_2nm[1]):
        for z_ind in range(resist_shape_2nm[2]):

            x_inds_1nm = range(x_ind * 2, x_ind * 2 + 1 + 1)
            y_inds_1nm = range(y_ind * 2, y_ind * 2 + 1 + 1)
            z_inds_1nm = range(z_ind * 2, z_ind * 2 + 1 + 1)

            for x_ind_1nm in x_inds_1nm:
                for y_ind_1nm in y_inds_1nm:
                    for z_ind_1nm in z_inds_1nm:

                        sum_lens_matrix_1nm[x_ind_1nm, y_ind_1nm, z_ind_1nm] = sum_lens_matrix_2nm[x_ind, y_ind, z_ind]
                        n_chains_matrix_1nm[x_ind_1nm, y_ind_1nm, z_ind_1nm] = n_chains_matrix_2nm[x_ind, y_ind, z_ind]

    progress_bar.update()

# %%
sum_lens_matrix_avg = np.average(sum_lens_matrix_1nm, axis=1)
n_chains_matrix_avg = np.average(n_chains_matrix_1nm, axis=1)

# n_chains_matrix_avg = np.average(n_chains_matrix, axis=1)
# sum_lens_matrix_avg = np.average(sum_lens_matrix, axis=1)

local_chain_length_avg = sum_lens_matrix_avg / n_chains_matrix_avg

plt.figure(dpi=300)
plt.imshow(local_chain_length_avg.transpose())
# plt.imshow(np.average(hist_2nm, axis=1).transpose())
plt.colorbar()
plt.show()

# %%
np.save('data/chains/Aktary/development/sum_lens_matrix_series_1_1nm_1500_old.npy', sum_lens_matrix_1nm)
np.save('data/chains/Aktary/development/n_chains_matrix_series_1_1nm_1500_old.npy', n_chains_matrix_1nm)
