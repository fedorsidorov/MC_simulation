import importlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import constants as const
import indexes
# import mapping_harris as mapping
import mapping_aktary as mapping
from functions import mapping_functions as mf

mapping = importlib.reload(mapping)
const = importlib.reload(const)
indexes = importlib.reload(indexes)
mf = importlib.reload(mf)

# %% 2nm histograms
sum_lens_matrix = np.load('data/chains/Aktary/development/sum_lens_matrix.npy')
n_chains_matrix = np.load('data/chains/Aktary/development/n_chains_matrix.npy')

# %% 4nm histograms
sum_lens_matrix_4nm = np.zeros(mapping.hist_4nm_shape)
n_chains_matrix_4nm = np.zeros(mapping.hist_4nm_shape)

resist_shape = mapping.hist_4nm_shape

progress_bar = tqdm(total=resist_shape[0], position=0)

for x_ind in range(resist_shape[0]):
    for y_ind in range(resist_shape[1]):
        for z_ind in range(resist_shape[2]):
            x_inds_2nm = range(x_ind * 2, x_ind * 2 + 1 + 1)
            y_inds_2nm = range(y_ind * 2, y_ind * 2 + 1 + 1)
            z_inds_2nm = range(z_ind * 2, z_ind * 2 + 1 + 1)

            # part_sum_lens = sum_lens_matrix[x_inds_2nm, y_inds_2nm, z_inds_2nm]
            # part_n_chains = n_chains_matrix[x_inds_2nm, y_inds_2nm, z_inds_2nm]

            sum_sum_lens = np.sum(sum_lens_matrix[x_inds_2nm, y_inds_2nm, z_inds_2nm])
            sum_n_chains = np.sum(n_chains_matrix[x_inds_2nm, y_inds_2nm, z_inds_2nm])

            sum_lens_matrix_4nm[x_ind, y_ind, z_ind] = sum_sum_lens
            n_chains_matrix_4nm[x_ind, y_ind, z_ind] = sum_n_chains

    progress_bar.update()

# %%
# sum_lens_matrix_avg = np.average(sum_lens_matrix_4nm, axis=1)
# n_chains_matrix_avg = np.average(n_chains_matrix_4nm, axis=1)

n_chains_matrix_avg = np.average(n_chains_matrix, axis=1)
sum_lens_matrix_avg = np.average(sum_lens_matrix, axis=1)

local_chain_length_avg = sum_lens_matrix_avg / n_chains_matrix_avg

plt.figure(dpi=300)
plt.imshow(local_chain_length_avg.transpose())
plt.colorbar()
plt.show()
