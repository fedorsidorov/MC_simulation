import importlib

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import constants as cp
import mapping_aktary as mapping

mapping = importlib.reload(mapping)
cp = importlib.reload(cp)


# %% functions
def get_hist_position(element, bins):
    n_bin = (element - bins[0]) // (bins[1] - bins[0])
    return int(n_bin)


# %% load data
folder_name = 'combine_chains'

hist_1nm = np.load('data/chains/combine_chains/best_sh_sn_chains/best_hist_1nm.npy')

plt.imshow(np.average(hist_1nm, axis=2).transpose())
plt.show()

n_chains = 754
n_mon_cell_max = int(np.max(hist_1nm))

# %% create arrays
pos_matrix = np.zeros(mapping.hist_1nm_shape, dtype=np.uint32)
resist_matrix = -np.ones((*mapping.hist_1nm_shape, n_mon_cell_max, 3), dtype=np.uint32)
chain_tables = []

# %%
progress_bar = tqdm(total=n_chains, position=0)

for chain_num in range(n_chains):

    now_chain = np.load('data/chains/' + folder_name +
                        '/best_sh_sn_chains/sh_sn_chain_' + str(chain_num) + '.npy')

    chain_table = np.zeros((len(now_chain), 5), dtype=np.uint16)

    for n_mon, mon_line in enumerate(now_chain):
        if n_mon == 0:
            mon_type = 0
        elif n_mon == len(now_chain) - 1:
            mon_type = 2
        else:
            mon_type = 1

        x_bin = get_hist_position(element=mon_line[0], bins=mapping.x_bins_1nm)
        y_bin = get_hist_position(element=mon_line[1], bins=mapping.y_bins_1nm)
        z_bin = get_hist_position(element=mon_line[2], bins=mapping.z_bins_1nm)

        mon_line_pos = pos_matrix[x_bin, y_bin, z_bin]

        if mon_line_pos > cp.uint16_max:
            print('uint16 is not enough!')

        resist_matrix[x_bin, y_bin, z_bin, mon_line_pos] = chain_num, n_mon, mon_type
        chain_table[n_mon] = x_bin, y_bin, z_bin, mon_line_pos, mon_type

        pos_matrix[x_bin, y_bin, z_bin] += 1

    chain_tables.append(chain_table)
    progress_bar.update()

# %%
print('resist_matrix size, Gb:', resist_matrix.nbytes / 1024 ** 3)
np.save('data/chains/' + folder_name + '/best_resist_matrix_1nm.npy', resist_matrix)

# %%
progress_bar = tqdm(total=len(chain_tables), position=0)

for n, chain_table in enumerate(chain_tables):
    np.save('data/chains/' + folder_name + '/best_chain_tables_1nm/chain_table_' + str(n) + '.npy', chain_table)
    progress_bar.update()
