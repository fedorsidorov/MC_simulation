import importlib
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import constants as cp
import mapping_harris as mapping

# import mapping_aktary as mapping

mapping = importlib.reload(mapping)
cp = importlib.reload(cp)


# %% functions
def get_hist_position(element, bins):
    n_bin = (element - bins[0]) // (bins[1] - bins[0])
    return int(n_bin)


# %% load data
chain_lens = np.load('/Volumes/ELEMENTS/PyCharm_may/prepared_chains/Harris/chain_lens.npy')
hist_2nm = np.load('/Volumes/ELEMENTS/PyCharm_may/rot_sh_sn_chains/Harris/hist_2nm.npy')
n_mon_cell_max = int(np.max(hist_2nm))

plt.imshow(np.average(hist_2nm, axis=0))
plt.show()

# %% create arrays
pos_matrix = np.zeros(mapping.hist_2nm_shape, dtype=np.uint32)
resist_matrix = -np.ones((*mapping.hist_2nm_shape, n_mon_cell_max, 3), dtype=np.uint32)
chain_tables = deque()

# %%
progress_bar = tqdm(total=len(chain_lens), position=0)

for chain_num in range(len(chain_lens)):

    now_chain = np.load('/Volumes/ELEMENTS/PyCharm_may/rot_sh_sn_chains/Harris/rot_sh_sn_chain_' +
                        str(chain_num) + '.npy')

    chain_table = np.zeros((len(now_chain), 5), dtype=np.uint32)

    for n_mon, mon_line in enumerate(now_chain):

        if len(now_chain) == 1:
            mon_type = 10
        elif n_mon == 0:
            mon_type = 0
        elif n_mon == len(now_chain) - 1:
            mon_type = 2
        else:
            mon_type = 1

        x_bin = get_hist_position(element=mon_line[0], bins=mapping.x_bins_2nm)
        y_bin = get_hist_position(element=mon_line[1], bins=mapping.y_bins_2nm)
        z_bin = get_hist_position(element=mon_line[2], bins=mapping.z_bins_2nm)

        mon_line_pos = pos_matrix[x_bin, y_bin, z_bin]

        if mon_line_pos > cp.uint32_max:
            print('uint32 is not enough!')

        resist_matrix[x_bin, y_bin, z_bin, mon_line_pos] = chain_num, n_mon, mon_type
        chain_table[n_mon] = x_bin, y_bin, z_bin, mon_line_pos, mon_type

        pos_matrix[x_bin, y_bin, z_bin] += 1

    chain_tables.append(chain_table)
    progress_bar.update()

# %%
print('resist_matrix size, Gb:', resist_matrix.nbytes / 1024 ** 3)
np.save('/Volumes/ELEMENTS/PyCharm_may/resist_matrix/Harris/resist_matrix_2nm.npy', resist_matrix)

# %%
progress_bar = tqdm(total=len(chain_tables), position=0)

for n, chain_table in enumerate(chain_tables):
    np.save('/Volumes/ELEMENTS/PyCharm_may/chain_tables/Harris_2nm/chain_table_' + str(n) + '.npy', chain_table)
    progress_bar.update()
