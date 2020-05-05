import importlib
import numpy as np
import matplotlib.pyplot as plt
import constants_mapping as const_m
from tqdm import tqdm
from itertools import product

const_m = importlib.reload(const_m)

# %%
chain_lens = np.load('data/Harris/prepared_chain_lens_array.npy')
hist_2nm = np.load('data/Harris/hist_2nm.npy')

plt.imshow(np.average(hist_2nm, axis=2))
plt.show()

# %% constants
n_chains = len(chain_lens)
n_mon_cell_max = np.max(hist_2nm)

# %%
pos_matrix = np.zeros(const_m.hist_2nm_shape, dtype=np.uint32)
resist_matrix = -np.ones((*const_m.hist_2nm_shape, n_mon_cell_max, 3), dtype=np.uint32)
chain_tables = []

# %%
for chain_num in range(n_chains):

    now_chain = np.load('data/Harris/shifted_snaked_chains/shifted_snaked_chain_' + str(chain_num) + '.npy')

    chain_table = np.zeros((len(now_chain), 5), dtype=np.uint32)

    for n_mon, mon_line in enumerate(now_chain):

        if n_mon == 0:
            mon_type = 0
        elif n_mon == len(now_chain) - 1:
            mon_type = 2
        else:
            mon_type = 1

        hist_x = np.histogram(mon_line[0], bins=const_m.x_bins_2nm)
        hist_y = np.histogram(mon_line[1], bins=const_m.y_bins_2nm)
        hist_z = np.histogram(mon_line[2], bins=const_m.z_bins_2nm)

        x_bin = np.where(hist_x == 1)[0]
        y_bin = np.where(hist_y == 1)[0]
        z_bin = np.where(hist_z == 1)[0]

        mon_line_pos = pos_matrix[x_bin, y_bin, z_bin]

        resist_matrix[x_bin, y_bin, z_bin, mon_line_pos] = chain_num, n_mon, mon_type
        chain_table[n_mon] = x_bin, y_bin, z_bin, mon_line_pos, mon_type

        pos_matrix[x_bin, y_bin, z_bin] += 1

    chain_tables.append(chain_table)

# %%
print('resist_matrix size, Gb:', resist_matrix.nbytes / 1024 ** 3)
np.save('MATRIX_resist_Harris_2020.npy', resist_matrix)

# %%
progress_bar = tqdm(total=len(chain_tables), position=0)

for n, chain_table in enumerate(chain_tables):
    np.save('data/Harris/chain_tables/chain_table_' + str(n) + '.npy', chain_table)
    progress_bar.update()

# %%
progress_bar = tqdm(total=len(chain_tables), position=0)

for i, chain in enumerate(chain_tables):
    for j, line in enumerate(chain):

        x, y, z, pos, mon_t = line.astype(int)
        mat_cn, n_mon, mat_type = resist_matrix[x, y, z, pos]

        if mat_cn != i or n_mon != j or mat_type != mon_t:
            print('ERROR!', i, j)
            print('chain_num:', mat_cn, i)
            print('n_mon', n_mon, j)
            print('mon_type', mon_t, mat_type)

    progress_bar.update()
