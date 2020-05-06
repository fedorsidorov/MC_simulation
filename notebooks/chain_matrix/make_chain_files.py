import importlib
import numpy as np
import matplotlib.pyplot as plt
import constants_mapping as const_m
from tqdm import tqdm

const_m = importlib.reload(const_m)


#%% functions
def get_hist_position(element, bins):
    for n_bin in range(len(bins) - 1):
        if bins[n_bin] <= element < bins[n_bin + 1]:
            return n_bin
    print('hist position error!')


def get_hist_position_fast(element, bins):
    n_bin = (element - bins[0]) // (bins[1] - bins[0])
    return int(n_bin)


# %% load data
index = '1'

chain_lens = np.load('data/Harris/prepared_chains_' + index + '/prepared_chain_lens.npy')
hist_2nm = np.load('data/Harris/shifted_snaked_chains_' + index + '/hist_2nm.npy')

# plt.imshow(np.average(hist_2nm, axis=2))
# plt.show()

n_chains = len(chain_lens)
n_mon_cell_max = int(np.max(hist_2nm))

# %% create arrays
# pos_matrix = np.zeros(const_m.hist_2nm_shape, dtype=np.uint16)
pos_matrix = np.zeros(const_m.hist_2nm_shape, dtype=np.uint32)
resist_matrix = -np.ones((*const_m.hist_2nm_shape, n_mon_cell_max, 3), dtype=np.uint32)
chain_tables = []

# %%
progress_bar = tqdm(total=n_chains, position=0)

for chain_num in range(n_chains):

    now_chain = np.load('data/Harris/shifted_snaked_chains_' + index + '/shifted_snaked_chain_' + str(chain_num) + '.npy')
    chain_table = np.zeros((len(now_chain), 5), dtype=np.uint16)
    # chain_table = np.zeros((len(now_chain), 5), dtype=np.uint32)

    for n_mon, mon_line in enumerate(now_chain):
        if n_mon == 0:
            mon_type = 0
        elif n_mon == len(now_chain) - 1:
            mon_type = 2
        else:
            mon_type = 1

        x_bin = get_hist_position_fast(element=mon_line[0], bins=const_m.x_bins_2nm)
        y_bin = get_hist_position_fast(element=mon_line[1], bins=const_m.y_bins_2nm)
        z_bin = get_hist_position_fast(element=mon_line[2], bins=const_m.z_bins_2nm)

        # x_bin_slow = get_hist_position(element=mon_line[0], bins=const_m.x_bins_2nm)
        # y_bin_slow = get_hist_position(element=mon_line[1], bins=const_m.y_bins_2nm)
        # z_bin_slow = get_hist_position(element=mon_line[2], bins=const_m.z_bins_2nm)

        # if x_bin != x_bin_slow:
        #     print('x error', mon_line[0], x_bin, x_bin_fast)
        # if y_bin != y_bin_slow:
        #     print('y error', mon_line[0], y_bin, y_bin_fast)
        # if x_bin != x_bin_slow:
        #     print('z error', mon_line[0], z_bin, z_bin_fast)

        mon_line_pos = pos_matrix[x_bin, y_bin, z_bin]

        resist_matrix[x_bin, y_bin, z_bin, mon_line_pos] = chain_num, n_mon, mon_type
        chain_table[n_mon] = x_bin, y_bin, z_bin, mon_line_pos, mon_type

        pos_matrix[x_bin, y_bin, z_bin] += 1

    chain_tables.append(chain_table)
    progress_bar.update()

# %%
# print('resist_matrix size, Gb:', resist_matrix.nbytes / 1024 ** 3)
# np.save('data/Harris/MATRIX_resist.npy', resist_matrix)

# %%
progress_bar = tqdm(total=len(chain_tables), position=0)
total_max = 0
table_max = chain_tables[0]

for _, chain_table in enumerate(chain_tables):
    now_max = np.max(chain_table)
    if now_max > total_max:
        total_max = now_max
        table_max = chain_table
    progress_bar.update()

# %%
progress_bar = tqdm(total=len(chain_tables), position=0)

for n, chain_table in enumerate(chain_tables):
    if np.max(chain_table) > const_m.uint16_max:
        print('type error')
    chain_table_uint16 = chain_table.astype(np.uint16)
    np.save('data/Harris/chain_tables/chain_table_' + str(n) + '.npy', chain_table_uint16)
    progress_bar.update()

# %%
# progress_bar = tqdm(total=len(chain_tables), position=0)
#
# for i, chain in enumerate(chain_tables):
#     for j, line in enumerate(chain):
#
#         x, y, z, pos, mon_t = line.astype(int)
#         mat_cn, n_mon, mat_type = resist_matrix[x, y, z, pos]
#
#         if mat_cn != i or n_mon != j or mat_type != mon_t:
#             print('ERROR!', i, j)
#             print('chain_num:', mat_cn, i)
#             print('n_mon', n_mon, j)
#             print('mon_type', mon_t, mat_type)
#
#     progress_bar.update()
