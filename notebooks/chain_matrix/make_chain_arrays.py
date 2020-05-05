import importlib
import numpy as np
from itertools import product


# %%
source_dir = mv.sim_path_MAC + 'CHAINS/Harris_500nm/comb_100x100x500_center/'

# %% constants
N_chains_total = 3711
N_mon_chain_max = 7500 * 3
N_mon_cell_max = 502

l_xyz = np.array((100, 100, 500))

x_beg, y_beg, z_beg = (-l_xyz[0] / 2, -l_xyz[0] / 2, 0)
xyz_beg = np.array((x_beg, y_beg, z_beg))
xyz_end = xyz_beg + l_xyz
x_end, y_end, z_end = xyz_end

step_2nm = 2

x_bins_2nm = np.arange(x_beg, x_end + 1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end + 1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end + 1, step_2nm)

bins_2nm = x_bins_2nm, y_bins_2nm, z_bins_2nm

x_grid_2nm = (x_bins_2nm[:-1] + x_bins_2nm[1:]) / 2
y_grid_2nm = (y_bins_2nm[:-1] + y_bins_2nm[1:]) / 2
z_grid_2nm = (z_bins_2nm[:-1] + z_bins_2nm[1:]) / 2

resist_shape = len(x_grid_2nm), len(y_grid_2nm), len(z_grid_2nm)

# %%
pos_matrix = np.zeros(resist_shape, dtype=np.uint16)
resist_matrix = - np.ones((*resist_shape, N_mon_cell_max, 3), dtype=np.uint16)

chain_table = - np.ones((N_chains_total, N_mon_chain_max, 5), dtype=np.uint16)

# %%
for chain_num in range(N_chains_total):

    mf.upd_progress_bar(chain_num, N_chains_total)

    now_chain = np.load(source_dir + 'chain_shift_' + str(chain_num) + '.npy')

    for n_mon, mon_line in enumerate(now_chain):

        if n_mon == 0:
            mon_type = 0
        elif n_mon == len(now_chain) - 1:
            mon_type = 2
        else:
            mon_type = 1

        if not (np.all(mon_line >= xyz_beg) and np.all(mon_line <= xyz_end)):
            chain_table[chain_num, n_mon, -1] = mon_type
            continue

        now_x, now_y, now_z = mon_line

        x_ind = mf.get_closest_el_ind(x_grid_2nm, now_x)
        y_ind = mf.get_closest_el_ind(y_grid_2nm, now_y)
        z_ind = mf.get_closest_el_ind(z_grid_2nm, now_z)

        mon_line_pos = pos_matrix[x_ind, y_ind, z_ind]

        resist_matrix[x_ind, y_ind, z_ind, mon_line_pos] = chain_num, n_mon, mon_type

        chain_table[chain_num, n_mon] = x_ind, y_ind, z_ind, mon_line_pos, mon_type

        pos_matrix[x_ind, y_ind, z_ind] += 1

# %%
print('resist_matrix size, Gb:', resist_matrix.nbytes / 1024 ** 3)
np.save('MATRIX_resist_Harris.npy', resist_matrix)

# %%
print('chain_table size, Gb:', chain_table.nbytes / 1024 ** 3)
np.save('TABLE_chains_Harris.npy', chain_table)

# %%
resist_matrix = np.load('MATRIX_resist_Harris.npy')
chain_table = np.load('TABLE_chains_Harris.npy')

# %%
n_distr = []

for i, j, k in product(range(len(x_grid_2nm)), range(len(y_grid_2nm)), range(len(z_grid_2nm))):
    now_arr = resist_matrix[i, j, k]

    inds = np.where(now_arr[:, 0] != 65535)[0]

    #    print(len(inds))

    n_distr.append(len(inds))

n_distr_list = np.array(n_distr)


