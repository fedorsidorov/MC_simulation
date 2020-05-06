import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import constants_mapping as const_m
import constants_physics as const_p

const_m = importlib.reload(const_m)
const_p = importlib.reload(const_p)


# %% functions
def move_scissions(sci_mat, xi, yi, zi, n_sci):
    if xi + 1 < np.shape(sci_mat)[0]:
        sci_mat[xi + 1, yi, zi] += n_sci
    elif yi + 1 < np.shape(sci_mat)[1]:
        sci_mat[xi, yi + 1, zi] += n_sci
    elif zi + 1 < np.shape(sci_mat)[2]:
        sci_mat[xi, yi, zi + 1] += n_sci
    else:
        print('no space for extra events, nowhere to move')


def rewrite_monomer_type(res_mat, ch_tab, n_mon, new_type):
    ch_tab[n_mon, const_m.monomer_type_ind] = new_type
    xi, yi, zi, mon_line_pos = ch_tab[n_mon, :const_m.monomer_type_ind].astype(int)
    res_mat[xi, yi, zi, mon_line_pos, const_m.monomer_type_ind] = new_type


def process_scission(res_mat, ch_tab, n_mon, mon_type):

    if mon_type == const_m.middle_monomer:  # bonded monomer
        # choose between left and right bond
        new_mon_type = np.random.choice([0, 2])
        rewrite_monomer_type(res_mat, ch_tab, n_mon, new_mon_type)
        n_next_mon = n_mon + new_mon_type - 1
        next_xi, next_yi, next_zi, _, next_mon_type = ch_tab[n_next_mon]

        # if next monomer was at the end
        if next_mon_type in [const_m.begin_monomer, const_m.end_monomer]:
            next_mon_new_type = const_m.free_monomer
            rewrite_monomer_type(res_mat, ch_tab, n_next_mon, next_mon_new_type)

        # if next monomer is full bonded
        elif next_mon_type == const_m.middle_monomer:
            next_mon_new_type = next_mon_type - (new_mon_type - 1)
            rewrite_monomer_type(res_mat, ch_tab, n_next_mon, next_mon_new_type)

        else:
            print('next monomer type error, next_mon_type =', mon_type, next_mon_type)

    elif mon_type in [const_m.begin_monomer, const_m.end_monomer]:  # half-bonded monomer
        new_mon_type = const_m.free_monomer
        rewrite_monomer_type(res_mat, ch_tab, n_mon, new_mon_type)
        n_next_mon = n_mon - (mon_type - 1)  # minus, Karl!
        next_xi, next_yi, next_zi, _, next_mon_type = ch_tab[n_next_mon]

        # if next monomer was at the end
        if next_mon_type in [const_m.begin_monomer, const_m.end_monomer]:
            next_mon_new_type = const_m.free_monomer
            rewrite_monomer_type(res_mat, ch_tab, n_next_mon, next_mon_new_type)

        # if next monomer is full bonded
        elif next_mon_type == const_m.middle_monomer:
            next_mon_new_type = next_mon_type + (mon_type - 1)
            rewrite_monomer_type(res_mat, ch_tab, n_next_mon, next_mon_new_type)

        else:
            print('next monomer type error, next_mon_type =', mon_type, next_mon_type)

    else:
        print('monomer type error, monomer_type =', monomer_type)


def get_chain_lens(ch_tab):
    lens_final = []
    p_bar = tqdm(total=len(ch_tab), position=0)

    for _, now_chain in enumerate(ch_tab):
        cnt = 0
        if len(now_chain) == 1:
            lens_final.append(1)
            continue
        for line in now_chain:
            mon_type = line[const_m.monomer_type_ind]
            if mon_type == 0:
                cnt = 1
            elif mon_type == 1:
                cnt += 1
            elif mon_type == 2:
                cnt += 1
                lens_final.append(cnt)
                cnt = 0
        p_bar.update()

    return np.array(lens_final)


# %% load arrays
deg_path = 'C-C2:4_C-C\':2'

e_matrix_val_exc_sci = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/e_matrix/' + deg_path + '/e_matrix_val_exc_sci.npy')
e_matrix_val_ion_sci = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/e_matrix/' + deg_path + '/e_matrix_val_ion_sci.npy')

scission_matrix = e_matrix_val_exc_sci + e_matrix_val_exc_sci

resist_matrix = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/Harris/MATRIX_resist_1.npy')
chain_lens_array = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/Harris/prepared_chains_1/prepared_chain_lens.npy')
lens_before = chain_lens_array
n_chains = len(chain_lens_array)

chain_tables = []
progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chain_tables.append(
        np.load('/Users/fedor/PycharmProjects/MC_simulation/data/Harris/chain_tables_1/chain_table_' + str(n) + '.npy'))
    progress_bar.update()

resist_shape = const_m.hist_2nm_shape

# %% mapping
n_scissions_moved = 0
progress_bar = tqdm(total=resist_shape[0], position=0)

for x_ind in range(resist_shape[0]):
    for y_ind in range(resist_shape[1]):
        for z_ind in range(resist_shape[2]):

            n_scissions = int(scission_matrix[x_ind, y_ind, z_ind])
            monomer_positions = list(
                np.where(resist_matrix[x_ind, y_ind, z_ind, :, const_m.n_chain_ind] != const_m.uint32_max)[0]
            )

            while n_scissions:

                #  check if there exist free monomers
                inds_free = np.where(
                    resist_matrix[x_ind, y_ind, z_ind, :, const_m.monomer_type_ind] == const_m.free_monomer
                )[0]

                for ind in inds_free:
                    if ind in monomer_positions:
                        monomer_positions.remove(ind)

                if len(monomer_positions) == 0:  # move events to one of further bins
                    move_scissions(scission_matrix, x_ind, y_ind, z_ind, n_scissions)
                    n_scissions_moved += n_scissions
                    break

                monomer_pos = np.random.choice(monomer_positions)
                n_scissions -= 1

                n_chain, n_monomer, monomer_type = resist_matrix[x_ind, y_ind, z_ind, monomer_pos, :]
                chain_table = chain_tables[n_chain]

                process_scission(resist_matrix, chain_table, n_monomer, monomer_type)

    progress_bar.update()

#%%
# lens_after = get_chain_lens(chain_tables)
np.save('data/Harris/lens_final_' + deg_path + '.npy', lens_after)
