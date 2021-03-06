import importlib

import numpy as np
from tqdm import tqdm

import indexes
import mapping_aktary as mapping

mapping = importlib.reload(mapping)
indexes = importlib.reload(indexes)

# %%
folder_name = 'combine_chains'

resist_matrix = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/chains/' +
                        folder_name + '/best_resist_matrix_1nm.npy')
n_chains = 754

chain_tables_final = []

progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chain_tables_final.append(
        np.load('/Users/fedor/PycharmProjects/MC_simulation/data/chains/' +
                folder_name + '/best_chain_tables_series_1_1nm_1500/chain_table_' + str(n) + '.npy'))
    progress_bar.update()

final_resist_shape = mapping.hist_1nm_shape
n_chains_matrix = np.zeros(final_resist_shape)
sum_lens_matrix = np.zeros(final_resist_shape)

# %%
p_bar = tqdm(total=len(chain_tables_final), position=0)

for chain_table in chain_tables_final:

    if len(chain_table) == 1:
        print('chain consisting of 1 monomer')
        continue

    now_chain = []

    for line in chain_table:

        x_bin, y_bin, z_bin, monomer_line_pos, monomer_type = line
        chain_num, n_monomer, _ = resist_matrix[x_bin, y_bin, z_bin, monomer_line_pos]

        if monomer_type == indexes.free_monomer:
            n_chains_matrix[x_bin, y_bin, z_bin] += 1
            sum_lens_matrix[x_bin, y_bin, z_bin] += 1
            continue

        now_chain.append(line)

        if monomer_type == indexes.end_monomer:
            now_chain_arr = np.array(now_chain)
            unique_bins = np.unique(now_chain_arr[:, :3], axis=0)

            for unique_bin_line in unique_bins:
                x_bin_unique, y_bin_unique, z_bin_unique = unique_bin_line
                n_chains_matrix[x_bin_unique, y_bin_unique, z_bin_unique] += 1
                sum_lens_matrix[x_bin_unique, y_bin_unique, z_bin_unique] += len(now_chain)

            now_chain = []

    p_bar.update()

# %%
np.save('data/chains/combine_chains/development/n_chains_matrix_series_1_1nm_1500.npy', n_chains_matrix)
np.save('data/chains/combine_chains/development/sum_lens_matrix_series_1_1nm_1500.npy', sum_lens_matrix)
