import importlib

import numpy as np
from tqdm import tqdm

import constants as const
import indexes
import mapping_harris as mapping
from functions import mapping_functions as mf

mapping = importlib.reload(mapping)
const = importlib.reload(const)
indexes = importlib.reload(indexes)
mf = importlib.reload(mf)

# %%
# for weight in ['0.175', '0.225', '0.275', '0.320', '0.375']:
for weight in ['0.425']:

    print(weight)

    # load arrays
    scission_matrix = np.load('data/choi_weight/e_matrix_val_ion_sci_' + weight + '.npy')

    resist_matrix = np.load('data/G_calibration/resist_matrix_5nm.npy')
    n_chains = 6081
    chain_lens = np.zeros(n_chains, dtype=int)

    chain_tables = []
    progress_bar = tqdm(total=n_chains, position=0)

    #
    for n in range(n_chains):
        now_chain_table = np.load('data/G_calibration/chain_tables_5nm/chain_table_' + str(n) + '.npy')
        chain_tables.append(now_chain_table)
        chain_lens[n] = len(now_chain_table)
        progress_bar.update()

    resist_shape = mapping.hist_5nm_shape

    # mapping
    n_scissions_moved = 0
    progress_bar = tqdm(total=resist_shape[0], position=0)

    for x_ind in range(resist_shape[0]):
        for y_ind in range(resist_shape[1]):
            for z_ind in range(resist_shape[2]):

                n_scissions = int(scission_matrix[x_ind, y_ind, z_ind])
                monomer_positions = list(
                    np.where(resist_matrix[x_ind, y_ind, z_ind, :, indexes.n_chain_ind] != const.uint32_max)[0]
                )

                while n_scissions:

                    #  check if there exist free monomers
                    inds_free = np.where(
                        resist_matrix[x_ind, y_ind, z_ind, :, indexes.monomer_type_ind] == indexes.free_monomer
                    )[0]

                    for ind in inds_free:
                        if ind in monomer_positions:
                            monomer_positions.remove(ind)

                    if len(monomer_positions) == 0:  # move events to one of further bins
                        mf.move_scissions(scission_matrix, x_ind, y_ind, z_ind, n_scissions)
                        n_scissions_moved += n_scissions
                        break

                    monomer_pos = np.random.choice(monomer_positions)
                    n_scissions -= 1

                    n_chain, n_monomer, monomer_type = resist_matrix[x_ind, y_ind, z_ind, monomer_pos, :]
                    chain_table = chain_tables[n_chain]

                    mf.process_scission(resist_matrix, chain_table, n_monomer, monomer_type)

        progress_bar.update()

    #
    lens_final = mf.get_chain_lens(chain_tables)
    np.save('data/choi_weight/harris_lens_final_' + weight + '.npy', lens_final)
