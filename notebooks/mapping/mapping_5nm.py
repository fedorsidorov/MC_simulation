import importlib

import numpy as np
from tqdm import tqdm

import constants as const
import indexes
import mapping_harris as mapping
# import mapping_aktary as mapping
from functions import mapping_functions as mf

mapping = importlib.reload(mapping)
const = importlib.reload(const)
indexes = importlib.reload(indexes)
mf = importlib.reload(mf)

# %% load arrays
e_matrix_val_exc_sci = np.load('data/e_matrix/Harris/C-C2:4_C-C\':2/e_matrix_val_exc_sci.npy')
e_matrix_val_ion_sci = np.load('data/e_matrix/Harris/C-C2:4_C-C\':2/e_matrix_val_ion_sci.npy')

scission_matrix = e_matrix_val_exc_sci + e_matrix_val_exc_sci

resist_matrix = np.load('/Volumes/ELEMENTS/PyCharm_may/resist_matrix/Harris/resist_matrix_5nm.npy')
chain_lens = np.load('/Volumes/ELEMENTS/PyCharm_may/prepared_chains/Harris/chain_lens.npy')
n_chains = len(chain_lens)

chain_tables = []
progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chain_tables.append(
        np.load('/Volumes/ELEMENTS/PyCharm_may/chain_tables/Harris_5nm/chain_table_' + str(n) + '.npy'))
    progress_bar.update()

resist_shape = mapping.hist_5nm_shape

# %% mapping
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

# %%
lens_final = mf.get_chain_lens(chain_tables)
np.save('data/exposed_chains/Harris/harris_lens_final_4+2.npy', lens_final)

# %%
# progress_bar = tqdm(total=len(chain_tables), position=0)
#
# for n, chain_table in enumerate(chain_tables):
#     np.save('data/chains/' + folder_name + '/best_chain_tables_series_1_5nm_1500/chain_table_' + str(n) +
#             '.npy', chain_table)
#     progress_bar.update()
#
# progress_bar.close()
