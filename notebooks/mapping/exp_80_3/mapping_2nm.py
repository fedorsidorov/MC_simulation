import importlib
import numpy as np
from tqdm import tqdm
import constants as const
import indexes
import mapping_exp_80_3 as mapping
from functions import mapping_functions as mf

mapping = importlib.reload(mapping)
const = importlib.reload(const)
indexes = importlib.reload(indexes)
mf = importlib.reload(mf)

# %% load arrays
scission_matrix = np.load('data/e_matrix/exp_80_3/scission_matrix_snaked.npy')
E_dep_matrix = np.load('data/e_matrix/exp_80_3/E_dep_matrix_snaked.npy')

resist_matrix = np.load('data/exp_80_3/resist_matrix_2nm_80_3.npy')
chain_lens = np.load('data/exp_80_3/chain_lens_80_3.npy')
n_chains = len(chain_lens)

# %%
chain_tables = []
progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chain_tables.append(
        np.load('data/exp_80_3/chain_tables/chain_table_' + str(n) + '.npy'))
    progress_bar.update()

resist_shape = mapping.hist_2nm_shape

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
np.save('data/exp_80_3/resist_matrix_2nm_80_3_exposed.npy', resist_matrix)

progress_bar = tqdm(total=len(chain_tables), position=0)

for n, chain_table in enumerate(chain_tables):
    np.save('data/exp_80_3/chain_tables_exposed/chain_table_' + str(n) + '.npy', chain_table)
    progress_bar.update()

progress_bar.close()
