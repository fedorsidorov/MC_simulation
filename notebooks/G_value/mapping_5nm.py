import importlib
from collections import deque

import numpy as np
from tqdm import tqdm

import constants as const
import indexes
from mapping import mapping_harris as mapping
from functions import mapping_functions as mf

mapping = importlib.reload(mapping)
const = importlib.reload(const)
indexes = importlib.reload(indexes)
mf = importlib.reload(mf)


# %% from 0.01 to 0.03
# weights = [
#     0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020,
#     0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.030
# ]

weights = [
    0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080,
    0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120, 0.125, 0.130, 0.135
]

for weight in weights:

    print(weight)

    e_matrix_val = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/e_matrix_val_TRUE.npy')
    e_matrix_dE = np.load('data/e_matrix_E_dep.npy')

    # e_matrix_val = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/e_matrix_val_TRUE_NEW.npy')
    # e_matrix_dE = np.load('data/e_matrix_E_dep_NEW.npy')

    resist_shape = np.shape(e_matrix_val)
    e_matrix_sci = np.zeros(resist_shape)

    print('start to form scission matrix')
    progress_bar = tqdm(total=resist_shape[0], position=0)

    for x_ind in range(resist_shape[0]):
        for y_ind in range(resist_shape[1]):
            for z_ind in range(resist_shape[2]):

                n_val = int(e_matrix_val[x_ind, y_ind, z_ind])

                scissions = np.where(np.random.random(n_val) < weight)[0]
                e_matrix_sci[x_ind, y_ind, z_ind] = len(scissions)

        progress_bar.update()
    print('scission matrix is formed')

    sample = '1'

    resist_matrix = np.load('/Volumes/TOSHIBA EXT/chains_harris/resist_matrix_' + sample + '.npy')
    chain_lens = np.load('/Volumes/TOSHIBA EXT/chains_harris/prepared_chains_' + sample + '/chain_lens.npy')
    n_chains = len(chain_lens)

    chain_tables = deque()

    print('load chain tables')
    progress_bar = tqdm(total=n_chains, position=0)

    for n in range(n_chains):
        now_chain_table =\
            np.load('/Volumes/TOSHIBA EXT/chains_harris/chain_tables_' + sample +
                    '/chain_table_' + str(n) + '.npy')
        chain_tables.append(now_chain_table)
        progress_bar.update()
    print('chain tables are loaded')

    print('start to process mapping')
    mf.process_mapping(e_matrix_sci, resist_matrix, chain_tables)
    print('mapping is processed')

    print('start to calculate final lens')
    lens_final = mf.get_chain_lens_fast(chain_tables, count_monomers=False)
    print('final lens are calculated')

    np.save('/Volumes/TOSHIBA EXT/G_calibration/' + sample + '/scission_matrix_' + str(weight) + '.npy',
            e_matrix_sci)
    np.save('/Volumes/TOSHIBA EXT/G_calibration/' + sample + '/harris_lens_final_' + str(weight) + '.npy',
            lens_final)

# %%
Mn = np.average(chain_lens) * const.u_MMA
Mf = np.average(lens_final) * const.u_MMA

total_E_loss = np.sum(e_matrix_dE)

GG_sim = (Mn / Mf - 1) * const.rho_PMMA * const.Na / (total_E_loss / mapping.volume_cm3 * Mn) * 100
GG_theor = np.sum(e_matrix_sci) / total_E_loss * 100

