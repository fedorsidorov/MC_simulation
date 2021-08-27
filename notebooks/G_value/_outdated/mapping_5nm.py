import importlib

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

# %%
sample = '3'

for weight in [0.1, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35]:

    print(weight)
    weight = str(weight)

    scission_matrix = np.load('data/scission_mat_weight/' + sample + '/e_matrix_scissions_' + weight + '.npy')
    resist_matrix = np.load('/Volumes/ELEMENTS/chains_harris/resist_matrix_' + sample + '.npy')
    chain_lens = np.load('/Volumes/ELEMENTS/chains_harris/prepared_chains_' + sample + '/chain_lens.npy')
    n_chains = len(chain_lens)

    chain_tables = []
    progress_bar = tqdm(total=n_chains, position=0)

    for n in range(n_chains):
        now_chain_table =\
            np.load('/Volumes/ELEMENTS/chains_harris/chain_tables_' + sample + '/chain_table_' + str(n) + '.npy')
        chain_tables.append(now_chain_table)
        progress_bar.update()

    mf.process_mapping(scission_matrix, resist_matrix, chain_tables)
    lens_final = mf.get_chain_lens(chain_tables)
    # np.save('data/G_calibration/' + sample + '/harris_lens_final_' + weight + '.npy', lens_final)
