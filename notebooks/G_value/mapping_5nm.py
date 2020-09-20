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
for weight in ['0.3', '0.4']:

    print(weight)

    scission_matrix = np.load('data/scission_mat_weight/e_matrix_scissions_' + weight + '.npy')
    resist_matrix = np.load('/Volumes/ELEMENTS/chains_harris/resist_matrix_1.npy')
    chain_lens = np.load('/Volumes/ELEMENTS/chains_harris/prepared_chains_1/chain_lens.npy')
    n_chains = len(chain_lens)

    chain_tables = []
    progress_bar = tqdm(total=n_chains, position=0)

    for n in range(n_chains):
        now_chain_table = np.load('/Volumes/ELEMENTS/chains_harris/chain_tables_1/chain_table_' + str(n) + '.npy')
        chain_tables.append(now_chain_table)
        progress_bar.update()

    mf.process_mapping(scission_matrix, resist_matrix, chain_tables)
    lens_final = mf.get_chain_lens(chain_tables)
    np.save('data/G_calibration/harris_lens_final_' + weight + '.npy', lens_final)
