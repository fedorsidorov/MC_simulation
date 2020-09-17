import numpy as np
from mapping import mapping_3p3um_80nm as mapping
from functions import mapping_functions as mf
import importlib
from tqdm import tqdm

mapping = importlib.reload(mapping)
mf = importlib.reload(mf)

scission_matrix = np.load('/Users/fedor/PycharmProjects/MC_simulation/scission_matrix.npy') * 2
resist_matrix = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/exp_3p3um_80nm/resist_matrix.npy')
chain_lens = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/exp_3p3um_80nm/chain_lens.npy')
n_chains = len(chain_lens)

chain_tables = []
progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chain_tables.append(
        np.load('/Users/fedor/PycharmProjects/MC_simulation/data/exp_3p3um_80nm/chain_tables/chain_table_' +
                str(n) + '.npy'))
    progress_bar.update()

resist_shape = mapping.hist_5nm_shape

mf.process_mapping(scission_matrix, resist_matrix, chain_tables)
zip_length = 1000
mf.process_depolymerization_2(resist_matrix, chain_tables, zip_length)
