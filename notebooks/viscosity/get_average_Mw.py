import numpy as np
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from mapping import mapping_viscosity as mapping
from functions import mapping_functions as mf

mapping = importlib.reload(mapping)
mf = importlib.reload(mf)

# %%
resist_matrix = np.load('/Volumes/ELEMENTS/chains_viscosity/resist_matrix_1.npy')
chain_lens = np.load('/Volumes/ELEMENTS/chains_viscosity/prepared_chains_1/chain_lens.npy')
n_chains = len(chain_lens)

chain_tables = []
progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chain_tables.append(
        np.load('/Volumes/ELEMENTS/chains_viscosity/chain_tables_1/chain_table_' + str(n) + '.npy'))
    progress_bar.update()

resist_shape = mapping.hist_5nm_shape

# %%
zip_length = 1000

scission_matrix = np.load('data/sci_mat_3p3um_80nm_accum/scission_matrix_total_7.npy')

# %%
print('mapping ...')
mf.process_mapping(scission_matrix, resist_matrix, chain_tables)

print('depolymerization ...')
mf.process_depolymerization(resist_matrix, chain_tables, zip_length)

print('chain lens ...')
sum_m, sum_m2, new_monomer_matrix = mf.get_sum_m_m2_mon_matrix(resist_matrix, chain_tables)

# %%
sum_m_n, sum_m2_n = mf.get_sum_m_m2(chain_tables)


