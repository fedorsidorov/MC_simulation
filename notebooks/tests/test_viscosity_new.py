import numpy as np
import importlib
from tqdm import tqdm
from mapping import mapping_viscosity_80nm as mapping
from functions import reflow_functions as rf
from functions import mapping_functions as mf
import matplotlib.pyplot as plt

mapping = importlib.reload(mapping)
rf = importlib.reload(rf)
mf = importlib.reload(mf)

# %%
scission_array = np.load('scission_array_sum.npy')
n_scissions_arr_100nm = np.sum(scission_array[:20])
n_scissions_total = np.sum(n_scissions_arr_100nm) * 20

# %%
scission_matrix = np.zeros(mapping.hist_5nm_shape)
progress_bar = tqdm(total=n_scissions_total, position=0)

for i in range(int(n_scissions_total)):

    x_ind = np.random.choice(len(mapping.x_centers_5nm))
    y_ind = np.random.choice(len(mapping.y_centers_5nm))
    z_ind = np.random.choice(len(mapping.z_centers_5nm))

    scission_matrix[x_ind, y_ind, z_ind] += 1
    progress_bar.update()

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

zip_length = 1

mf.process_mapping(scission_matrix, resist_matrix, chain_tables)
mf.process_depolymerization_WO_CT(resist_matrix, chain_tables, zip_length)
sum_m, sum_m2 = mf.get_sum_m_m2(chain_tables)
print('Mw =', sum_m2 / sum_m)
