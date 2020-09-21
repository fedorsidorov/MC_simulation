import numpy as np
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from mapping import mapping_viscosity as mapping
from functions import mapping_functions as mf
import constants

mapping = importlib.reload(mapping)
mf = importlib.reload(mf)

# %%
# zip_length = 500
# zip_length = 700
# zip_length = 1000
zip_length = 1500

f_nums = ['5', '10', '15', '20']

sci_mat_avg_list = []
Mw_list = []

for f_num in f_nums:

    print(f_num)

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

    scission_matrix = np.load('data/sci_mat_viscosity/scission_matrix_total_' + f_num + '.npy')
    sci_mat_avg_list.append(np.average(scission_matrix))

    print('mapping ...')
    mf.process_mapping(scission_matrix, resist_matrix, chain_tables)

    print('depolymerization ...')
    mf.process_depolymerization(resist_matrix, chain_tables, zip_length)

    print('calculate Mw ...')
    sum_m, sum_m2 = mf.get_sum_m_m2(chain_tables)

    print('Mw =', sum_m2 / sum_m)
    Mw_list.append(sum_m2 / sum_m)

Mw_arr = np.array(Mw_list)
sci_avg_arr = np.array(sci_mat_avg_list)

# %%
np.save('notebooks/viscosity/Mw_arr_1500.npy', Mw_arr)
np.save('notebooks/viscosity/sci_avg_arr_1500.npy', sci_avg_arr)
