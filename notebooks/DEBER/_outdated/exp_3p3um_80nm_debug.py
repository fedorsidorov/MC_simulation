import importlib
import constants
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
from mapping import mapping_3p3um_80nm as mapping
# from functions import DEBER_functions as deber
from functions import mapping_functions as mf
from functions import diffusion_functions as df
from functions import reflow_functions as rf

from scipy.optimize import curve_fit

mapping = importlib.reload(mapping)
# deber = importlib.reload(deber)
mf = importlib.reload(mf)
df = importlib.reload(df)
rf = importlib.reload(rf)

# %%
resist_matrix = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/exp_3p3um_80nm/resist_matrix.npy')
chain_lens = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/exp_3p3um_80nm/chain_lens.npy')
n_chains = len(chain_lens)

chain_tables = []
progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chain_tables.append(
        np.load('/Users/fedor/PycharmProjects/MC_simulation/data/exp_3p3um_80nm/chain_tables/chain_table_' + str(n) + '.npy'))
    progress_bar.update()

resist_shape = mapping.hist_5nm_shape

# %%
l_x = mapping.l_x * 1e-7
l_y = mapping.l_y * 1e-7
area = l_x * l_y
d_PMMA = mapping.z_max * 1e-7  # cm
j_exp_s = 1.9e-9  # A / cm
j_exp_l = 1.9e-9 * l_x
dose_s = 0.6e-6  # C / cm^2
dose_l = dose_s * l_x
T_C = 125
t = dose_l / j_exp_l  # 316 s
dt = 1  # s
Q = dose_s * area
n_electrons = Q / constants.e_SI  # 2 472
n_electrons_s = int(np.around(n_electrons / t))

zip_length = 1000
Tg = 120

xx = mapping.x_centers_5nm * 1e-7  # cm
zz_vac = np.zeros(len(xx))  # cm
monomer_matrix = np.zeros(np.shape(resist_matrix)[:3])

zz_vac_list = [zz_vac]

for i in range(32):

    print(i)

    scission_matrix = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/e_DATA/scission_matrix_' + str(i) + '.npy')
    mf.process_mapping(scission_matrix, resist_matrix, chain_tables)
    print('mapping is done')
    mf.process_depolymerization(resist_matrix, chain_tables, zip_length)
    print('depolymerization is done')


# %%
sum_m, sum_m2, new_monomer_matrix = mf.get_chain_len_matrix(resist_matrix, chain_tables)
monomer_matrix += new_monomer_matrix

sum_m_1d = np.sum(np.sum(sum_m, axis=1), axis=1)
sum_m2_1d = np.sum(np.sum(sum_m2, axis=1), axis=1)
monomer_matrix_1d = np.sum(np.sum(monomer_matrix, axis=1), axis=1)

sum_m_1d_100nm = df.get_100nm_array(sum_m_1d)
sum_m2_1d_100nm = df.get_100nm_array(sum_m2_1d)
monomer_matrix_1d_100nm = df.get_100nm_array(monomer_matrix_1d)

# %%
matrix_Mw_1d_100nm = mf.get_local_Mw_matrix(sum_m_1d_100nm, sum_m2_1d_100nm, monomer_matrix_1d_100nm)
popt, _ = curve_fit(df.minus_exp_gauss, mapping.x_centers_100nm, matrix_Mw_1d_100nm)

plt.figure(dpi=300)
plt.semilogy(mapping.x_centers_100nm, matrix_Mw_1d_100nm, '-o')
# plt.semilogy(mapping.x_centers_50nm, df.minus_exp_gauss(mapping.x_centers_50nm, *popt), '.-')
plt.show()

# %%
etas_100nm_fit = rf.get_viscosity_W(T_C, matrix_Mw_1d_100nm)
mobs_100nm_fit = rf.get_SE_mobility(etas_100nm_fit)

plt.figure(dpi=300)
plt.semilogy(mapping.x_centers_100nm, mobs_100nm_fit, 'o-')
plt.show()


