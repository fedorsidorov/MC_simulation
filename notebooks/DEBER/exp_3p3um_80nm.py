import importlib
import constants
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mapping import mapping_3p3um_80nm as mapping
from functions import DEBER_functions as deber
from functions import mapping_functions as mf
from functions import diffusion_functions as df
from functions import reflow_functions as rf
from functions import plot_functions as pf

mapping = importlib.reload(mapping)
deber = importlib.reload(deber)
mf = importlib.reload(mf)
df = importlib.reload(df)
rf = importlib.reload(rf)
pf = importlib.reload(pf)

# %%
resist_matrix = np.load('data/exp_3p3um_80nm/resist_matrix.npy')
chain_lens = np.load('data/exp_3p3um_80nm/chain_lens.npy')
n_chains = len(chain_lens)

chain_tables = []
progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chain_tables.append(
        np.load('data/exp_3p3um_80nm/chain_tables/chain_table_' + str(n) + '.npy'))
    progress_bar.update()

resist_shape = mapping.hist_5nm_shape

# %%
xx = mapping.x_centers_5nm * 1e-7  # cm
zz_vac = np.zeros(len(xx))  # cm
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

# %%
# eta = 5e+6  # Pa s
# eta = 1e+5  # Pa s
# zz_vac_list = []

# print('simulate e-beam scattering')
e_DATA, e_DATA_PMMA_val = deber.get_e_DATA_PMMA_val(xx, zz_vac, d_PMMA, n_electrons=10, E0=20e+3, r_beam=100e-7)

weight = 0.35
scission_matrix, E_dep_matrix = deber.get_scission_matrix(e_DATA, weight)

print('G = ', np.sum(scission_matrix) / np.sum(E_dep_matrix) * 100)
print(np.where(scission_matrix != 0))

# %%
scission_matrix_2d = np.sum(scission_matrix, axis=1)
scission_matrix_1d = np.sum(scission_matrix_2d, axis=1)

plt.figure(dpi=300)
plt.plot(mapping.x_centers_5nm, scission_matrix_1d)
plt.show()

print(np.sum(scission_matrix_1d))

# %%
mf.process_mapping(scission_matrix, resist_matrix, chain_tables)

# %%
zip_length = 5
mf.process_depolymerization_2(resist_matrix, chain_tables, zip_length)

# %%
monomer_matrix = np.zeros(np.shape(resist_matrix)[:3])

# %%
sum_m, sum_m2, new_monomer_matrix = mf.get_chain_len_matrix(resist_matrix, chain_tables)
monomer_matrix += new_monomer_matrix

# sum_m_2d = np.average(sum_m, axis=1)
# sum_m2_2d = np.average(sum_m2, axis=1)
# monomer_matrix_2d = np.sum(monomer_matrix, axis=1)

# %%
# matrix_Mw = mf.get_local_Mw_matrix(sum_m, sum_m2, new_monomer_matrix)
# matrix_Mw = mf.get_local_Mw_matrix(sum_m, sum_m2, monomer_matrix)
matrix_Mw_1d = mf.get_local_Mw_matrix(sum_m, sum_m2, monomer_matrix)

# matrix_Mw = np.zeros(np.shape(sum_m))
#
# for i in range(np.shape(sum_m)[0]):
#     for j in range(np.shape(sum_m)[1]):
#         for k in range(np.shape(sum_m)[2]):
#             matrix_Mw[i, j, k] = (sum_m2[i, j, k] + (1 * 100) ** 2 * monomer_matrix[i, j, k]) /\
#                 (sum_m[i, j, k] + (1 * 100) * monomer_matrix[i, j, k])

# %%
# matrix_Mw_2d = np.average(matrix_Mw, axis=1)
# matrix_Mw_1d = np.average(matrix_Mw_2d, axis=1)

plt.figure(dpi=300)
# plt.imshow(matrix_Mw_2d[200:-200, :].transpose())
plt.plot(matrix_Mw_1d)
plt.show()

# %%
etas = rf.get_viscosity_W(T_C, matrix_Mw_1d)

plt.figure(dpi=300)
plt.semilogy(mapping.x_centers_5nm, etas)
plt.show()

# %%
Tg = 120
dT = T_C - Tg

monomer_matrix_2d = np.sum(new_monomer_matrix, axis=1)
monomer_matrix_2d_final = df.track_all_monomers(monomer_matrix_2d, xx, zz_vac, d_PMMA, dT, wp=1, t_step=5)
# np.sum(monomer_matrix_2d_final)
