import importlib
import constants
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
from mapping import mapping_3p3um_80nm as mapping
from functions import MC_functions as mcf
from functions import DEBER_functions as deber
from functions import mapping_functions as mf
from functions import diffusion_functions as df
from functions import reflow_functions as rf
from functions import plot_functions as pf
from functions import evolver_functions as ef

from scipy.optimize import curve_fit

mapping = importlib.reload(mapping)
deber = importlib.reload(deber)
mcf = importlib.reload(mcf)
mf = importlib.reload(mf)
df = importlib.reload(df)
ef = importlib.reload(ef)
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

time_s = 10

e_DATA, e_DATA_PMMA_val = deber.get_e_DATA_PMMA_val(
    xx=xx,
    zz_vac=zz_vac,
    d_PMMA=d_PMMA,
    n_electrons=n_electrons_s*time_s,
    E0=20e+3,
    r_beam=100e-7
)

weight = 0.35
scission_matrix, E_dep_matrix = deber.get_scission_matrix(e_DATA, weight)
print('G = ', np.sum(scission_matrix) / np.sum(E_dep_matrix) * 100)

# %%
scission_matrix_2d = np.sum(scission_matrix, axis=1)
scission_matrix_1d = np.sum(scission_matrix_2d, axis=1)

plt.figure(dpi=300)
plt.plot(mapping.x_centers_5nm, scission_matrix_1d)
plt.show()

print('n_scissions =', np.sum(scission_matrix_1d))

# %%
mf.process_mapping(scission_matrix, resist_matrix, chain_tables)

# %%
zip_length = 1000
mf.process_depolymerization(resist_matrix, chain_tables, zip_length)

# %%
# for ct in chain_tables:
#     if len(np.where(ct[:, -1] == 20)[0]) > 0:
#         print('found')
#         break

# %%
monomer_matrix = np.zeros(np.shape(resist_matrix)[:3])

# %%
sum_m, sum_m2, new_monomer_matrix = mf.get_chain_len_matrix(resist_matrix, chain_tables)
monomer_matrix += new_monomer_matrix


# %%
def get_25nm_array(array):
    hist_weights = np.histogram(mapping.x_centers_5nm, bins=mapping.x_bins_25nm, weights=array)[0]
    hist = np.histogram(mapping.x_centers_5nm, bins=mapping.x_bins_25nm)[0]
    return hist_weights / hist


def get_50nm_array(array):
    hist_weights = np.histogram(mapping.x_centers_5nm, bins=mapping.x_bins_50nm, weights=array)[0]
    hist = np.histogram(mapping.x_centers_5nm, bins=mapping.x_bins_50nm)[0]
    return hist_weights / hist


def get_100nm_array(array):
    hist_weights = np.histogram(mapping.x_centers_5nm, bins=mapping.x_bins_100nm, weights=array)[0]
    hist = np.histogram(mapping.x_centers_5nm, bins=mapping.x_bins_100nm)[0]
    return hist_weights / hist


sum_m_1d = np.sum(np.sum(sum_m, axis=1), axis=1)
sum_m2_1d = np.sum(np.sum(sum_m2, axis=1), axis=1)
monomer_matrix_1d = np.sum(np.sum(monomer_matrix, axis=1), axis=1)

sum_m_1d_25nm = get_25nm_array(sum_m_1d)
sum_m2_1d_25nm = get_25nm_array(sum_m2_1d)
monomer_matrix_1d_25nm = get_25nm_array(monomer_matrix_1d)

sum_m_1d_50nm = get_50nm_array(sum_m_1d)
sum_m2_1d_50nm = get_50nm_array(sum_m2_1d)
monomer_matrix_1d_50nm = get_50nm_array(monomer_matrix_1d)

sum_m_1d_100nm = get_100nm_array(sum_m_1d)
sum_m2_1d_100nm = get_100nm_array(sum_m2_1d)
monomer_matrix_1d_100nm = get_100nm_array(monomer_matrix_1d)

# sum_m_2d = np.average(sum_m, axis=1)
# sum_m2_2d = np.average(sum_m2, axis=1)
# monomer_matrix_2d = np.sum(monomer_matrix, axis=1)

# %%
matrix_Mw_1d_25nm = mf.get_local_Mw_matrix(sum_m_1d_25nm, sum_m2_1d_25nm, monomer_matrix_1d_25nm)
matrix_Mw_1d_50nm = mf.get_local_Mw_matrix(sum_m_1d_50nm, sum_m2_1d_50nm, monomer_matrix_1d_50nm)
matrix_Mw_1d_100nm = mf.get_local_Mw_matrix(sum_m_1d_100nm, sum_m2_1d_100nm, monomer_matrix_1d_100nm)

plt.figure(dpi=300)
# plt.semilogy(mapping.x_centers_25nm, matrix_Mw_1d_25nm)
# plt.semilogy(mapping.x_centers_50nm, matrix_Mw_1d_50nm)
plt.semilogy(mapping.x_centers_100nm, matrix_Mw_1d_100nm, '-o')
plt.show()


# %%
def minus_exp_gauss(xx, M0, a, s):
    return np.exp(M0 - a*np.exp(-xx**2 / s))
    # return M0 - a*np.exp(-xx**2 / s)


popt, _ = curve_fit(minus_exp_gauss, mapping.x_centers_100nm, matrix_Mw_1d_100nm)

plt.figure(dpi=300)
plt.semilogy(mapping.x_centers_100nm, matrix_Mw_1d_100nm, '-o')
plt.semilogy(mapping.x_centers_50nm, minus_exp_gauss(mapping.x_centers_50nm, *popt), '.-')
plt.show()


# %%
etas_25nm = rf.get_viscosity_W(T_C, matrix_Mw_1d_25nm)
etas_50nm = rf.get_viscosity_W(T_C, matrix_Mw_1d_50nm)
etas_50nm_fit = rf.get_viscosity_W(T_C, minus_exp_gauss(mapping.x_centers_50nm, *popt))
etas_100nm = rf.get_viscosity_W(T_C, matrix_Mw_1d_100nm)

mobs_50nm = rf.get_SE_mobility(etas_50nm)
mobs_50nm_fit = rf.get_SE_mobility(etas_50nm_fit)

plt.figure(dpi=300)
# plt.semilogy(mapping.x_centers_25nm, etas_25nm)
# plt.semilogy(mapping.x_centers_100nm, etas_100nm, 'o-')
plt.semilogy(mapping.x_centers_50nm, mobs_50nm, 'o-')
plt.semilogy(mapping.x_centers_50nm, mobs_50nm_fit, 'o-')
plt.show()

# %%
Tg = 120
dT = T_C - Tg

monomer_matrix_2d = np.sum(monomer_matrix, axis=1)
monomer_matrix_2d_final = df.track_all_monomers(monomer_matrix_2d, xx, zz_vac, d_PMMA, dT, wp=1, t_step=time_s)
# monomer_matrix_2d_final = np.load('monomer_matrix_2d_final.npy')

# %%
plt.figure(dpi=300)
# plt.plot(mapping.x_centers_5nm, monomer_matrix_2d_final[:, 0])
plt.plot(mapping.x_centers_25nm,  get_25nm_array(monomer_matrix_2d_final[:, 0]))
plt.show()


# %%
def get_zz_vac_monomer_matrix(zz_vac_old, mon_matrix_2d):

    n_monomers_out = get_50nm_array(mon_matrix_2d[:, 0])
    V_monomer_out = n_monomers_out * constants.V_mon
    dh_monomer_out = V_monomer_out / (mapping.l_y * 1e-7 * mapping.step_50nm * 1e-7)

    zz_vac_new = zz_vac_old + dh_monomer_out
    mon_matrix_2d_final = copy.deepcopy(mon_matrix_2d)
    mon_matrix_2d_final[:, 0] = 0

    return zz_vac_new, mon_matrix_2d_final


# %%
zz_vac = np.zeros(len(mapping.x_centers_50nm))

zz_vac_new_new, monomer_matrix_2d_new = get_zz_vac_monomer_matrix(zz_vac, monomer_matrix_2d_final)

plt.figure(dpi=300)
plt.plot(mapping.x_centers_50nm, zz_vac_new_new)
plt.show()

# %%
zz_vac_evolver = 80e-7 - zz_vac_new_new

# ef.create_datafile(mapping.x_centers_50nm * 1e-3, zz_vac_evolver * 1e+4, mobs_50nm_fit * 1e+6)
ef.create_datafile(mapping.x_centers_50nm * 1e-3, zz_vac_evolver * 1e+4, mobs_50nm_fit)
ef.run_evolver()

# %%
tt, pp = ef.get_evolver_times_profiles()

plt.figure(dpi=300)
plt.plot(pp[0][:, 0], pp[0][:, 1], 'o-')
plt.plot(pp[1][:, 0], pp[1][:, 1], 'o-')
plt.show()

# %%
xx_final = pp[1][:, 0]
zz_vac_final = 80e-7 - pp[1][:, 1]*1e-4

plt.figure(dpi=300)
plt.plot(xx_final, zz_vac_final)
plt.show()

zz_vac = zz_vac_final



