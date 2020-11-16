# %%
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
from functions import SE_functions as ef

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
l_x = mapping.l_x * 1e-7
l_y = mapping.l_y * 1e-7
area = l_x * l_y
d_PMMA = mapping.z_max * 1e-7  # cm
j_exp_s = 1.9e-9  # A / cm
j_exp_l = 1.9e-9 * l_x
dose_s = 0.6e-6  # C / cm^2
dose_l = dose_s * l_x
t = dose_l / j_exp_l  # 316 s
dt = 1  # s
Q = dose_s * area
n_electrons = Q / constants.e_SI  # 2 472
n_electrons_s = int(np.around(n_electrons / t))

r_beam = 150e-7

zip_length = 1000
T_C = 125
Tg = 120
dT = T_C - Tg

time_s = 10

xx = mapping.x_centers_50nm * 1e-7  # cm
zz_vac = np.zeros(len(xx))  # cm
monomer_matrix = np.zeros(np.shape(resist_matrix)[:3])

zz_vac_list = [zz_vac]

for i in range(32):

    print('!!!!!!!!!', i, '!!!!!!!!!')

    plt.figure(dpi=300)
    plt.plot(xx, zz_vac)
    plt.title('suda herachat electrons, i = ' + str(i))
    plt.show()

    print('getting e_DATA ...')
    e_DATA, e_DATA_PMMA_val = deber.get_e_DATA_PMMA_val(
        xx=xx,
        zz_vac=zz_vac,
        d_PMMA=d_PMMA,
        n_electrons=n_electrons_s*time_s,
        E0=20e+3,
        r_beam=r_beam
    )
    print('e_DATA is obtained')

    print('getting scission_matrix ...')
    scission_matrix, E_dep_matrix = deber.get_scission_matrix(e_DATA, weight=0.35)
    print('scission_matrix is obtained')

    print('process mapping ...')
    mf.process_mapping(scission_matrix, resist_matrix, chain_tables)
    print('mapping is carried out')

    print('process depolymerization ...')
    mf.process_depolymerization(resist_matrix, chain_tables, zip_length)
    print('depolymerization is carried out')

    print('getting chain len matrix ...')
    sum_m, sum_m2, new_monomer_matrix = mf.get_chain_len_matrix(resist_matrix, chain_tables)
    print('chain len matrix is obtained')

    monomer_matrix += new_monomer_matrix

    sum_m_1d = np.sum(np.sum(sum_m, axis=1), axis=1)
    sum_m2_1d = np.sum(np.sum(sum_m2, axis=1), axis=1)
    monomer_matrix_1d = np.sum(np.sum(monomer_matrix, axis=1), axis=1)

    sum_m_1d_100nm = df.get_100nm_array(sum_m_1d)
    sum_m2_1d_100nm = df.get_100nm_array(sum_m2_1d)
    monomer_matrix_1d_100nm = df.get_100nm_array(monomer_matrix_1d)

    print('getting local Mw matrix ...')
    matrix_Mw_1d_100nm = mf.get_local_Mw_matrix(sum_m_1d_100nm, sum_m2_1d_100nm, monomer_matrix_1d_100nm)
    np.save('matrix_Mw_1d_100nm_' + str(i) + '.npy', matrix_Mw_1d_100nm)
    print('local Mw matrix is obtained')

    # popt, _ = curve_fit(df.minus_exp_gauss, mapping.x_centers_100nm, matrix_Mw_1d_100nm)
    beg_ind = 1
    popt, _ = curve_fit(df.exp_gauss, mapping.x_centers_100nm[beg_ind:-beg_ind],
                        matrix_Mw_1d_100nm[beg_ind:-beg_ind], p0=[13.8, 0.5, 300])
    print(popt)

    plt.figure(dpi=300)
    plt.semilogy(mapping.x_centers_100nm, matrix_Mw_1d_100nm, '-o')
    plt.semilogy(mapping.x_centers_50nm, df.exp_gauss(mapping.x_centers_50nm, *popt), '.-')
    plt.title('Mw gauss fit, i = ' + str(i))
    plt.show()

    etas_50nm_fit = rf.get_viscosity_W(T_C, df.exp_gauss(mapping.x_centers_50nm, *popt))
    mobs_50nm_fit = rf.get_SE_mobility(etas_50nm_fit)

    # plt.figure(dpi=300)
    # plt.semilogy(mapping.x_centers_50nm, mobs_50nm_fit, 'o-')
    # plt.title('obtained mobilities, i = ' + str(i))
    # plt.show()

    print('simulate diffusion ...')
    monomer_matrix_2d = np.sum(monomer_matrix, axis=1)
    monomer_matrix_2d_final = df.track_all_monomers(monomer_matrix_2d,
                                                    xx, zz_vac, d_PMMA, dT, wp=1, t_step=time_s, dtdt=0.5)
    print('diffusion is simulated')

    # zz_vac = np.zeros(len(mapping.x_centers_50nm))  # 50 nm !!!
    zz_vac_new, monomer_matrix_2d_new = df.get_zz_vac_monomer_matrix(zz_vac, monomer_matrix_2d_final)

    plt.figure(dpi=300)
    plt.plot(mapping.x_centers_50nm, zz_vac_new)
    plt.title('profile after diffusion, i = ' + str(i))
    plt.show()

    zz_vac_evolver = 80e-7 - zz_vac_new

    plt.figure(dpi=300)
    plt.plot(mapping.x_centers_50nm, zz_vac_evolver)
    plt.title('profile before SE, i = ' + str(i))
    plt.show()

    ef.create_datafile(mapping.x_centers_50nm * 1e-3, zz_vac_evolver * 1e+4, mobs_50nm_fit)
    ef.run_evolver()

    tt, pp = ef.get_evolver_times_profiles()

    xx_final_cm = pp[1][:, 0] * 1e-4
    zz_vac_final_cm = 80e-7 - pp[1][:, 1]*1e-4

    xx_final_cm = np.concatenate(([mapping.x_min * 1e-7], xx_final_cm, [mapping.x_max * 1e-7]))
    zz_vac_final_cm = np.concatenate(([zz_vac_final_cm[0]], zz_vac_final_cm, [zz_vac_final_cm[-1]]))

    plt.figure(dpi=300)
    plt.plot(xx_final_cm * 1e+7, 80 - zz_vac_final_cm * 1e+7)
    plt.title('profile after SE, i = ' + str(i))
    plt.show()

    zz_vac = mcf.lin_lin_interp(xx_final_cm, zz_vac_final_cm)(xx)
    zz_vac_list.append(zz_vac)

# %%
tt, pp = ef.get_evolver_times_profiles()
xx_final_cm = pp[1][1:, 0] * 1e-4
zz_vac_final_cm = 80e-7 - pp[1][1:, 1]*1e-4

# %%
# plt.figure(dpi=300)
# plt.plot(xx_final_cm, zz_vac_final_cm)
# plt.show()

ind = 15

plt.figure(dpi=300)
plt.plot(pp[1][1:, 0], pp[1][1:, 1], '.')
plt.plot(pp[ind][1:, 0], pp[ind][1:, 1], '.')
# plt.plot(xx_final_cm * 1e+7, zz_vac_final_cm * 1e+7)
# plt.plot(xx_final_cm * 1e+7, pp[1][:, 1] * 1e-4 * 1e+7)
plt.title('after SE, i = ' + str(i))
plt.show()

