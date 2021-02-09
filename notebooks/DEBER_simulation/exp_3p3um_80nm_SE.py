import importlib
import constants
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import copy
from mapping import mapping_3p3um_80nm as mm
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import MC_functions as mcf
from functions import DEBER_functions as deber
from functions import mapping_functions as mf
from functions import diffusion_functions as df
from functions import reflow_functions as rf
from functions import plot_functions as pf
from functions import SE_functions as ef
from functions import scission_functions as sf

import indexes as ind

mm = importlib.reload(mm)
deber = importlib.reload(deber)
emf = importlib.reload(emf)
mcf = importlib.reload(mcf)
ind = importlib.reload(ind)
af = importlib.reload(af)
mf = importlib.reload(mf)
df = importlib.reload(df)
ef = importlib.reload(ef)
rf = importlib.reload(rf)
pf = importlib.reload(pf)
sf = importlib.reload(sf)

# %%
lx = mm.lx * 1e-7
ly = mm.ly * 1e-7
area = lx * ly

d_PMMA = mm.z_max
d_PMMA_cm = d_PMMA * 1e-7  # cm

j_exp_s = 1.9e-9  # A / cm
j_exp_l = 1.9e-9 * lx
dose_s = 0.6e-6  # C / cm^2
dose_l = dose_s * lx
t = dose_l / j_exp_l  # 316 s
dt = 1  # s
Q = dose_s * area
n_electrons = Q / constants.e_SI  # 2 472
n_electrons_s = int(np.around(n_electrons / t))

T_C = 125
Tg = 120
dT = T_C - Tg

time_s = 10

# %%
xx = mm.x_bins_10nm  # nm
zz_vac = np.ones(len(xx)) * 0  # nm
zz_vac_list = [zz_vac]

file_cnt = 0
n_files = 3200
primary_electrons_in_file = 10

zip_length = 1000
r_beam = 100
weight = 0.3
source = 'data/e_DATA_Pv_80nm/'

scission_matrix = np.zeros(mm.hist_10nm_shape)
monomer_matrix_2d = np.zeros(np.shape(np.sum(scission_matrix, axis=1)))

for i in range(1):
    # i = 0

    # plt.figure(dpi=300)
    # plt.plot(xx, zz_vac)
    # plt.title('suda herachat electrons, i = ' + str(i))
    # plt.show()

    print('!!!!!!!!!', i, '!!!!!!!!!')

    for _ in range(8):
        now_DATA = np.load(source + 'e_DATA_' + str(file_cnt % n_files) + '.npy')
        file_cnt += 1

        if file_cnt > n_files:
            emf.rotate_DATA(now_DATA)

        for primary_e_id in range(primary_electrons_in_file):

            now_prim_e_DATA = emf.get_e_id_e_DATA(now_DATA, primary_electrons_in_file, primary_e_id)

            emf.add_gaussian_xy_shift_to_track(now_prim_e_DATA, 0, r_beam, [mm.y_min, mm.y_max])

            af.snake_array(
                array=now_prim_e_DATA,
                x_ind=ind.e_DATA_x_ind,
                y_ind=ind.e_DATA_y_ind,
                z_ind=ind.e_DATA_z_ind,
                xyz_min=[mm.x_min, mm.y_min, -np.inf],
                xyz_max=[mm.x_max, mm.y_max, np.inf]
            )

            for pos, line in enumerate(now_prim_e_DATA):

                now_x, now_z = line[ind.e_DATA_x_ind], line[ind.e_DATA_z_ind]

                if now_z < mcf.lin_lin_interp(xx, zz_vac)(now_x):
                    now_prim_e_DATA[pos, ind.e_DATA_process_id_ind] = 0  # change type to simulate zz_vac

            now_prim_e_val_DATA = \
                now_prim_e_DATA[np.where(now_prim_e_DATA[:, ind.e_DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]

            scission_matrix += np.histogramdd(
                sample=now_prim_e_val_DATA[:, ind.e_DATA_coord_inds],
                bins=mm.bins_10nm,
                weights=sf.get_scissions(now_prim_e_val_DATA, weight=weight)
            )[0]

    print('scission matrix is obtained, sum =', np.sum(scission_matrix))

    scission_array = np.average(np.average(scission_matrix, axis=1), axis=1)
    scission_array_100nm = df.move_10nm_to_100nm(scission_array)
    sci_fit_params = rf.get_fit_params_sci_arr(mm.x_centers_100nm, scission_array_100nm)
    scission_array_50nm_fit = rf.gauss(mm.x_centers_50nm, *sci_fit_params)

    # T = 120, zip_length = 1000
    scale_hack = 1e+6
    mobs_50nm = rf.move_sci_to_mobs(scission_array_50nm_fit, 120, 1000) / scale_hack

    plt.figure(dpi=300)
    plt.semilogy(mm.x_centers_50nm, mobs_50nm)
    plt.title('mobs_50nm, i = ' + str(i))
    plt.show()

    now_monomer_matrix_2d = np.sum(scission_matrix, axis=1) * zip_length
    monomer_matrix_2d += now_monomer_matrix_2d

    print('simulate diffusion ...')
    monomer_portion = 100
    monomer_matrix_2d_final = df.track_all_monomers(monomer_matrix_2d, xx, zz_vac, d_PMMA, dT,
                                                    wp=1, t_step=time_s, dtdt=0.5, n_hack=monomer_portion)
    print('diffusion is simulated')

    zz_vac_50nm = mcf.lin_lin_interp(xx, zz_vac)(mm.x_centers_50nm)
    zz_vac_new_50nm, monomer_matrix_2d_final =\
        df.get_zz_vac_50nm_monomer_matrix(zz_vac_50nm, monomer_matrix_2d_final, n_hack=monomer_portion)

    monomer_matrix_2d = monomer_matrix_2d_final
    print(np.sum(monomer_matrix_2d))

    plt.figure(dpi=300)
    plt.plot(mm.x_centers_50nm, zz_vac_new_50nm, 'o-')
    plt.title('- profile after diffusion, i = ' + str(i))
    plt.show()

    zz_vac_evolver = 80 - zz_vac_new_50nm * 3

    plt.figure(dpi=300)
    plt.plot(mm.x_centers_50nm, zz_vac_evolver, 'o-')
    plt.title('profile before SE, i = ' + str(i))
    plt.show()

    # scission_array = np.sum(np.sum(scission_matrix, axis=1), axis=1)
    # scission_array_50nm = df.move_10nm_to_50nm(scission_array)

    evolver_yy = np.concatenate([[mm.x_min], mm.x_centers_50nm, [mm.x_max]])
    evolver_zz = np.concatenate([[zz_vac_evolver[0]], zz_vac_evolver, [zz_vac_evolver[-1]]])
    evolver_mobs = np.concatenate([[mobs_50nm[0]], mobs_50nm, [mobs_50nm[-1]]])

    ef.create_datafile_non_period(evolver_yy, evolver_zz, evolver_mobs)


# %%
def exp_gauss(x, a, b, c):
    return np.exp(a + b / c * np.exp(-x**2 / c**2))


popt = curve_fit(exp_gauss, evolver_yy, evolver_mobs, p0=[-30, 378, 150])[0]

plt.figure(dpi=300)
plt.plot(evolver_yy, evolver_mobs, 'o-')
plt.plot(evolver_yy, exp_gauss(evolver_yy, *popt))
# plt.plot(evolver_yy, exp_gauss(evolver_yy, -30.4, 658, 200))
plt.show()


# %%
yy_pre = np.linspace(-mm.lx / 40, mm.lx / 40, 20)
yy_beg = np.linspace(-mm.lx / 2, -mm.lx / 40 - 1, 40)
yy_end = np.linspace(mm.lx / 40 + 1, mm.lx / 2, 40)
yy_pre = np.concatenate([yy_beg, yy_pre, yy_end]) * 1e-3

zz_pre = mcf.lin_lin_interp(evolver_yy, evolver_zz * 1e-3)(yy_pre / 1e-3)
mobs = mcf.lin_log_interp(evolver_yy, evolver_mobs)(yy_pre)
z_max = np.max(zz_pre)


def exp_gauss(x, a, b, c):
    return np.exp(a + b / c * np.exp(-x ** 2 / c ** 2))


# popt = curve_fit(exp_gauss, yy_pre, mobs)[0]
popt = curve_fit(exp_gauss, yy_pre * 1e+3, mobs, p0=[-30, 378, 150])[0]

plt.figure(dpi=300)
plt.plot()
# plt.plot(yy_pre, mobs, 'o-')
# plt.plot(yy_pre, exp_gauss(yy_pre * 1e+3, *popt))
# plt.plot(yy_pre, exp_gauss(yy_pre, -30.4, 658, 200))
plt.show()


# %%
    # ef.create_datafile_non_period(evolver_yy, evolver_zz, evolver_mobs)
    # ef.run_evolver()

# %%
tt, pp = ef.get_evolver_times_profiles()
xx_final = pp[1][:, 0] * 1e+3  # um -> nm
zz_vac_final = 80 - pp[1][:, 1] * 1e+3  # um -> nm

xx_final = np.concatenate(([mm.x_min], xx_final, [mm.x_max]))
zz_vac_final = np.concatenate(([zz_vac_final[0]], zz_vac_final, [zz_vac_final[-1]]))

plt.figure(dpi=300)
plt.plot(mm.x_centers_50nm, zz_vac_evolver, 'o-')
plt.plot(xx_final, 80 - zz_vac_final, 'o-')
plt.title('profile after SE, i = ' + str(i))
plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.grid()
plt.show()

zz_vac = mcf.lin_lin_interp(xx_final, zz_vac_final)(xx)
zz_vac_list.append(zz_vac)
