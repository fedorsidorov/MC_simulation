import importlib
import constants
import matplotlib.pyplot as plt
import numpy as np
from mapping import mapping_3p3um_80nm as mapping
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import MC_functions as mcf
from functions import DEBER_functions as deber
from functions import mapping_functions as mf
from functions import diffusion_functions as df
from functions import reflow_functions as rf
from functions import plot_functions as pf
from functions._outdated import SE_functions as ef
from functions import scission_functions as sf

import indexes as ind

mapping = importlib.reload(mapping)
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
l_x = mapping.l_x * 1e-7
l_y = mapping.l_y * 1e-7
area = l_x * l_y

d_PMMA = mapping.z_max
d_PMMA_cm = d_PMMA * 1e-7  # cm

j_exp_s = 1.9e-9  # A / cm
j_exp_l = 1.9e-9 * l_x
dose_s = 0.6e-6  # C / cm^2
dose_l = dose_s * l_x
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
xx = mapping.x_bins_10nm  # nm
zz_vac = np.ones(len(xx)) * 0  # nm
zz_vac_list = [zz_vac]

file_cnt = 0
n_files = 3200
primary_electrons_in_file = 10

zip_length = 1000
r_beam = 100
weight = 0.3
source = 'data/e_DATA_Pn_80nm_point/'

scission_matrix = np.zeros(mapping.hist_10nm_shape)
monomer_matrix_2d = np.zeros(np.shape(np.sum(scission_matrix, axis=1)))

for i in range(32):
    # i = 0

    # plt.figure(dpi=300)
    # plt.plot(xx, zz_vac)
    # plt.title('suda herachat electrons, i = ' + str(i))
    # plt.show()

    print('!!!!!!!!!', i, '!!!!!!!!!')

    for _ in range(8):
        now_DATA = np.load(source + 'e_DATA_Pn_' + str(file_cnt % n_files) + '.npy')
        file_cnt += 1

        if file_cnt > n_files:
            emf.rotate_DATA(now_DATA)

        for primary_e_id in range(primary_electrons_in_file):

            now_prim_e_DATA = emf.get_e_id_DATA_corr(now_DATA, primary_electrons_in_file, primary_e_id)

            emf.add_gaussian_xy_shift_to_track(now_prim_e_DATA, 0, r_beam, [mapping.y_min, mapping.y_max])

            af.snake_array(
                array=now_prim_e_DATA,
                x_ind=ind.DATA_x_ind,
                y_ind=ind.DATA_y_ind,
                z_ind=ind.DATA_z_ind,
                xyz_min=[mapping.x_min, mapping.y_min, -np.inf],
                xyz_max=[mapping.x_max, mapping.y_max, np.inf]
            )
#
            for pos, line in enumerate(now_prim_e_DATA):

                now_x, now_z = line[ind.DATA_x_ind], line[ind.DATA_z_ind]

                if now_z < mcf.lin_lin_interp(xx, zz_vac)(now_x):
                    now_prim_e_DATA[pos, ind.DATA_process_id_ind] = 0  # change type to simulate zz_vac

            now_prim_e_val_DATA = \
                now_prim_e_DATA[np.where(now_prim_e_DATA[:, ind.DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]

            scission_matrix += np.histogramdd(
                sample=now_prim_e_val_DATA[:, ind.DATA_coord_inds],
                bins=mapping.bins_10nm,
                weights=sf.get_scissions_easy(now_prim_e_val_DATA, weight=weight)
            )[0]

    print('scission matrix is obtained, sum =', np.sum(scission_matrix))

    # scission_array = np.average(np.average(scission_matrix, axis=1), axis=1)
    # scission_array_100nm = df.move_10nm_to_100nm(scission_array)
    # sci_fit_params = rf.get_fit_params_sci_arr(mapping.x_centers_100nm, scission_array_100nm)
    # scission_array_50nm_fit = rf.gauss(mapping.x_centers_50nm, *sci_fit_params)

    # sci_fit_params = rf.get_fit_params_zero_sci_arr(mapping.x_centers_100nm, scission_array_100nm)
    # scission_array_50nm_fit = rf.gauss_zero(mapping.x_centers_50nm, *sci_fit_params)

    # plt.figure(dpi=300)
    # plt.plot(mapping.x_centers_100nm, scission_array_100nm)
    # plt.plot(mapping.x_centers_50nm, scission_array_50nm_fit)
    # plt.title('scission_array_100nm, i = ' + str(i))
    # plt.show()

    # T = 120, zip_length = 1000
    # scale_hack = 1e+6
    # mobs_50nm = rf.move_sci_to_mobs(scission_array_50nm_fit, 120, 1000) / scale_hack
    mob = rf.move_sci_to_mobs(np.average(scission_matrix), 160, 1500)
    print('now mobility is', mob)
    mobs_50nm = np.ones(len(mapping.x_centers_50nm)) * mob

    # scission_array_50nm = df.move_10nm_to_50nm(np.average(np.average(scission_matrix, axis=1), axis=1))

    # mobs_50nm = np.zeros(len(scission_array_50nm))
    # mobs_50nm = np.ones(len(scission_array_50nm)) * np.average(scission_array_50nm[23:-23])

    # avg_edges = (np.average(scission_array_50nm[:23]) + np.average(scission_array_50nm[-23:])) / 2
    # avg_mob = rf.move_sci_to_mobs(avg_edges, 120, 1000)
    # mobs_50nm[:23] = avg_mob
    # mobs_50nm[-23:] = avg_mob
    # mobs_50nm[:-23] = rf.move_sci_to_mobs(np.average(scission_array_50nm[23:-23]), 120, 1000)
    # plt.figure(dpi=300)
    # plt.plot(mobs_50nm)
    # plt.title('mobs_50nm, i = ' + str(i))
    # plt.show()

    now_monomer_matrix_2d = np.sum(scission_matrix, axis=1) * zip_length
    monomer_matrix_2d += now_monomer_matrix_2d

    print('simulate diffusion ...')
    monomer_portion = 10
    monomer_matrix_2d_final = df.track_all_monomers(monomer_matrix_2d, xx, zz_vac, d_PMMA, dT,
                                                    wp=1, t_step=time_s, dtdt=0.5, n_hack=monomer_portion)
    print('diffusion is simulated')

    zz_vac_50nm = mcf.lin_lin_interp(xx, zz_vac)(mapping.x_centers_50nm)
    zz_vac_new_50nm, monomer_matrix_2d_final =\
        df.get_zz_vac_50nm_monomer_matrix(zz_vac_50nm, monomer_matrix_2d_final, n_hack=monomer_portion)

    monomer_matrix_2d = monomer_matrix_2d_final
    print(np.sum(monomer_matrix_2d))

    # plt.figure(dpi=300)
    # plt.plot(mapping.x_centers_50nm, zz_vac_new_50nm, 'o-')
    # plt.title('- profile after diffusion, i = ' + str(i))
    # plt.show()

    zz_vac_evolver = 80 - zz_vac_new_50nm

    # plt.figure(dpi=300)
    # plt.plot(mapping.x_centers_50nm, zz_vac_evolver, 'o-')
    # plt.title('profile before SE, i = ' + str(i))
    # plt.show()

    # scission_array = np.sum(np.sum(scission_matrix, axis=1), axis=1)
    # scission_array_50nm = df.move_10nm_to_50nm(scission_array)

    ef.create_datafile(mapping.x_centers_50nm * 1e-3, zz_vac_evolver * 1e-3, mobs_50nm)
    ef.run_evolver()

    tt, pp = ef.get_evolver_times_profiles()
    xx_final = pp[1][:, 0] * 1e+3  # um -> nm
    zz_vac_final = 80 - pp[1][:, 1] * 1e+3  # um -> nm

    xx_final = np.concatenate(([mapping.x_min], xx_final, [mapping.x_max]))
    zz_vac_final = np.concatenate(([zz_vac_final[0]], zz_vac_final, [zz_vac_final[-1]]))

    plt.figure(dpi=300)
    plt.plot(mapping.x_centers_50nm, zz_vac_evolver, 'o-')
    plt.plot(xx_final, 80 - zz_vac_final, 'o-')
    plt.title('profile after SE, i = ' + str(i))
    plt.show()

    zz_vac = mcf.lin_lin_interp(xx_final, zz_vac_final)(xx)
    zz_vac_list.append(zz_vac)
