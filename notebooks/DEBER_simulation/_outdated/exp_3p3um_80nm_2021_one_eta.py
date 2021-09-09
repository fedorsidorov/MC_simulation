import importlib
import matplotlib.pyplot as plt
import numpy as np
import constants as const
from mapping import mapping_3p3um_80nm as mm
from functions import array_functions as af
from functions import MC_functions as mcf
from functions import e_matrix_functions as emf
from functions import scission_functions as sf
from functions import reflow_functions as rf
from functions import SE_functions as ef
from functions import e_beam_MC as eb_MC

import indexes as ind

const = importlib.reload(const)
eb_MC = importlib.reload(eb_MC)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
mcf = importlib.reload(mcf)
mm = importlib.reload(mm)
af = importlib.reload(af)
ef = importlib.reload(ef)
sf = importlib.reload(sf)
rf = importlib.reload(rf)

# %%
j_exp_s = 1.9e-9  # A / cm^2
j_exp_l = j_exp_s * mm.lx_cm  # A / cm
dose_s = 0.6e-6  # C / cm^2
dose_l = dose_s * mm.lx_cm  # C / cm
total_time = dose_l / j_exp_l  # 316 s
Q = dose_s * mm.area_cm2
n_electrons = Q / const.e_SI  # 24 716
n_electrons_s = int(np.around(n_electrons / total_time))

step_time = 10
primary_electrons_in_file = n_electrons_s * step_time

# zip_length = 456
zip_length = 1000
r_beam = 100
scission_weight = 0.082  # 125 C

r_beam_x = 100
r_beam_y = mm.ly / 2

E0 = 20e+3

SE_mobility = 1e-1

hack_factor = 5

# %%
xx_vac = mm.x_centers_20nm  # nm
zz_vac = np.ones(len(xx_vac)) * 0  # nm
zz_vac_list = [zz_vac]

total_tau_array = np.zeros(len(mm.x_centers_20nm))

for n_step in range(32):

    xx_vac_for_sim = np.concatenate(([-1e+6], xx_vac, [1e+6]))
    zz_vac_for_sim = np.concatenate(([zz_vac[0]], zz_vac, [zz_vac[-1]]))

    now_e_DATA = eb_MC.track_all_electrons(
        n_electrons=int(primary_electrons_in_file / 2 / hack_factor),
        E0=E0,
        d_PMMA=mm.d_PMMA,
        z_cut=np.inf,
        Pn=True,
        xx_vac=xx_vac_for_sim,
        zz_vac=zz_vac_for_sim,
        r_beam_x=r_beam_x,
        r_beam_y=r_beam_y
    )

    now_Pv_e_DATA = now_e_DATA[np.where(
        np.logical_and(
            now_e_DATA[:, ind.e_DATA_layer_id_ind] == ind.PMMA_ind,
            now_e_DATA[:, ind.e_DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind))
    ]

    af.snake_array(
            array=now_Pv_e_DATA,
            x_ind=ind.e_DATA_x_ind,
            y_ind=ind.e_DATA_y_ind,
            z_ind=ind.e_DATA_z_ind,
            xyz_min=[mm.x_min, mm.y_min, -np.inf],
            xyz_max=[mm.x_max, mm.y_max, np.inf]
        )

    now_Pv_e_DATA = emf.delete_snaked_vacuum_events(now_Pv_e_DATA, xx_vac_for_sim, zz_vac_for_sim)

    val_matrix = np.histogramdd(
        sample=now_Pv_e_DATA[:, [ind.e_DATA_x_ind, ind.e_DATA_z_ind]],
        bins=[mm.x_bins_20nm, mm.z_bins_20nm]
    )[0]

    val_matrix += val_matrix[::-1, :]
    val_matrix *= hack_factor

    scission_matrix = np.zeros(np.shape(val_matrix), dtype=int)

    for x_ind in range(len(val_matrix)):
        for z_ind in range(len(val_matrix[0])):
            n_val = int(val_matrix[x_ind, z_ind])

            scissions = np.where(np.random.random(n_val) < scission_weight)[0]
            scission_matrix[x_ind, z_ind] = len(scissions)

    scission_array = np.sum(scission_matrix, axis=1)

    monomer_matrix = scission_matrix * zip_length
    monomers_array = np.zeros(len(mm.x_centers_20nm))

    for x_ind in range(len(monomer_matrix)):
        for z_ind in range(len(monomer_matrix[0])):
            now_x = mm.x_centers_20nm[x_ind]
            now_z = mm.z_centers_20nm[z_ind]
            now_gauss_scale = now_z

            now_n_monomers = monomer_matrix[x_ind, z_ind]
            now_escape_x_arr = np.random.normal(now_x, now_gauss_scale, now_n_monomers)

            monomers_array += np.histogram(now_escape_x_arr, bins=mm.x_bins_20nm)[0]

    delta_z_vac_array = monomers_array * const.V_mon_nm3 / mm.step_20nm / mm.ly
    zz_vac_after_diffusion = zz_vac + delta_z_vac_array

    zz_vac = zz_vac_after_diffusion
    zz_vac_list.append(zz_vac)

    zz_evolver = mm.d_PMMA - zz_vac_after_diffusion

    # go to um!!!
    xx_evolver_final = np.concatenate([[mm.x_bins_20nm[0]], mm.x_centers_20nm, [mm.x_bins_20nm[-1]]]) * 1e-3
    zz_evolver_final = np.concatenate([[zz_evolver[0]], zz_evolver, [zz_evolver[-1]]]) * 1e-3
    mobs_evolver_final = np.ones(len(xx_evolver_final)) * SE_mobility

    file_full_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/datafile_DEBER_2021.fe'
    commands_full_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/commands.txt'

    ef.create_datafile_latest(
        yy=xx_evolver_final,
        zz=zz_evolver_final,
        width=mm.ly * 1e-3,
        mobs=mobs_evolver_final,
        path=file_full_path
    )

    ef.run_evolver(file_full_path, commands_full_path)

    SE = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_single.txt')
    SE = SE[np.where(
        np.logical_and(
            np.abs(SE[:, 0]) < 0.1,
            SE[:, 2] > 1e-3
        ))]

    xx_SE = SE[1:, 1]
    zz_SE = SE[1:, 2]

    sort_inds = np.argsort(xx_SE)
    xx_SE_sorted = xx_SE[sort_inds]
    zz_SE_sorted = zz_SE[sort_inds]

    xx_SE_sorted[0] = -mm.lx * 1e-3 / 2
    xx_SE_sorted[-1] = mm.lx * 1e-3 / 2

    zz_SE_sorted[0] = zz_SE_sorted[1]
    zz_SE_sorted[-1] = zz_SE_sorted[-2]

    zz_vac_final = mcf.lin_lin_interp(xx_SE_sorted * 1e+3, zz_SE_sorted * 1e+3)(xx_vac)

    plt.figure(dpi=300)
    plt.plot(mm.x_centers_20nm, zz_evolver, label='after monomer diffusion')
    plt.plot(xx_vac, zz_vac_final, label='after Surface Evolver')
    plt.title('step ' + str(n_step))
    plt.grid()
    plt.xlabel('x, nm')
    plt.xlabel('z, nm')
    plt.legend()
    plt.show()

    plt.figure(dpi=300)
    plt.plot(mm.x_centers_20nm, zz_evolver, label='after monomer diffusion')
    plt.plot(xx_vac, zz_vac_final, label='after Surface Evolver')
    plt.title('step ' + str(n_step))
    plt.grid()
    plt.xlabel('x, nm')
    plt.xlabel('z, nm')
    plt.legend()
    plt.savefig('notebooks/DEBER_simulation/1000_1e-2/' + str(n_step) + '.jpg', dpi=300)

    zz_vac = mm.d_PMMA - zz_vac_final
    zz_vac_list.append(zz_vac)
