import importlib
import matplotlib.pyplot as plt
import numpy as np
import constants as const
from mapping import mapping_5um_900nm as mm
from functions import array_functions as af
from functions import MC_functions as mcf
from functions import e_matrix_functions as emf
from functions import scission_functions as sf
from functions import reflow_functions as rf
from functions import SE_functions as ef
from functions import e_beam_MC as eb_MC
from functions import diffusion_functions as df
from tqdm import tqdm
import indexes as ind

const = importlib.reload(const)
eb_MC = importlib.reload(eb_MC)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
mcf = importlib.reload(mcf)
mm = importlib.reload(mm)
af = importlib.reload(af)
df = importlib.reload(df)
ef = importlib.reload(ef)
sf = importlib.reload(sf)
rf = importlib.reload(rf)


# %%
def get_resist_fraction_matrix(zz_vac):
    resist_fraction_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))

    for ii in range(len(mm.x_centers_5nm)):
        beg_ind = np.where(mm.z_bins_5nm > zz_vac[ii])[0][0]
        resist_fraction_matrix[ii, beg_ind:] = 1

        if beg_ind > 0:
            resist_fraction_matrix[ii, :beg_ind - 1] = 0
            trans_resist_fraction = 1 - (zz_vac[ii] - mm.z_bins_5nm[beg_ind - 1]) / mm.step_5nm
            resist_fraction_matrix[ii, beg_ind - 1] = trans_resist_fraction

    return resist_fraction_matrix


def get_Mn_true_Mn_matrix(resist_fraction_matrix, free_monomer_in_resist_matrix, tau_matrix):
    Mn_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))
    true_Mn_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))

    print('Get Mn and true_Mn matrices')
    progress_bar = tqdm(total=len(mm.x_centers_5nm), position=0)

    for ii in range(len(mm.x_centers_5nm)):
        for kk in range(len(mm.z_centers_5nm)):

            if resist_fraction_matrix[ii, kk] == 0:
                continue

            if tau_matrix[ii, kk] > tau_125[-1]:
                Mn = Mn_125[-1]
            else:
                tau_ind = np.argmin(np.abs(tau_matrix[ii, kk] - tau_125))
                Mn = Mn_125[tau_ind]

            Mn_matrix[ii, kk] = Mn

            n_chains = y_slice_V / const.V_mon_nm3 / (Mn / MMA_weight)
            true_Mn = (n_chains * Mn + free_monomer_in_resist_matrix[ii, kk] * MMA_weight) / \
                      (n_chains + free_monomer_in_resist_matrix[ii, kk])

            true_Mn_matrix[ii, kk] = true_Mn

        progress_bar.update()

    return Mn_matrix, true_Mn_matrix


def get_eta_SE_mob_arrays(true_Mn_matrix, temp_C, viscosity_power):
    eta_array = np.zeros(len(mm.x_centers_5nm))
    SE_mob_array = np.zeros(len(mm.x_centers_5nm))

    for ii in range(len(mm.x_centers_5nm)):
        inds = np.where(true_Mn_matrix[ii, :] > 0)[0]
        Mn_avg = np.average(true_Mn_matrix[ii, inds])
        eta = rf.get_viscosity_experiment_Mn(temp_C, Mn_avg, viscosity_power)

        eta_array[ii] = eta
        SE_mob_array[ii] = rf.get_SE_mobility(eta)

    return eta_array, SE_mob_array


def get_delta_tau_matix(resist_monomer_matrix, scission_matrix):
    ks_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))
    inds_mon = np.where(resist_monomer_matrix > 0)
    ks_matrix[inds_mon] = scission_matrix[inds_mon] / resist_monomer_matrix[inds_mon] / step_time
    delta_tau_matrix = y_0 * ks_matrix * step_time
    return delta_tau_matrix


def get_wp_D_matrix(global_free_monomer_in_resist_matrix, resist_monomer_matrix, temp_C, D_factor):
    wp_matrix = np.zeros(np.shape(now_free_monomer_matrix_0))
    D_matrix = np.zeros(np.shape(now_free_monomer_matrix_0))

    for ii in range(len(mm.x_centers_5nm)):
        for kk in range(len(mm.z_centers_5nm)):

            if resist_monomer_matrix[ii, kk] == 0:
                continue

            n_free_monomers = global_free_monomer_in_resist_matrix[ii, kk]
            wp = 1 - n_free_monomers / resist_monomer_matrix[ii, kk]

            if wp <= 0:
                # print('now_wp <= 0')
                wp = 0

            D = df.get_D(temp_C, wp)

            wp_matrix[ii, kk] = wp
            D_matrix[ii, kk] = D * D_factor

    return wp_matrix, D_matrix


def get_free_mon_matrix_mon_out_array_after_diffusion(global_free_monomer_in_resist_matrix,
                                                      D_matrix, xx_vac, zz_vac, d_PMMA):
    free_monomer_in_resist_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))
    out_monomer_array = np.zeros(len(mm.x_centers_5nm))

    print('Simulate diffusion')
    progress_bar = tqdm(total=len(mm.x_centers_5nm), position=0)

    for ii in range(len(mm.x_centers_5nm)):
        for kk in range(len(mm.z_centers_5nm)):

            n_free_monomers = int(global_free_monomer_in_resist_matrix[ii, kk])

            if n_free_monomers == 0:
                continue

            xx0_mon_arr = np.ones(n_free_monomers) * mm.x_centers_5nm[ii]
            zz0_mon_arr = np.ones(n_free_monomers) * mm.z_centers_5nm[kk]

            xx_mon = df.get_final_x_arr(x0_arr=xx0_mon_arr, D=D_matrix[ii, kk], delta_t=step_time)
            zz_mon = df.get_final_z_arr(z0_arr_raw=zz0_mon_arr, d_PMMA=d_PMMA, D=D_matrix[ii, kk], delta_t=step_time)

            af.snake_coord_1d(array=xx_mon, coord_min=mm.x_bins_5nm[0], coord_max=mm.x_bins_5nm[-1])

            zz_mon_vac = mcf.lin_lin_interp(xx_vac, zz_vac)(xx_mon)
            weights = zz_mon > zz_mon_vac
            xx_zz_mon = np.vstack([xx_mon, zz_mon]).transpose()

            if len(np.where(zz_mon > 900)[0]) > 0:
                print('here')

            free_monomer_in_resist_matrix += np.histogramdd(
                sample=xx_zz_mon,
                bins=[mm.x_bins_5nm, mm.z_bins_5nm],
                weights=weights.astype(int)
            )[0]

            out_monomer_array += np.histogram(
                a=xx_mon,
                bins=mm.x_bins_5nm,
                weights=np.logical_not(weights).astype(int)
            )[0]

        progress_bar.update()

    return free_monomer_in_resist_matrix, out_monomer_array


# %%
T_C = 160
zip_length = 10000
step_time = 5

MMA_weight = 100

y_slice_V = mm.step_5nm * mm.ly * mm.step_5nm

y_0 = 3989

tau_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Boyd_kinetic_curves/arrays/tau.npy')
Mn_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Boyd_kinetic_curves/arrays/Mn_125.npy')*100
Mw_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Boyd_kinetic_curves/arrays/Mw_125.npy')*100

global_tau_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))
global_free_monomer_in_resist_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)), dtype=int)
global_outer_monomer_array = np.zeros(len(mm.x_centers_5nm))

for n_step in range(1):

    # get new scission matrix
    xx_vac_raw = np.load('notebooks/DEBER_simulation/vary_zz_vac_0p2_5s/xx_vac.npy')
    now_zz_vac_raw = np.load('notebooks/DEBER_simulation/vary_zz_vac_0p2_5s/zz_vac_' + str(n_step) + '.npy')
    now_scission_matrix = \
        np.load('notebooks/DEBER_simulation/vary_zz_vac_0p2_5s/scission_matrix_' + str(n_step) + '.npy')

    now_xx_vac = mm.x_centers_5nm
    now_zz_vac = mcf.lin_lin_interp(xx_vac_raw, now_zz_vac_raw)(now_xx_vac)

    now_xx_vac_for_sim = np.concatenate(([mm.x_bins_5nm[0]], now_xx_vac, [mm.x_bins_5nm[-1]]))
    now_zz_vac_for_sim = np.concatenate(([now_zz_vac[0]], now_zz_vac, [now_zz_vac[-1]]))

    plt.figure(dpi=300)
    plt.plot(now_xx_vac, now_zz_vac)
    plt.xlabel('x, nm')
    plt.ylabel('y, nm')
    plt.grid()
    plt.show()

    # get resist fraction matrix
    now_resist_fraction_matrix = get_resist_fraction_matrix(now_zz_vac)
    now_resist_monomer_matrix = now_resist_fraction_matrix * y_slice_V / const.V_mon_nm3

    plt.figure(dpi=300)
    plt.imshow(now_resist_fraction_matrix.transpose())
    plt.show()

    # simulate depolymerization
    now_free_monomer_matrix_0 = (now_scission_matrix * zip_length * now_resist_fraction_matrix).astype(int)

    # now_outer_monomer_array = np.zeros(len(mm.x_centers_5nm))

    # correct global_free_monomers_in_resist matrix
    global_free_monomer_in_resist_matrix_corr = \
        (global_free_monomer_in_resist_matrix * now_resist_fraction_matrix).astype(int)

    # add monomers that are cut by new PMMA surface
    # now_outer_monomer_array +=\
    global_outer_monomer_array +=\
        np.sum(global_free_monomer_in_resist_matrix - global_free_monomer_in_resist_matrix_corr, axis=1)

    global_free_monomer_in_resist_matrix += now_free_monomer_matrix_0

    # update tau_matrix
    now_delta_tau_matrix = get_delta_tau_matix(now_resist_monomer_matrix, now_scission_matrix)
    global_tau_matrix += now_delta_tau_matrix

    # get Mn and true_Mn matrices
    now_Mn_matrix, now_true_Mn_matrix = \
        get_Mn_true_Mn_matrix(now_resist_fraction_matrix, global_free_monomer_in_resist_matrix, global_tau_matrix)

    # get eta and SE_mob arrays
    now_eta_array, now_SE_mob_array = get_eta_SE_mob_arrays(now_true_Mn_matrix, temp_C=T_C, viscosity_power=3.4)

    plt.figure(dpi=300)
    # plt.semilogy(mm.x_centers_5nm, now_eta_array)
    plt.semilogy(mm.x_centers_5nm, now_SE_mob_array)
    plt.xlabel('x, nm')
    plt.ylabel('SE mobility')
    plt.show()

    # get wp, D matrices
    now_wp_matrix, now_D_matrix =\
        get_wp_D_matrix(global_free_monomer_in_resist_matrix, now_resist_monomer_matrix, temp_C=T_C, D_factor=1e-4)

    # simulate diffusion
    now_free_monomer_in_resist_matrix, now_outer_monomer_array = get_free_mon_matrix_mon_out_array_after_diffusion(
        global_free_monomer_in_resist_matrix,
        now_D_matrix,
        xx_vac=now_xx_vac_for_sim,
        zz_vac=now_zz_vac_for_sim,
        d_PMMA=mm.d_PMMA
    )

    global_free_monomer_in_resist_matrix += now_free_monomer_in_resist_matrix.astype(int)
    global_outer_monomer_array += now_outer_monomer_array

    global_free_monomer_in_resist_matrix -= now_free_monomer_matrix_0

    plt.figure(dpi=300)
    plt.imshow(global_free_monomer_in_resist_matrix.transpose())
    # plt.imshow(np.log(now_D_matrix.transpose()))
    plt.show()

    plt.figure(dpi=300)
    plt.plot(mm.x_centers_5nm, global_outer_monomer_array)
    plt.show()

    if (np.sum(now_free_monomer_in_resist_matrix) + np.sum(now_outer_monomer_array)) /\
       np.sum(now_free_monomer_matrix_0) != 1:
        print('some monomers are lost!')

    new_zz_vac = global_outer_monomer_array * const.V_mon_nm3 / mm.step_5nm / mm.ly

    plt.figure(dpi=300)
    plt.plot(mm.x_centers_5nm, mm.d_PMMA - new_zz_vac)
    plt.show()

    zz_evolver = mm.d_PMMA - new_zz_vac

    # go to um!!!
    xx_evolver_final = np.concatenate([[mm.x_bins_5nm[0]], mm.x_centers_5nm, [mm.x_bins_5nm[-1]]]) * 1e-3
    zz_evolver_final = np.concatenate([[zz_evolver[0]], zz_evolver, [zz_evolver[-1]]]) * 1e-3
    mobs_evolver_final = np.concatenate([[now_SE_mob_array[0]], now_SE_mob_array, [now_SE_mob_array[-1]]])

    file_full_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/datafile_DEBER_2021_5.fe'
    commands_full_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/commands_5.txt'

    ef.create_datafile_latest(
        yy=xx_evolver_final,
        zz=zz_evolver_final,
        width=mm.ly * 1e-3,
        mobs=mobs_evolver_final,
        path=file_full_path
    )

    ef.run_evolver(file_full_path, commands_full_path)

    SE = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_single_5.txt')
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

    zz_evolver_final_final = mcf.lin_lin_interp(xx_SE_sorted * 1e+3, zz_SE_sorted * 1e+3)(mm.x_centers_5nm)
    zz_vac_final = mm.d_PMMA - zz_evolver_final_final

    plt.figure(dpi=300)
    plt.plot(mm.x_centers_5nm, zz_vac_final)

    if n_step < 46:
        now_zz_vac_raw_next = np.load('notebooks/DEBER_simulation/vary_zz_vac_0p2_5s/zz_vac_' + str(n_step+1) + '.npy')
        plt.plot(xx_vac_raw, now_zz_vac_raw_next)

    plt.grid()
    plt.show()




