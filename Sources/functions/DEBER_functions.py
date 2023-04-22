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
tau_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Boyd_Schulz_Zimm/arrays/tau.npy')
Mn_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Boyd_Schulz_Zimm/arrays/Mn_125.npy')\
         * const.MMA_weight
Mw_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Boyd_Schulz_Zimm/arrays/Mw_125.npy')\
         * const.MMA_weight

y_0 = 3989
Mn_0 = 271374


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

            n_chains = mm.y_slice_V_nm3 / const.V_mon_nm3 / (Mn / const.MMA_weight)
            true_Mn = (n_chains * Mn + free_monomer_in_resist_matrix[ii, kk] * const.MMA_weight) / \
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


def get_delta_tau_matix(resist_monomer_matrix, scission_matrix, step_time):
    ks_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))
    inds_mon = np.where(resist_monomer_matrix > 0)
    ks_matrix[inds_mon] = scission_matrix[inds_mon] / resist_monomer_matrix[inds_mon] / step_time
    delta_tau_matrix = y_0 * ks_matrix * step_time
    return delta_tau_matrix


def get_wp_D_matrix(global_free_monomer_in_resist_matrix, resist_monomer_matrix, temp_C):
    wp_matrix = np.zeros(np.shape(resist_monomer_matrix))
    D_matrix = np.zeros(np.shape(resist_monomer_matrix))

    for i in range(len(mm.x_centers_5nm)):
        for k in range(len(mm.z_centers_5nm)):

            if resist_monomer_matrix[i, k] == 0:
                wp = 0
            else:
                n_free_monomers = global_free_monomer_in_resist_matrix[i, k]
                wp = 1 - n_free_monomers / resist_monomer_matrix[i, k]

            if wp <= 0:
                wp = 0

            D = df.get_D(temp_C, wp)

            wp_matrix[i, k] = wp
            D_matrix[i, k] = D

    return wp_matrix, D_matrix


def get_true_D_matrix(D_matrix, Mn_matrix, k_diff, Mn_diff):
    true_D_matrix = np.zeros(np.shape(D_matrix))

    for i in range(len(mm.x_centers_5nm)):
        for k in range(len(mm.z_centers_5nm)):

            if Mn_matrix[i, k] == 0:
                continue

            true_D_matrix[i, k] = D_matrix[i, k] * np.power(10, k_diff * (1/Mn_matrix[i, k] - 1/Mn_diff))

    return true_D_matrix


def get_free_mon_matrix_mon_out_array_after_diffusion(global_free_monomer_in_resist_matrix,
                                                      D_matrix, xx_vac, zz_vac, d_PMMA, step_time):
    now_xx_vac_for_sim = np.concatenate(([-1e+6], xx_vac, [1e+6]))
    now_zz_vac_for_sim = np.concatenate(([zz_vac[0]], zz_vac, [zz_vac[-1]]))

    free_monomer_in_resist_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))
    out_monomer_array = np.zeros(len(mm.x_centers_5nm))

    print('Simulate diffusion')
    progress_bar = tqdm(total=len(mm.x_centers_5nm), position=0)

    for i in range(len(mm.x_centers_5nm)):
        for k in range(len(mm.z_centers_5nm)):

            n_free_monomers = int(global_free_monomer_in_resist_matrix[i, k])

            if n_free_monomers == 0:
                continue

            xx0_mon_arr = np.ones(n_free_monomers) * mm.x_centers_5nm[i]
            zz0_mon_arr = np.ones(n_free_monomers) * mm.z_centers_5nm[k]

            if D_matrix[i, k] == 0:
                xx_mon = xx0_mon_arr
                zz_mon = zz0_mon_arr

            else:
                xx_mon = df.get_final_x_arr(x0_arr=xx0_mon_arr, D=D_matrix[i, k], delta_t=step_time)
                zz_mon = df.get_final_z_arr(z0_arr_raw=zz0_mon_arr, d_PMMA=d_PMMA, D=D_matrix[i, k], delta_t=step_time)

            af.snake_coord_1d(array=xx_mon, coord_min=mm.x_bins_5nm[0], coord_max=mm.x_bins_5nm[-1])

            zz_mon_vac = mcf.lin_lin_interp(now_xx_vac_for_sim, now_zz_vac_for_sim)(xx_mon)
            weights = zz_mon > zz_mon_vac
            xx_zz_mon = np.vstack([xx_mon, zz_mon]).transpose()

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


def get_scission_matrix(n_electrons_in_file, E0, scission_weight, xx_vac, zz_vac):
    now_xx_vac_for_sim = np.concatenate(([-1e+6], xx_vac, [1e+6]))
    now_zz_vac_for_sim = np.concatenate(([zz_vac[0]], zz_vac, [zz_vac[-1]]))

    print('Get scission matrix')

    now_e_DATA = eb_MC.track_all_electrons(
        n_electrons=int(n_electrons_in_file / 2),
        E0=E0,
        d_PMMA=mm.d_PMMA,
        z_cut=np.inf,
        Pn=True,
        xx_vac=now_xx_vac_for_sim,
        zz_vac=now_zz_vac_for_sim,
        r_beam_x=mm.r_beam_x,
        r_beam_y=mm.r_beam_y
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

    now_Pv_e_DATA = emf.delete_snaked_vacuum_events(now_Pv_e_DATA, now_xx_vac_for_sim, now_zz_vac_for_sim)

    val_matrix = np.histogramdd(
        sample=now_Pv_e_DATA[:, [ind.e_DATA_x_ind, ind.e_DATA_z_ind]],
        bins=[mm.x_bins_5nm, mm.z_bins_5nm]
    )[0]

    val_matrix += val_matrix[::-1, :]

    scission_matrix = np.zeros(np.shape(val_matrix), dtype=int)

    for x_ind in range(len(val_matrix)):
        for z_ind in range(len(val_matrix[0])):
            n_val = int(val_matrix[x_ind, z_ind])

            scissions = np.where(np.random.random(n_val) < scission_weight)[0]
            scission_matrix[x_ind, z_ind] = len(scissions)

    return scission_matrix


def get_zz_after_evolver(vlist_path):
    SE = np.loadtxt(vlist_path)
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

    zz_after_evolver = mcf.lin_lin_interp(xx_SE_sorted * 1e+3, zz_SE_sorted * 1e+3)(mm.x_centers_5nm)

    return zz_after_evolver


def get_zz_after_evolver_20(vlist_path):
    SE = np.loadtxt(vlist_path)
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

    zz_after_evolver = mcf.lin_lin_interp(xx_SE_sorted * 1e+3, zz_SE_sorted * 1e+3)(mm.x_centers_20nm)

    return zz_after_evolver


def get_zz_after_evolver_50(vlist_path):
    SE = np.loadtxt(vlist_path)
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

    zz_after_evolver = mcf.lin_lin_interp(xx_SE_sorted * 1e+3, zz_SE_sorted * 1e+3)(mm.x_centers_50nm)

    return zz_after_evolver


