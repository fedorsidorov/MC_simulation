import importlib
import matplotlib.pyplot as plt
from functions import MC_functions as mcf
import numpy as np
from tqdm import tqdm
from mapping import mapping_3p3um_80nm as mapping
import constants
import copy

constants = importlib.reload(constants)
mapping = importlib.reload(mapping)
mcf = importlib.reload(mcf)


# %%
def get_D(dT, wp=1):  # in cm^2 / s
    # dT = T_C - 120
    C1, C2, C3, C4 = 26.0, 37.0, 0.0797, 0
    log_D = wp * dT * C4 + (C1 - wp * C2) + dT * C3
    return 10**log_D


def get_dt_dx_dz(T_C, wp=1, dt=0.1):  # nanometers!!!
    D = get_D(T_C, wp)  # in cm^2 / s
    # sigma = 5e-7  # bin size in cm
    # dt = sigma**2 / 4 / D
    # dt = 0.1
    sigma = np.sqrt(2 * D * dt)
    # print(sigma)
    xx = np.linspace(-3 * sigma, 3 * sigma, 1000)
    probs = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-xx ** 2 / (2 * sigma ** 2))
    probs_norm = probs / np.sum(probs)
    return dt, np.random.choice(xx, p=probs_norm) * 1e+7, np.random.choice(xx, p=probs_norm) * 1e+7  # nanometers!!!


def track_monomer(x0, z0, xx, zz_vac, d_PMMA, dT, wp, t_step, dtdt):  # nanometers!!!

    def get_z_vac_for_x(x):
        if x > np.max(xx):
            return zz_vac[-1]
        elif x < np.min(xx):
            return zz_vac[0]
        else:
            return mcf.lin_lin_interp(xx, zz_vac)(x)

    now_x = x0
    now_z = z0

    # history = np.zeros((100000, 2))
    # cnt = 0
    # history[cnt, :] = now_x, now_z
    # cnt += 1

    total_time = 0

    while total_time < t_step:

        dt, dx, dz = get_dt_dx_dz(dT, wp, dt=dtdt)
        total_time += dt
        now_x += dx

        if now_z + dz > d_PMMA:
            now_z -= dz
        else:
            now_z += dz

        # history[cnt, :] = now_x, now_z
        # cnt += 1

        if now_z < zz_vac.min():
            if now_z < get_z_vac_for_x(now_x):
                return now_x, 0, total_time
                # return now_x, 0, total_time, history[np.where(history[:, 0] != 0)]

    return now_x, now_z, total_time
    # return now_x, now_z, total_time, history[np.where(history[:, 0] != 0)]


def exp_gauss(xx, A, B, s):
    return np.exp(A - B*np.exp(-xx**2 / s**2))


def track_all_monomers(monomer_matrix_2d, xx, zz_vac, d_PMMA, dT, wp, t_step, dtdt, n_hack=1):

    monomer_matrix_2d_final = np.zeros((np.shape(monomer_matrix_2d)))
    non_zero_inds = np.array(np.where(monomer_matrix_2d != 0)).transpose()

    progress_bar = tqdm(total=len(non_zero_inds), position=0)

    for line in non_zero_inds:

        ind_x, ind_z = line
        x0, z0 = mapping.x_centers_10nm[ind_x], mapping.z_centers_10nm[ind_z]  # nanometers!!!

        n_tens_monomers = int(np.round(monomer_matrix_2d[ind_x, ind_z] / n_hack))

        for n in range(n_tens_monomers):
            x_final, z_final, total_time = track_monomer(x0, z0, xx, zz_vac, d_PMMA, dT, wp, t_step, dtdt)
            monomer_matrix_2d_final += np.histogramdd(sample=np.array((x_final, z_final)).reshape((1, 2)),
                                                      bins=(mapping.x_bins_10nm, mapping.z_bins_10nm))[0]

        progress_bar.update()

    return monomer_matrix_2d_final


# def track_all_monomers(monomer_matrix_2d, xx, zz_vac, d_PMMA, dT, wp, t_step, dtdt):
#
#     monomer_matrix_2d_final = np.zeros((np.shape(monomer_matrix_2d)))
#     non_zero_inds = np.array(np.where(monomer_matrix_2d != 0)).transpose()
#
#     progress_bar = tqdm(total=len(non_zero_inds), position=0)
#
#     for line in non_zero_inds:
#
#         ind_x, ind_z = line
#         x0, z0 = mapping.x_centers_10nm[ind_x], mapping.z_centers_10nm[ind_z]  # nanometers!!!
#
#         n_tens_monomers = int(np.round(monomer_matrix_2d[ind_x, ind_z] / 10))  # in tens!!!
#
#         for n in range(n_tens_monomers):
#             x_final, z_final, total_time = track_monomer(x0, z0, xx, zz_vac, d_PMMA, dT, wp, t_step, dtdt)
#             monomer_matrix_2d_final += np.histogramdd(sample=np.array((x_final, z_final)).reshape((1, 2)),
#                                                       bins=(mapping.x_bins_10nm, mapping.z_bins_10nm))[0]
#
#         progress_bar.update()
#
#     return monomer_matrix_2d_final


def get_25nm_array(array):
    hist_weights = np.histogram(mapping.x_centers_10nm, bins=mapping.x_bins_25nm, weights=array)[0]
    hist = np.histogram(mapping.x_centers_10nm, bins=mapping.x_bins_25nm)[0]
    return hist_weights / hist


def get_50nm_array(array):
    hist_weights = np.histogram(mapping.x_centers_10nm, bins=mapping.x_bins_50nm, weights=array)[0]
    hist = np.histogram(mapping.x_centers_10nm, bins=mapping.x_bins_50nm)[0]
    return hist_weights / hist


def get_100nm_array(array):
    hist_weights = np.histogram(mapping.x_centers_10nm, bins=mapping.x_bins_100nm, weights=array)[0]
    hist = np.histogram(mapping.x_centers_10nm, bins=mapping.x_bins_100nm)[0]
    return hist_weights / hist


def get_zz_vac_50nm_monomer_matrix(zz_vac_old_50nm, mon_matrix_2d, n_hack=10):

    n_monomers_out = get_50nm_array(mon_matrix_2d[:, 0]) * n_hack
    V_monomer_out = n_monomers_out * constants.V_mon
    dh_monomer_out_cm = V_monomer_out / (mapping.l_y * 1e-7 * mapping.step_50nm * 1e-7)
    dh_monomer_out = dh_monomer_out_cm * 1e+7

    zz_vac_new_50nm = zz_vac_old_50nm + dh_monomer_out
    mon_matrix_2d_final = copy.deepcopy(mon_matrix_2d)
    mon_matrix_2d_final[:, 0] = 0

    return zz_vac_new_50nm, mon_matrix_2d_final


# %%
# hx, hz, time = track_monomer(50, 50, xx*1e+7, zz_vac*1e+7, d_PMMA*1e+7, T_C, 1, 10)
#
# inds = np.where(hz != 0)[0]
#
# plt.figure(dpi=300)
# plt.plot(hx[inds], hz[inds])
# plt.show()
