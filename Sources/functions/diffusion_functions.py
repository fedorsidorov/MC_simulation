import importlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mapping import mapping_3p3um_80nm as mm
from functions import MC_functions as mcf
import constants as const
import copy

const = importlib.reload(const)
mcf = importlib.reload(mcf)
mm = importlib.reload(mm)


# %%
# D = 3.16e-7 * 1e+7 ** 2  # cm^2 / s -> nm^2 / s
D = 3.16e-6 * 1e+7 ** 2  # cm^2 / s -> nm^2 / s
# D = 3.16e-5 * 1e+7 ** 2  # cm^2 / s -> nm^2 / s

delta_t = 1e-7  # s
sigma = np.sqrt(2 * D * delta_t)  # nm

xx_sample = np.linspace(-3 * sigma, 3 * sigma, 100)
probs = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-xx_sample ** 2 / (2 * sigma ** 2))
probs_norm = probs / np.sum(probs)


# %%
def get_delta_coord_fast():
    return np.random.choice(xx_sample, p=probs_norm)


def get_D(dT, wp=1):  # in cm^2 / s
    # dT = T_C - mobs_120
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
        x0, z0 = mm.x_centers_10nm[ind_x], mm.z_centers_10nm[ind_z]  # nanometers!!!

        n_tens_monomers = int(np.round(monomer_matrix_2d[ind_x, ind_z] / n_hack))

        for n in range(n_tens_monomers):
            x_final, z_final, total_time = track_monomer(x0, z0, xx, zz_vac, d_PMMA, dT, wp, t_step, dtdt)
            monomer_matrix_2d_final += np.histogramdd(sample=np.array((x_final, z_final)).reshape((1, 2)),
                                                      bins=(mm.x_bins_10nm, mm.z_bins_10nm))[0]

        progress_bar.update()

    return monomer_matrix_2d_final


# def get_25nm_array(array):
#     hist_weights = np.histogram(mapping.x_centers_10nm, bins=mapping.x_bins_25nm, weights=array)[0]
#     hist = np.histogram(mapping.x_centers_10nm, bins=mapping.x_bins_25nm)[0]
#     return hist_weights / hist


def move_10nm_to_50nm(array):
    hist_weights = np.histogram(mm.x_centers_10nm, bins=mm.x_bins_50nm, weights=array)[0]
    hist = np.histogram(mm.x_centers_10nm, bins=mm.x_bins_50nm)[0]
    return hist_weights / hist


def move_10nm_to_100nm(array):
    hist_weights = np.histogram(mm.x_centers_10nm, bins=mm.x_bins_100nm, weights=array)[0]
    hist = np.histogram(mm.x_centers_10nm, bins=mm.x_bins_100nm)[0]
    return hist_weights / hist


def get_zz_vac_50nm_monomer_matrix(zz_vac_old_50nm, mon_matrix_2d, n_hack=10):

    n_monomers_out = move_10nm_to_50nm(mon_matrix_2d[:, 0]) * n_hack
    V_monomer_out = n_monomers_out * const.V_mon_cm3
    dh_monomer_out_cm = V_monomer_out / (mm.ly * 1e-7 * mm.step_50nm * 1e-7)
    dh_monomer_out = dh_monomer_out_cm * 1e+7

    zz_vac_new_50nm = zz_vac_old_50nm + dh_monomer_out
    mon_matrix_2d_final = copy.deepcopy(mon_matrix_2d)
    mon_matrix_2d_final[:, 0] = 0

    return zz_vac_new_50nm, mon_matrix_2d_final


def track_monomer_easy(xz_0, xx, zz_vac, d_PMMA):

    def get_z_vac_for_x(x):
        if x > np.max(xx):
            return zz_vac[-1]
        elif x < np.min(xx):
            return zz_vac[0]
        else:
            return mcf.lin_lin_interp(xx, zz_vac)(x)

    now_x = xz_0[0]  # cm
    now_z = xz_0[1]  # cm

    pos_max = 1000

    pos = 1

    now_z_vac = get_z_vac_for_x(now_x)

    while now_z >= now_z_vac and pos < pos_max:
        now_x += get_delta_coord_fast() * 1e-7  # nm -> cm
        delta_z = get_delta_coord_fast() * 1e-7  # nm -> cm

        if now_z + delta_z > d_PMMA:
            now_z -= delta_z
        else:
            now_z += delta_z

        pos += 1

    return now_x


def get_profile_after_diffusion(scission_matrix, zip_length, xx, zz_vac, d_PMMA, mult):

    scission_matrix_sum_y = np.sum(scission_matrix, axis=1)
    n_monomers_groups = zip_length // mult
    x_escape_array = np.zeros(int(np.sum(scission_matrix_sum_y) * n_monomers_groups))
    pos = 0

    sci_pos_arr = np.array(np.where(scission_matrix_sum_y > 0)).transpose()
    progress_bar = tqdm(total=len(sci_pos_arr), position=0)

    for sci_coords in sci_pos_arr:

        x_ind, z_ind = sci_coords
        n_scissions = int(scission_matrix_sum_y[x_ind, z_ind])

        xz_0 = mm.x_centers_10nm[x_ind] * 1e-7, mm.z_centers_10nm[z_ind] * 1e-7

        for _ in range(n_scissions):
            for _ in range(n_monomers_groups):
                x_escape_array[pos] = track_monomer_easy(xz_0, xx * 1e-7, zz_vac * 1e-7, d_PMMA)
                pos += 1

        progress_bar.update()

    mon_h_cm = const.V_mon_cm3 * 1e+7 ** 3 / mm.step_10nm ** 2 * 1e-7

    x_escape_array_corr = np.zeros(np.shape(x_escape_array))

    for i, x_esc in enumerate(x_escape_array):
        while x_esc > mm.x_max:
            x_esc -= mm.lx
        while x_esc < mm.x_min:
            x_esc += mm.lx
        x_escape_array_corr[i] = x_esc

    x_escape_hist = np.histogram(x_escape_array_corr, bins=mm.x_bins_10nm * 1e-7)[0]

    x_escape_hist_corr = x_escape_hist * mult / (mm.ly / mm.step_10nm)

    delta_zz_vac = x_escape_hist_corr * mon_h_cm
    delta_zz_vac_nm = delta_zz_vac * 1e+7
    # delta_zz_vac_doubled = delta_zz_vac + delta_zz_vac[::-1]

    zz_vac_new = zz_vac + delta_zz_vac_nm
    # zz_vac_new = zz_vac + delta_zz_vac_doubled

    return zz_vac_new


# %%
# hx, hz, time = track_monomer(50, 50, xx*1e+7, zz_vac*1e+7, d_PMMA*1e+7, T_C, 1, 10)
#
# inds = np.where(hz != 0)[0]
#
# plt.figure(dpi=300)
# plt.plot(hx[inds], hz[inds])
# plt.show()
