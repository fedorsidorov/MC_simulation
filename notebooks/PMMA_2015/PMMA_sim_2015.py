import importlib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

import grid
grid = importlib.reload(grid)

# %% load arrays_PMMA
# elastic_kind = 'easy'
elastic_kind = 'muffin'
PMMA_el_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/PMMA/PMMA_'
                       + elastic_kind + '_u.npy') * 1e-7

PMMA_el_IMFP[:228] = PMMA_el_IMFP[228]

PMMA_el_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/PMMA/'
                                  'MMA_' + elastic_kind + '_diff_cs_plane_norm_cumulated.npy')

# %% Mermin
PMMA_val_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/PMMA/arrays_PMMA/'
                        'PMMA_val_IMFP_nm.npy')
PMMA_val_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/PMMA/'
                                   'arrays_PMMA/PMMA_val_DIMFP_cumulated_corr.npy')


# %% phonon
def get_PMMA_U_phonon(E, T=300):
    a0 = 5.292e-11 * 1e+2  # cm
    kT = 8.617e-5 * T  # eV
    hw = 0.1
    e_0 = 3.9
    e_inf = 2.2

    nT = (np.exp(hw / kT) - 1) ** (-1)
    KT = 1 / a0 * ((nT + 1) / 2) * ((e_0 - e_inf) / (e_0 * e_inf))

    u_ph = KT * hw / E * np.log((1 + np.sqrt(1 - hw / E)) / (1 - np.sqrt(1 - hw / E)))  # cm^-1

    return u_ph * 1e-7  # nm^-1


PMMA_ph_IMFP = get_PMMA_U_phonon(grid.EE)

# %% 2011
# Wf_PMMA = 1
# C_pol = 1.5  # nm^-1
# gamma_pol = 0.14  # eV^-1

# 2015
# Wf_PMMA = 4.68
Wf_PMMA = 1
C_pol = 0.1  # nm^-1
gamma_pol = 0.15  # eV^-1

hw_phonon = 0.1

PMMA_E_cut = Wf_PMMA

PMMA_pol_IMFP = C_pol * np.exp(-gamma_pol * grid.EE)

PMMA_IMFP =\
    np.vstack((PMMA_el_IMFP, PMMA_val_IMFP, PMMA_ph_IMFP, PMMA_pol_IMFP)).transpose()

# norm IMFP array
PMMA_IMFP_norm = np.zeros(np.shape(PMMA_IMFP))
for i in range(len(PMMA_IMFP)):
    if np.sum(PMMA_IMFP[i, :]) != 0:
        PMMA_IMFP_norm[i, :] = PMMA_IMFP[i, :] / np.sum(PMMA_IMFP[i, :])

PMMA_total_IMFP = np.sum(PMMA_IMFP, axis=1)
process_indexes = list(range(len(PMMA_IMFP[0, :])))

# PMMA_ee_DIMFP_cumulated = np.array([PMMA_val_DIMFP_cumulated])

# %% plot cross sections
plt.figure(dpi=300)

for j in range(len(PMMA_IMFP[0])):
    plt.loglog(grid.EE, PMMA_IMFP[:, j], '--')

plt.loglog(grid.EE, PMMA_total_IMFP)

plt.xlabel('E, eV')
plt.ylabel(r'$\mu$, nm$^{-1}$')
plt.ylim(1e-5, 1e+2)
plt.grid()
plt.show()


# %% functions
def plot_e_DATA(e_DATA_arr):
    fig, ax = plt.subplots(dpi=300)

    for e_id in range(int(np.max(e_DATA_arr[:, 0]) + 1)):
        inds = np.where(e_DATA_arr[:, 0] == e_id)[0]

        if len(inds) == 0:
            continue

        ax.plot(e_DATA_arr[inds, 3], e_DATA_arr[inds, 5], '-', linewidth='1')

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.grid()
    plt.show()


def get_scattered_flight_ort(flight_ort, phi, theta):
    u, v, w = flight_ort

    if w == 1:
        u_new = np.sin(theta) * np.cos(phi)
        v_new = np.sin(theta) * np.sin(phi)
        w_new = np.cos(theta)

    elif w == -1:
        u_new = -np.sin(theta) * np.cos(phi)
        v_new = -np.sin(theta) * np.sin(phi)
        w_new = -np.cos(theta)

    else:
        u_new = u * np.cos(theta) + np.sin(theta) / np.sqrt(1 - w ** 2) * (u * w * np.cos(phi) - v * np.sin(phi))
        v_new = v * np.cos(theta) + np.sin(theta) / np.sqrt(1 - w ** 2) * (v * w * np.cos(phi) + u * np.sin(phi))
        w_new = w * np.cos(theta) - np.sqrt(1 - w ** 2) * np.sin(theta) * np.cos(phi)

    new_flight_ort = np.array((u_new, v_new, w_new))

    if np.linalg.norm(new_flight_ort) != 1:
        new_flight_ort = new_flight_ort / np.linalg.norm(new_flight_ort)

    return new_flight_ort


def get_T_PMMA(E_cos2_theta):

    if E_cos2_theta >= Wf_PMMA:

        if 1 - Wf_PMMA / E_cos2_theta < 0:
            print('sqrt T error')

        T_PMMA = 4 * np.sqrt(1 - Wf_PMMA / E_cos2_theta) / (1 + np.sqrt(1 - Wf_PMMA / E_cos2_theta)) ** 2
        return T_PMMA
    else:
        return 0.


def get_phonon_scat_hw_phi_theta(E):
    phi = 2 * np.pi * np.random.random()
    # hw_phonon = 0.1
    Ep = E - hw_phonon

    if E * Ep < 0:
        print('sqrt ph error', E, Ep)

    B = (E + Ep + 2 * np.sqrt(E * Ep)) / (E + Ep - 2 * np.sqrt(E * Ep))
    u = np.random.random()
    cos_theta = (E + Ep) / (2 * np.sqrt(E * Ep)) * (1 - B ** u) + B ** u
    theta = np.arccos(cos_theta)
    return hw_phonon, phi, theta


def track_electron(e_id, par_id, E_0, coords_0, flight_ort_0):
    E = E_0
    coords = coords_0
    flight_ort = flight_ort_0

    e_DATA_deque = deque()
    e_DATA_initial_line = [e_id, par_id, -1, *coords_0, 0, 0, E]
    e_DATA_deque.append(e_DATA_initial_line)

    e_2nd_deque = deque()
    next_e_2nd_id = 0

    #  main cycle
    while E > PMMA_E_cut:

        E_dep = 0
        hw = 0
        E_2nd = 0

        E_ind = np.argmin(np.abs(grid.EE - E))

        u1 = np.random.random()
        free_path = -1 / PMMA_total_IMFP[E_ind] * np.log(u1)
        delta_r = flight_ort * free_path
        coords = coords + delta_r

        if coords[-1] < 0:  # e emerges from the specimenn

            cos_theta = -flight_ort[-1]

            if np.random.random() < get_T_PMMA(E * cos_theta ** 2):  # electron emerges
                if E * cos_theta ** 2 < Wf_PMMA:
                    print('Wf problems', E, cos_theta)

                # TODO check energy while leaving PMMA
                # E -= Wf_PMMA
                break

            else:  # electron scatters TODO different scattering models
                coords -= delta_r
                # factor = coords[2] / delta_r[2]
                factor = 0
                coords += delta_r * factor  # reach surface
                delta_r[2] *= -1
                coords += delta_r * (1 - factor)
                flight_ort[2] *= -1

        proc_ind = np.random.choice(process_indexes, p=PMMA_IMFP_norm[E_ind, :])

        if proc_ind == 0:  # elastic scattering
            phi = 2 * np.pi * np.random.random()
            u2 = np.random.random()
            theta_ind = np.argmin(np.abs(PMMA_el_DIMFP_cumulated[E_ind, :] - u2))
            theta = grid.THETA_rad[theta_ind]
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

        elif proc_ind == 1:  # e-e scattering
            u2 = np.random.random()
            hw_ind = np.argmin(np.abs(PMMA_val_DIMFP_cumulated[E_ind, :] - u2))
            hw = grid.EE[hw_ind]

            delta_E = hw

            if delta_E < 0:
                print('delta E < 0 !!!')

            phi = 2 * np.pi * np.random.random()
            phi_2nd = phi - np.pi

            sin2 = delta_E / E
            # TODO check uniform 2ndary angle distribution
            sin2_2nd = 1 - delta_E / E

            if sin2 > 1:
                print('sin2 > 1 !!!', sin2)

            if sin2_2nd < 0:
                print('sin2_2nd < 0 !!!')

            theta = np.arcsin(np.sqrt(sin2))
            theta_2nd = np.arcsin(np.sqrt(sin2_2nd))
            # theta_2nd = np.pi * np.random.random()

            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)
            flight_ort_2nd = get_scattered_flight_ort(flight_ort, phi_2nd, theta_2nd)

            E_2nd = delta_E

            e_2nd_list = [next_e_2nd_id, e_id, E_2nd, *coords, *flight_ort_2nd]
            e_2nd_deque.append(e_2nd_list)
            next_e_2nd_id += 1

        elif proc_ind == 4:  # phonon
            hw, phi, theta = get_phonon_scat_hw_phi_theta(E)
            E_dep = hw
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

        else:  # polaron
            break

        E -= hw
        flight_ort = new_flight_ort

        e_DATA_line = [e_id, par_id, proc_ind, *coords, E_dep, E_2nd, E]
        e_DATA_deque.append(e_DATA_line)

    e_DATA_final_line = [e_id, par_id, -2, *coords, E, 0, 0]
    e_DATA_deque.append(e_DATA_final_line)

    return e_DATA_deque, e_2nd_deque


def track_all_electrons(n_electrons, E0):

    e_deque = deque()
    total_e_DATA = deque()

    next_e_id = 0

    for _ in range(n_electrons):
        e_deque.append([
            next_e_id,
            -1,
            E0,
            0, 0, 0,
            0, 0, 1]
        )
        next_e_id += 1

    progress_bar = tqdm(total=n_electrons, position=0)

    while e_deque:

        now_e_id, now_par_id, now_E0, now_x0, now_y0, now_z0, now_ort_x, now_ort_y, now_ort_z = e_deque.popleft()

        if now_par_id == -1:
            progress_bar.update()

        now_e_DATA, now_e_2nd_deque = track_electron(
            now_e_id, now_par_id, now_E0,
            np.array([now_x0, now_y0, now_z0]),
            np.array([now_ort_x, now_ort_y, now_ort_z])
        )

        for e_2nd_line in now_e_2nd_deque:
            e_2nd_line[0] += next_e_id

        next_e_id += len(now_e_2nd_deque)

        total_e_DATA = total_e_DATA + now_e_DATA
        e_deque = now_e_2nd_deque + e_deque

    return np.around(np.vstack(total_e_DATA), decimals=4)


# %%
n_files = 100
n_primaries_in_file = 100

# E_beam_arr = [250]
E_beam_arr = [200, 250, 300, 350, 400, 500]

for n in range(1, n_files):
    print('File #' + str(n))

    for E_beam in E_beam_arr:
        print(E_beam)
        e_DATA = track_all_electrons(n_primaries_in_file, E_beam)
        e_DATA_outer = e_DATA[np.where(e_DATA[:, 5] < 0)]
        np.save('data/2ndaries/2ndaries_outer_Wf_1_2015/' + str(E_beam) + '/e_DATA_' + str(n) + '.npy',
                e_DATA_outer)

# %%
# d_DATA = deepcopy(e_DATA)
# d_DATA = np.delete(d_DATA, np.where(d_DATA[:, 8] < 2)[0], axis=0)

# %%
plot_e_DATA(e_DATA)
