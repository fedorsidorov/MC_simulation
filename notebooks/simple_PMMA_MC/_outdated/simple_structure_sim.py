import importlib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

import grid
from notebooks.simple_PMMA_MC._outdated import simple_arrays as arr

grid = importlib.reload(grid)
arr = importlib.reload(arr)

# %%
Si_E_pl = 16.7
Si_E_bind = [0, 20.1, 102, 151.1, 1828.9]

Wf_PMMA = 4.68
hw_phonon = 0.1
# PMMA_E_cut = Wf_PMMA
PMMA_E_cut = 15
PMMA_E_bind = [0, 284.2, 543.1]

structure_E_bind = [PMMA_E_bind, Si_E_bind]
structure_E_cut = [PMMA_E_cut, Si_E_pl]

# %% plot PMMA
# plt.figure(dpi=300)
#
# for j in range(len(arr.PMMA_IMFP[0])):
#     plt.loglog(grid.EE, arr.PMMA_IMFP[:, j])
#
# plt.loglog(grid.EE, arr.PMMA_total_IMFP, '.')
# plt.loglog(grid.EE, arr.PMMA_val_IMFP + arr.PMMA_C_IMFP + arr.PMMA_O_IMFP, '.')
#
# plt.xlabel('E, eV')
# plt.ylabel(r'$\mu$, nm$^{-1}$')
# plt.ylim(1e-5, 1e+2)
# plt.grid()
# plt.show()

# %% plot simple_Si_MC
# plt.figure(dpi=300)
#
# for j in range(len(arr.Si_IMFP[0])):
#     plt.loglog(grid.EE, arr.Si_IMFP[:, j])
#
# plt.loglog(grid.EE, arr.Si_total_IMFP, '.')
#
# plt.xlabel('E, eV')
# plt.ylabel(r'$\mu$, nm$^{-1}$')
# plt.ylim(1e-5, 1e+2)
# plt.grid()
# plt.show()


# %% functions
def plot_e_DATA(e_DATA_arr, d_PMMA, E_cut=5):

    e_DATA_arr = e_DATA_arr[np.where(e_DATA_arr[:, -1] > E_cut)]
    fig, ax = plt.subplots(dpi=300)

    for e_id in range(int(np.max(e_DATA_arr[:, 0]) + 1)):
        inds = np.where(e_DATA_arr[:, 0] == e_id)[0]
        if len(inds) == 0:
            continue
        ax.plot(e_DATA_arr[inds, 4], e_DATA_arr[inds, 6], '-', linewidth='1')

    plt.plot(np.linspace(-500, 500, 100), np.ones(100) * d_PMMA, 'k-')

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

        T_PMMA = 4 * np.sqrt(1 - Wf_PMMA / E_cos2_theta) / \
            (1 + np.sqrt(1 - Wf_PMMA / E_cos2_theta)) ** 2
        return T_PMMA
    else:
        return 0.


def get_phonon_scat_hw_phi_theta(E):
    phi = 2 * np.pi * np.random.random()
    Ep = E - hw_phonon

    if E * Ep < 0:
        print('sqrt ph error', E, Ep)

    B = (E + Ep + 2 * np.sqrt(E * Ep)) / (E + Ep - 2 * np.sqrt(E * Ep))
    u = np.random.random()
    cos_theta = (E + Ep) / (2 * np.sqrt(E * Ep)) * (1 - B ** u) + B ** u
    theta = np.arccos(cos_theta)
    return hw_phonon, phi, theta


def track_electron(e_id, par_id, E_0, coords_0, flight_ort_0, d_PMMA):
    E = E_0
    coords = coords_0
    flight_ort = flight_ort_0

    e_DATA_deque = deque()
    e_DATA_initial_line = [e_id, par_id, 10, -1, *coords_0, 0, 0, E]
    e_DATA_deque.append(e_DATA_initial_line)

    e_2nd_deque = deque()
    next_e_2nd_id = 0

    #  main cycle
    while True:

        E_bind = 0
        hw = 0
        E_2nd = 0

        E_ind = np.argmin(np.abs(grid.EE - E))

        if coords[2] <= d_PMMA:
            layer_ind = 0
        else:
            layer_ind = 1

        if E < structure_E_cut[layer_ind]:
            break

        u1 = np.random.random()
        free_path = -1 / arr.structure_total_IMFP[layer_ind][E_ind] * np.log(1 - u1)
        delta_r = flight_ort * free_path

        z1 = coords[2]
        coords = coords + delta_r
        z2 = coords[2]

        if (z1 < d_PMMA) ^ (z2 < d_PMMA):  # interface crossing

            coords = coords - delta_r

            p1 = arr.structure_total_IMFP[layer_ind][E_ind]
            p2 = arr.structure_total_IMFP[layer_ind - 1][E_ind]
            scale_factor = np.abs(d_PMMA - z1) / np.abs(z2 - z1)
            d = free_path * scale_factor

            if u1 < (1 - np.exp(-p1 * d)) or E < Si_E_pl:  # TODO why E < Si_E_pl ?
                free_path_corr = 1 / p1 * (-np.log(1 - u1))
            else:
                free_path_corr = d + (1 / p2) * (-np.log(1 - u1) - p1 * d)

            delta_r_corr = flight_ort * free_path_corr
            coords = coords + delta_r_corr

            if coords[2] > d_PMMA and E < Si_E_pl:  # low-E e comes to simple_Si_MC from PMMA
                break

        if coords[-1] < 0:  # e emerges from the specimenn

            cos_theta = -flight_ort[-1]

            if np.random.random() < get_T_PMMA(E * cos_theta ** 2):  # electron emerges
                if E * cos_theta ** 2 < Wf_PMMA:
                    print('Wf problems', E, cos_theta)

                # e_DATA_line = [e_id, par_id, -2, *coords, 0, 0, E - Wf_PMMA]
                # e_DATA_deque.append(e_DATA_line)
                break

            else:  # electron scatters TODO different scattering models
                coords = coords - delta_r
                delta_r[2] *= -1
                coords += delta_r
                new_flight_ort = flight_ort
                new_flight_ort[2] *= -1

        proc_ind = np.random.choice(arr.structure_process_indexes[layer_ind],
                                    p=arr.structure_IMFP_norm[layer_ind][E_ind, :])

        if proc_ind == 0:  # elastic scattering
            phi = 2 * np.pi * np.random.random()
            u2 = np.random.random()
            theta_ind = np.argmin(np.abs(arr.structure_el_DIMFP_cumulated[layer_ind][E_ind, :] - u2))
            theta = grid.THETA_rad[theta_ind]
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

        elif layer_ind == 1 and proc_ind == 1:  # simple_Si_MC plasmon
            hw = Si_E_pl
            E_bind = Si_E_pl

            phi = 2 * np.pi * np.random.random()
            sin2 = Si_E_pl / E

            if sin2 > 1:
                print('plasmon sin error', layer_ind)

            theta = np.arcsin(np.sqrt(sin2))
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

            E_2nd = 0

        elif layer_ind == 0 and proc_ind in [1, 2, 3] or layer_ind == 1 and proc_ind > 1:  # e-e scattering
            ss_ind = proc_ind - 1
            u2 = np.random.random()
            hw_ind = np.argmin(np.abs(arr.structure_ee_DIMFP_cumulated[layer_ind][ss_ind, E_ind, :] - u2))
            hw = grid.EE[hw_ind]

            E_bind = structure_E_bind[layer_ind][ss_ind]
            delta_E = hw - E_bind

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

            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)
            flight_ort_2nd = get_scattered_flight_ort(flight_ort, phi_2nd, theta_2nd)

            E_2nd = delta_E

            e_2nd_list = [next_e_2nd_id, e_id, E_2nd, *coords, *flight_ort_2nd]
            e_2nd_deque.append(e_2nd_list)
            next_e_2nd_id += 1

        elif layer_ind == 0 and proc_ind == 4:  # phonon
            hw, phi, theta = get_phonon_scat_hw_phi_theta(E)
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

        elif layer_ind == 0 and proc_ind == 5:  # polaron
            break

        else:
            new_flight_ort = flight_ort
            print('proc_ind error!')

        E -= hw
        flight_ort = new_flight_ort

        e_DATA_line = [e_id, par_id, layer_ind, proc_ind, *coords, E_bind, E_2nd, E]
        e_DATA_deque.append(e_DATA_line)

    e_DATA_final_line = [e_id, par_id, -2, 10, *coords, E, 0, 0]
    e_DATA_deque.append(e_DATA_final_line)

    return e_DATA_deque, e_2nd_deque


def track_all_electrons(n_electrons, E0, d_PMMA):

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
            np.array([now_ort_x, now_ort_y, now_ort_z]),
            d_PMMA
        )

        for e_2nd_line in now_e_2nd_deque:
            e_2nd_line[0] += next_e_id

        next_e_id += len(now_e_2nd_deque)

        total_e_DATA = total_e_DATA + now_e_DATA
        e_deque = now_e_2nd_deque + e_deque

    return np.around(np.vstack(total_e_DATA), decimals=4)


# %%
n_files = 100
n_electrons_in_file = 100

E0 = 20e+3
d_PMMA = 900

for n in range(n_files):

    print('File #' + str(n))
    e_DATA = track_all_electrons(n_electrons_in_file, E0, d_PMMA=d_PMMA)
    np.save('/Volumes/Transcend/NEW_e_DATA_900nm/e_DATA_' + str(n) + '.npy', e_DATA)

# %%
# e_DATA = track_all_electrons(n_electrons=10, E0=E0, d_PMMA=d_PMMA)

# %%
plot_e_DATA(e_DATA, d_PMMA=d_PMMA, E_cut=15)

# %%
# np.save('/Volumes/Transcend/NEW_e_DATA_900nm/e_DATA_0.npy', e_DATA)

