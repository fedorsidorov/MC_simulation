import importlib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

import grid
grid = importlib.reload(grid)

# %% load arrays
Si_E_pl = 16.7
Si_E_cut_ind = 278
Si_E_bind = [16.65, 6.52, 13.63, 107.98, 151.55, 1828.5]

Si_el_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/' +
                     'simple_MC/arrays/MuElec_elastic_arrays/u.npy') * 1e-7
Si_el_DIMFP_cumulated = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_MC/arrays/' +
            'MuElec_elastic_arrays/u_diff_cumulated.npy')

Si_ee_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/' +
                     'simple_MC/arrays/MuElec_inelastic_arrays/u_ee_6.npy') * 1e-7
Si_ee_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/' +
                                'simple_MC/arrays/MuElec_inelastic_arrays/u_diff_cumulated_6.npy')

Si_IMFP = np.vstack((Si_el_IMFP, Si_ee_IMFP.transpose())).transpose()

# norm IMFP array
Si_IMFP_norm = np.zeros(np.shape(Si_IMFP))
for i in range(len(Si_IMFP)):
    if np.sum(Si_IMFP[i, :]) != 0:
        Si_IMFP_norm[i, :] = Si_IMFP[i, :] / np.sum(Si_IMFP[i, :])

Si_total_IMFP = np.sum(Si_IMFP, axis=1)
process_indexes = list(range(len(Si_IMFP[0, :])))


# %%
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


def track_electron(e_id, par_id, E_0, coords_0, flight_ort_0):

    E = E_0
    coords = coords_0
    flight_ort = flight_ort_0

    e_DATA_deque = deque()

    e_DATA_line = [e_id, par_id, -1, *coords_0, 0, 0, E]
    e_DATA_deque.append(e_DATA_line)

    e_2nd_deque = deque()
    next_e_2nd_id = 0

    #  main cycle
    while E > Si_E_pl:
        E_ind = np.argmin(np.abs(grid.EE - E))

        u1 = np.random.random()
        free_path = -1 / Si_total_IMFP[E_ind] * np.log(u1)
        delta_r = flight_ort * free_path
        coords = coords + delta_r

        if coords[-1] < 0:  # e emerges from the specimenn
            e_DATA_line = [e_id, par_id, -2, *coords, 0, 0, 0]
            e_DATA_deque.append(e_DATA_line)
            break

        proc_ind = np.random.choice(process_indexes, p=Si_IMFP_norm[E_ind, :])

        if proc_ind == 0:  # elastic scattering
            phi = 2 * np.pi * np.random.random()
            u2 = np.random.random()
            theta_ind = np.argmin(np.abs(Si_el_DIMFP_cumulated[E_ind, :] - u2))
            theta = grid.THETA_rad[theta_ind]
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

            # flight_ort = new_flight_ort
            hw = 0
            E_bind = 0
            E_2nd = 0

        elif proc_ind == 1:  # plasmon
            hw = Si_E_pl
            E_bind = Si_E_pl

            phi = 2 * np.pi * np.random.random()
            sin2 = Si_E_pl / E
            theta = np.arcsin(np.sqrt(sin2))
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

            E_2nd = 0

        else:  # e-e scattering
            ss_ind = proc_ind - 1
            u2 = np.random.random()
            hw_ind = np.argmin(np.abs(Si_ee_DIMFP_cumulated[ss_ind, E_ind, :] - u2))
            hw = grid.EE[hw_ind]

            E_bind = Si_E_bind[ss_ind]
            delta_E = hw - E_bind
            if delta_E < 0:
                print('delta E < 0 !!!')

            phi = 2 * np.pi * np.random.random()
            phi_2nd = phi - np.pi

            sin2 = delta_E / E
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

        E -= hw
        flight_ort = new_flight_ort

        e_DATA_line = [e_id, par_id, proc_ind, *coords, E_bind, E_2nd, E]
        e_DATA_deque.append(e_DATA_line)

    e_DATA_line = [e_id, par_id, -2, *coords, E, 0, 0]
    e_DATA_deque.append(e_DATA_line)

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
n_electrons_in_file = 100

E0 = 10e+3

for n in range(n_files):

    print('File #' + str(n))

    e_DATA = track_all_electrons(n_electrons_in_file, E0)
    np.save('data/si_si_si/10keV_pl/e_DATA_' + str(n) + '.npy', e_DATA)


# %%
plot_e_DATA(e_DATA)
