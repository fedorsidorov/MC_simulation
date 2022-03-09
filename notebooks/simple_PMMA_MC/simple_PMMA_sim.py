import importlib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

import grid
grid = importlib.reload(grid)

# %% load arrays
model = 'easy'  # 'easy', 'atomic', 'muffin'
extrap = ''  # '', 'extrap_'

PMMA_el_u = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/PMMA/'
                    'PMMA_' + model + '_u_' + extrap + 'nm.npy')

PMMA_el_u[:228] = PMMA_el_u[228] * 0.1

# u_0 = PMMA_el_u[228] * 1e-1
#
# PMMA_el_u[:228] = np.exp(
#     np.log(u_0) +
#     (np.log(grid.EE[:228]) - np.log(grid.EE[0])) *
#     (np.log(PMMA_el_u[228]) - np.log(u_0)) / (np.log(grid.EE[228]) - np.log(grid.EE[0]))
# )

PMMA_el_u_diff_cumulated =\
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/PMMA/'
            'PMMA_diff_cs_cumulated_' + model + '_' + extrap + '+1.npy')

# e-e
PMMA_ee_u = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Dapor_PMMA_Mermin/final_arrays/u_nm.npy')
PMMA_ee_u_diff_cumulated =\
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Dapor_PMMA_Mermin/final_arrays/diff_u_cumulated.npy')

# phonon
PMMA_ph_u = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/'
                    'arrays_PMMA/PMMA_ph_IMFP_nm.npy')

# 2015
Wf_PMMA = 4.68
C_pol = 0.1  # nm^-1
gamma_pol = 0.15  # eV^-1

PMMA_pol_u = C_pol * np.exp(-gamma_pol * grid.EE)

# %
hw_phonon = 0.1

PMMA_E_cut = Wf_PMMA
# PMMA_E_cut = 50  # CASINO

PMMA_u = np.vstack((PMMA_el_u, PMMA_ee_u, PMMA_ph_u, PMMA_pol_u)).transpose()

# norm u array
PMMA_u_norm = np.zeros(np.shape(PMMA_u))
for i in range(len(PMMA_u)):
    if np.sum(PMMA_u[i, :]) != 0:
        PMMA_u_norm[i, :] = PMMA_u[i, :] / np.sum(PMMA_u[i, :])

PMMA_u_total = np.sum(PMMA_u, axis=1)
process_indexes = list(range(len(PMMA_u[0, :])))


# % plot cross sections
plt.figure(dpi=300)

labels = 'elastic', 'e-e', 'phonon', 'polaron'

for j in range(len(PMMA_u[0])):
    plt.loglog(grid.EE, PMMA_u[:, j], label=labels[j])

plt.xlabel('E, eV')
plt.ylabel(r'$\mu$, nm$^{-1}$')
plt.ylim(1e-5, 1e+2)
plt.legend()
plt.grid()
plt.show()

# %%
dapor_P_100 = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Dapor_PMMA_Mermin/curves/'
                         'Dapor_P_100eV.txt')
dapor_P_1000 = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Dapor_PMMA_Mermin/curves/'
                          'Dapor_P_1keV.txt')

E_ind_100 = 454
E_ind_1000 = 681

plt.figure(dpi=300)
plt.semilogx(grid.EE, PMMA_ee_u_diff_cumulated[E_ind_100, :], label='my Mermin 100')
plt.semilogx(grid.EE, PMMA_ee_u_diff_cumulated[E_ind_1000, :], label='my Mermin 1000')

plt.semilogx(dapor_P_100[:, 0], dapor_P_100[:, 1], '--', label='Dapor Drude 100')
plt.semilogx(dapor_P_1000[:, 0], dapor_P_1000[:, 1], '--', label='Dapor Drude 1000')

plt.legend()
plt.xlim(1, 300)
plt.grid()
plt.show()


# %% functions
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

    # e_DATA_line: [e_id, par_id, proc_id, x_new, y_new, z_new, E_loss, E_new]
    e_DATA_deque = deque()
    e_DATA_initial_line = [e_id, par_id, -1, *coords_0, 0, E]
    e_DATA_deque.append(e_DATA_initial_line)

    e_2nd_deque = deque()
    next_e_2nd_id = 0

    #  main cycle
    while E > PMMA_E_cut:

        E_ind = np.argmin(np.abs(grid.EE - E))

        u1 = np.random.random()
        free_path = -1 / PMMA_u_total[E_ind] * np.log(u1)
        delta_r = flight_ort * free_path

        if coords[-1] + delta_r[-1] < 0:  # PMMA surface crossing

            cos2_theta = (-flight_ort[-1]) ** 2

            if np.random.random() < get_T_PMMA(E * cos2_theta):  # electron emerges
                if E * cos2_theta < Wf_PMMA:
                    print('Wf problems')

                coords += delta_r * 3
                break

            else:  # electron scatters TODO different scattering models
                factor = coords[-1] / np.abs(delta_r[-1])
                # factor = 0
                coords += delta_r * factor  # reach surface

                e_DATA_line = [e_id, par_id, -5, *coords, 0, E]
                e_DATA_deque.append(e_DATA_line)

                delta_r[-1] *= -1
                coords += delta_r * (1 - factor)
                flight_ort[-1] *= -1

        coords = coords + delta_r

        proc_ind = np.random.choice(process_indexes, p=PMMA_u_norm[E_ind, :])

        if proc_ind == 0:  # elastic scattering
            phi = 2 * np.pi * np.random.random()
            u2 = np.random.random()
            theta_ind = np.argmin(np.abs(PMMA_el_u_diff_cumulated[E_ind, :] - u2))
            theta = grid.THETA_rad[theta_ind]
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

            e_DATA_line = [e_id, par_id, proc_ind, *coords, 0, E]
            e_DATA_deque.append(e_DATA_line)

        elif proc_ind == 1:  # e-e scattering
            u2 = np.random.random()
            hw_ind = np.argmin(np.abs(PMMA_ee_u_diff_cumulated[E_ind, :] - u2))
            hw = grid.EE[hw_ind]

            if hw < 0:
                print('delta E < 0 !!!')

            phi = 2 * np.pi * np.random.random()
            phi_2nd = phi - np.pi
            # phi_2nd = np.random.random() * 2 * np.pi

            sin2 = hw / E
            # TODO check uniform 2ndary angle distribution
            sin2_2nd = 1 - hw / E

            if sin2 > 1:
                print('sin2 > 1 !!!', sin2)

            if sin2_2nd < 0:
                print('sin2_2nd < 0 !!!')

            theta = np.arcsin(np.sqrt(sin2))
            theta_2nd = np.arcsin(np.sqrt(sin2_2nd))
            # theta_2nd = np.pi * np.random.random()

            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)
            flight_ort_2nd = get_scattered_flight_ort(flight_ort, phi_2nd, theta_2nd)

            E_2nd = hw

            e_2nd_list = [next_e_2nd_id, e_id, E_2nd, *coords, *flight_ort_2nd]
            e_2nd_deque.append(e_2nd_list)
            next_e_2nd_id += 1
            E -= hw

            e_DATA_line = [e_id, par_id, proc_ind, *coords, hw, E]
            e_DATA_deque.append(e_DATA_line)

        elif proc_ind == 2:  # phonon
            hw, phi, theta = get_phonon_scat_hw_phi_theta(E)
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)
            E -= hw

            e_DATA_line = [e_id, par_id, proc_ind, *coords, hw, E]
            e_DATA_deque.append(e_DATA_line)

        elif proc_ind == 3:  # polaron
            e_DATA_line = [e_id, par_id, proc_ind, *coords, E, 0]
            e_DATA_deque.append(e_DATA_line)
            break

        else:
            print('WTF with process_ind')

        flight_ort = new_flight_ort

    if coords[-1] < 0:
        e_DATA_final_line = [e_id, par_id, -2, *coords, 0, E]
        e_DATA_deque.append(e_DATA_final_line)

    elif E > 0:
        e_DATA_final_line = [e_id, par_id, -2, *coords, E, 0]
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
# n_files = 1
n_primaries_in_file = 100
# n_primaries_in_file = 10

# E_beam_arr = [1000]
E_beam_arr = [50, 100, 150, 200, 250, 300, 400, 500]
# E_beam_arr = [400, 500, 700, 1000, 1400]
# E_beam_arr = [50, 100, 150, 200, 250, 300, 400, 500, 700, 1000, 1400]

for n in range(n_files):

    print('File #' + str(n))

    for E_beam in E_beam_arr:

        print(E_beam)
        e_DATA = track_all_electrons(n_primaries_in_file, E_beam)
        e_DATA_outer = e_DATA[np.where(e_DATA[:, 5] < 0)]

        np.save('data/2ndaries/0.1_new/' + str(E_beam) + '/e_DATA_' + str(n) + '.npy', e_DATA_outer)

# %%
# ans = np.load('data/4CASINO/1000/e_DATA_0.npy')

# %%
e_DATA = now_e_DATA[:, :]
# e_DATA = now_e_DATA_Pv[:, :]

fig, ax = plt.subplots(dpi=300)

for e_id in range(int(np.max(e_DATA[:, 0]) + 1)):
    inds = np.where(e_DATA[:, 0] == e_id)[0]

    if len(inds) == 0:
        continue

    ax.plot(e_DATA[inds, 4], e_DATA[inds, 6], '-', linewidth='1')

ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

# plt.xlim(-1, 1)
# plt.ylim(-0.5, 1.5)

plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.grid()
plt.show()
