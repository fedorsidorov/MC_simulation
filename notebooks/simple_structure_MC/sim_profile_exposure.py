import importlib
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
from functions import MC_functions as mcf
import grid

grid = importlib.reload(grid)
mcf = importlib.reload(mcf)


# %% constants
# d_PMMA = 100
d_PMMA = 1e+10
arr_size = 1000


# %% energies
Wf_PMMA = 4.68
PMMA_E_cut = 3
PMMA_ee_E_bind = [0]

Si_E_pl = 16.7
# Si_E_cut = Si_E_pl
Si_E_cut = 30
Si_ee_E_bind = [0, 20.1, 102, 151.1, 1828.9]

E_cut = [PMMA_E_cut, Si_E_cut]
ee_E_bind = [PMMA_ee_E_bind, Si_ee_E_bind]
PMMA_ee_E_bind = [0]


# %% load arrays
elastic_model = 'easy'  # 'easy', 'atomic', 'muffin'
elastic_extrap = ''  # '', 'extrap_'
PMMA_elastic_mult = 0.08
E_10eV_ind = 228

PMMA_elastic_u = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/PMMA/PMMA_'
    + elastic_model + '_u_' + elastic_extrap + 'nm.npy'
)

PMMA_elastic_u[:E_10eV_ind] = PMMA_elastic_u[E_10eV_ind] * PMMA_elastic_mult

PMMA_elastic_u_diff_cumulated = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/PMMA/PMMA_diff_cs_cumulated_'
    + elastic_model + '_' + elastic_extrap + '+1.npy'
)

Si_elastic_u = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/Si/Si_'
    + elastic_model + '_u_' + elastic_extrap + 'nm.npy'
)

Si_elastic_u_diff_cumulated = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/Si/Si_diff_cs_cumulated_'
    + elastic_model + '_' + elastic_extrap + '+1.npy'
)

# e-e
PMMA_electron_u = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Dapor_PMMA_Mermin/final_arrays/u_nm.npy'
)

PMMA_electron_u_diff_cumulated = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Dapor_PMMA_Mermin/final_arrays/diff_u_cumulated.npy'
)

PMMA_electron_u_diff_cumulated[np.where(np.abs(PMMA_electron_u_diff_cumulated - 1) < 1e-10)] = 1

for i in range(arr_size):
    for j in range(arr_size - 1):

        if PMMA_electron_u_diff_cumulated[i, j] == 0 and PMMA_electron_u_diff_cumulated[i, j + 1] == 0:
            PMMA_electron_u_diff_cumulated[i, j] = -2

        if PMMA_electron_u_diff_cumulated[i, arr_size - j - 1] == 1 and \
                PMMA_electron_u_diff_cumulated[i, arr_size - j - 2] == 1:
            PMMA_electron_u_diff_cumulated[i, arr_size - j - 1] = -2

Si_electron_u = np.zeros((arr_size, 5))

for j in range(5):
    Si_electron_u[:, j] = np.load(
        '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/u/u_' + str(j) + '_nm_precised.npy'
    )

Si_electron_u_diff_cumulated = np.zeros((5, arr_size, arr_size))

for n in range(5):
    Si_electron_u_diff_cumulated[n, :, :] = np.load(
        '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/u_diff_cumulated/u_diff_'
        + str(n) + '_cumulated_precised.npy'
    )

    Si_electron_u_diff_cumulated[np.where(np.abs(Si_electron_u_diff_cumulated - 1) < 1e-10)] = 1

    for i in range(arr_size):
        for j in range(arr_size - 1):

            if Si_electron_u_diff_cumulated[n, i, j] == 0 and Si_electron_u_diff_cumulated[n, i, j + 1] == 0:
                Si_electron_u_diff_cumulated[n, i, j] = -2

            if Si_electron_u_diff_cumulated[n, i, arr_size - j - 1] == 1 and \
                    Si_electron_u_diff_cumulated[n, i, arr_size - j - 2] == 1:
                Si_electron_u_diff_cumulated[n, i, arr_size - j - 1] = -2

    zero_inds = np.where(Si_electron_u_diff_cumulated[n, -1, :] == 0)[0]

    if len(zero_inds) > 0:

        zero_ind = zero_inds[0]

        if grid.EE[zero_ind] < Si_ee_E_bind[n]:
            Si_electron_u_diff_cumulated[n, :, zero_ind] = -2


Si_electron_u_diff_cumulated[0, :4, 5] = -2

Si_electron_u_diff_cumulated[1, :301, :297] = -2
Si_electron_u_diff_cumulated[1, 300, 296] = 0

Si_electron_u_diff_cumulated[2, :461, :457] = -2
Si_electron_u_diff_cumulated[2, 460, 456] = 0

Si_electron_u_diff_cumulated[3, :500, :496] = -2
Si_electron_u_diff_cumulated[3, 499, 495] = 0

Si_electron_u_diff_cumulated[4, :745, :742] = -2
Si_electron_u_diff_cumulated[4, 744, 741] = 0

# phonon, polaron
PMMA_phonon_u = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_PMMA/PMMA_ph_IMFP_nm.npy'
)

# 2015
C_polaron = 0.1  # nm^-1
gamma_polaron = 0.15  # eV^-1

PMMA_polaron_u = C_polaron * np.exp(-gamma_polaron * grid.EE)

# total u
PMMA_processes_u = np.vstack((PMMA_elastic_u, PMMA_electron_u, PMMA_phonon_u, PMMA_polaron_u)).transpose()
Si_processes_u = np.vstack((Si_elastic_u, Si_electron_u.transpose())).transpose()

# normed arrays
PMMA_processes_u_norm = np.zeros(np.shape(PMMA_processes_u))

for i in range(len(PMMA_processes_u)):
    if np.sum(PMMA_processes_u[i, :]) != 0:
        PMMA_processes_u_norm[i, :] = PMMA_processes_u[i, :] / np.sum(PMMA_processes_u[i, :])

PMMA_u_total = np.sum(PMMA_processes_u, axis=1)
PMMA_process_indexes = list(range(len(PMMA_processes_u[0, :])))


Si_processes_u_norm = np.zeros(np.shape(Si_processes_u))

for i in range(len(Si_processes_u)):
    if np.sum(Si_processes_u[i, :]) != 0:
        Si_processes_u_norm[i, :] = Si_processes_u[i, :] / np.sum(Si_processes_u[i, :])

Si_u_total = np.sum(Si_processes_u, axis=1)
Si_process_indexes = list(range(len(Si_processes_u[0, :])))

# structure process lists
structure_processes_u = [PMMA_processes_u, Si_processes_u]
structure_u_total = [PMMA_u_total, Si_u_total]
structure_u_norm = [PMMA_processes_u_norm, Si_processes_u_norm]

structure_elastic_u_diff_cumulated = [PMMA_elastic_u_diff_cumulated, Si_elastic_u_diff_cumulated]

structure_process_indexes = [PMMA_process_indexes, Si_process_indexes]

# %% plot PMMA cross sections
# plt.figure(dpi=300)
#
# labels = 'elastic', 'e-e', 'phonon', 'polaron'
#
# for j in range(len(PMMA_processes_u[0])):
#     plt.loglog(grid.EE, PMMA_processes_u[:, j], label=labels[j])
#
# plt.xlabel('E, eV')
# plt.ylabel(r'$\mu$, nm$^{-1}$')
# plt.ylim(1e-5, 1e+2)
# plt.legend()
# plt.grid()
# plt.show()

# %% plot Si cross sections
# plt.figure(dpi=300)
#
# labels = 'elastic', 'e-e 0', 'e-e 1', 'e-e 2', 'e-e 3', 'e-e 4'
#
# for j in range(len(Si_processes_u[0])):
#     plt.loglog(grid.EE, Si_processes_u[:, j], label=labels[j])
#
# plt.xlabel('E, eV')
# plt.ylabel(r'$\mu$, nm$^{-1}$')
# plt.ylim(1e-5, 1e+2)
# plt.legend()
# plt.grid()
# plt.show()


# %% imshow
# plt.figure(dpi=300)
# plt.imshow(PMMA_elastic_u_diff_cumulated)
# plt.imshow(Si_elastic_u_diff_cumulated)
# plt.imshow(PMMA_electron_u_diff_cumulated)
# plt.imshow(Si_electron_u_diff_cumulated[4])
# plt.show()


# %% zz_vac
xx_vac = np.linspace(-1000, 1000, 1000)
zz_vac = (np.cos(xx_vac * 2 * np.pi / 2000) + 1) * 40 * 0

xx_vac_final = np.concatenate(([-1e+6], xx_vac, [1e+6]))
zz_vac_final = np.concatenate(([zz_vac[0]], zz_vac, [zz_vac[-1]]))

fig, ax = plt.subplots(dpi=300)

ax.plot(xx_vac_final, zz_vac_final)
ax.plot(xx_vac_final, np.ones(len(xx_vac_final)) * d_PMMA)

plt.xlim(-1000, 1000)
plt.ylim(-50, 150)

plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.xlabel('x, nm')
plt.ylabel('z, nm')
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
    hw_phonon = 0.1
    phi = 2 * np.pi * np.random.random()
    Ep = E - hw_phonon

    if E * Ep < 0:
        print('sqrt ph error', E, Ep)

    B = (E + Ep + 2 * np.sqrt(E * Ep)) / (E + Ep - 2 * np.sqrt(E * Ep))
    u = np.random.random()
    cos_theta = (E + Ep) / (2 * np.sqrt(E * Ep)) * (1 - B ** u) + B ** u
    theta = np.arccos(cos_theta)
    return hw_phonon, phi, theta


def get_now_z_vac(now_x):
    return mcf.lin_lin_interp(xx_vac_final, zz_vac_final)(now_x)


def track_electron(e_id, par_id, E_0, coords_0, flight_ort_0):
    E = E_0
    coords = coords_0
    flight_ort = flight_ort_0

    if coords[-1] > d_PMMA:  # get layer_ind at the very stary
        layer_ind = 1
    else:
        layer_ind = 0

    # e_DATA_line: [e_id, par_id, layer_ind, proc_id, x_new, y_new, z_new, E_loss, E_2nd, E_new]
    e_DATA_deque = deque()
    e_DATA_initial_line = [e_id, par_id, layer_ind, -1, *coords_0, 0, 0, E]
    e_DATA_deque.append(e_DATA_initial_line)

    e_2nd_deque = deque()
    next_e_2nd_id = 0

    while True:

        if coords[-1] > d_PMMA:  # get layer_ind
            layer_ind = 1
        else:
            layer_ind = 0

        if E < E_cut[layer_ind]:  # check energy
            break

        E_ind = np.argmin(np.abs(grid.EE - E))  # get E_ind

        u1 = np.random.random()
        free_path = -1 / structure_u_total[layer_ind][E_ind] * np.log(1 - u1)  # (1 - u1) !!!
        delta_r = flight_ort * free_path

        # if coords[-1] + delta_r[-1] >= 0:  # electrons remains in the structure
        if coords[-1] + delta_r[-1] >= get_now_z_vac(coords[0] + delta_r[0]):

            if (coords[-1] - d_PMMA) * (coords[-1] + delta_r[-1] - d_PMMA) > 0:  # electron remains in the same layer
                coords = coords + delta_r

            else:  # electron crosses PMMA/Si interface
                d = np.linalg.norm(delta_r) * np.abs(coords[-1] - d_PMMA) / np.abs(delta_r[-1])
                W1 = structure_u_total[layer_ind][E_ind]
                W2 = structure_u_total[1 - layer_ind][E_ind]

                free_path_corr = d + 1 / W2 * (-np.log(1 - u1) - W1 * d)

                if free_path_corr < 0:
                    print('free path corr < 0 !!!')

                delta_r_corr = flight_ort * free_path_corr
                coords = coords + delta_r_corr

                # if par_id == -1:
                #     print(e_id, layer_ind, free_path, free_path_corr)

        else:  # electron trajectory crosses PMMA surface
            cos2_theta = (-flight_ort[-1]) ** 2

            if np.random.random() < get_T_PMMA(E * cos2_theta):  # electron emerges
                if E * cos2_theta < Wf_PMMA:
                    print('Wf problems')

                coords += delta_r * 3
                break

            else:  # electron scatters
                now_z_vac = get_now_z_vac(coords[0])
                factor = (coords[-1] - now_z_vac) / np.abs(delta_r[-1])
                coords += delta_r * factor  # reach surface

                e_DATA_line = [e_id, par_id, layer_ind, -10, *coords, 0, 0, E]
                e_DATA_deque.append(e_DATA_line)

                delta_r[-1] *= -1
                coords += delta_r * (1 - factor)
                flight_ort[-1] *= -1

        proc_ind = np.random.choice(structure_process_indexes[layer_ind], p=structure_u_norm[layer_ind][E_ind, :])

        # handle scattering
        if proc_ind == 0:  # elastic scattering
            phi = 2 * np.pi * np.random.random()
            u2 = np.random.random()
            theta_ind = np.argmin(np.abs(structure_elastic_u_diff_cumulated[layer_ind][E_ind, :] - u2))
            theta = grid.THETA_rad[theta_ind]
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

            e_DATA_line = [e_id, par_id, layer_ind, proc_ind, *coords, 0, 0, E]
            e_DATA_deque.append(e_DATA_line)

        elif (layer_ind == 0 and proc_ind == 1) or (layer_ind == 1 and proc_ind >= 1):  # e-e scattering
            u2 = np.random.random()

            osc_ind = proc_ind - 1

            if layer_ind == 0:  # PMMA
                hw_ind = np.argmin(np.abs(PMMA_electron_u_diff_cumulated[E_ind, :] - u2))
            else:  # Si
                hw_ind = np.argmin(np.abs(Si_electron_u_diff_cumulated[osc_ind, E_ind, :] - u2))

            hw = grid.EE[hw_ind]

            Eb = ee_E_bind[layer_ind][osc_ind]
            delta_E = hw - Eb

            if delta_E < 0:
                print('delta E < 0 !!!')

            phi = 2 * np.pi * np.random.random()
            phi_2nd = phi - np.pi
            # phi_2nd = np.random.random() * 2 * np.pi

            sin2 = delta_E / E
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

            E -= hw

            e_DATA_line = [e_id, par_id, layer_ind, proc_ind, *coords, Eb, E_2nd, E]
            e_DATA_deque.append(e_DATA_line)

        elif layer_ind == 0 and proc_ind == 2:  # phonon
            hw, phi, theta = get_phonon_scat_hw_phi_theta(E)
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)
            E -= hw

            e_DATA_line = [e_id, par_id, layer_ind, proc_ind, *coords, hw, 0, E]
            e_DATA_deque.append(e_DATA_line)

        elif layer_ind == 0 and proc_ind == 3:  # polaron
            e_DATA_line = [e_id, par_id, layer_ind, proc_ind, *coords, E, 0, 0]
            e_DATA_deque.append(e_DATA_line)
            break

        else:
            print('WTF with process_ind')
            new_flight_ort = np.array(([0, 0, 0]))

        flight_ort = new_flight_ort

    if coords[-1] < 0:  # electron emerges from specimen
        e_DATA_final_line = [e_id, par_id, -1, -2, *coords, 0, 0, E]
        e_DATA_deque.append(e_DATA_final_line)

    elif E > E_cut[layer_ind]:  # polaron
        e_DATA_final_line = [e_id, par_id, layer_ind, 3, *coords, E, 0, 0]
        e_DATA_deque.append(e_DATA_final_line)

    elif E < E_cut[layer_ind]:  # electron energy lower than E_cut
        e_DATA_final_line = [e_id, par_id, layer_ind, -2, *coords, E, 0, 0]
        e_DATA_deque.append(e_DATA_final_line)

    return e_DATA_deque, e_2nd_deque


def track_all_electrons(n_electrons, E0):

    e_deque = deque()
    total_e_DATA_deque = deque()

    next_e_id = 0

    x_beg = 0
    z_beg = get_now_z_vac(x_beg) + 1e-1

    for _ in range(n_electrons):
        e_deque.append([
            next_e_id,
            -1,
            E0,
            x_beg, 0, z_beg,
            0, 0, 1]
        )
        next_e_id += 1

    progress_bar = tqdm(total=n_electrons, position=0)

    while e_deque:

        now_e_id, now_par_id, now_E0, now_x0, now_y0, now_z0, now_ort_x, now_ort_y, now_ort_z = e_deque.popleft()

        if now_par_id == -1:
            progress_bar.update()

        now_e_DATA_deque, now_e_2nd_deque = track_electron(
            now_e_id, now_par_id, now_E0,
            np.array([now_x0, now_y0, now_z0]),
            np.array([now_ort_x, now_ort_y, now_ort_z])
        )

        for e_2nd_line in now_e_2nd_deque:
            e_2nd_line[0] += next_e_id

        next_e_id += len(now_e_2nd_deque)

        total_e_DATA_deque = total_e_DATA_deque + now_e_DATA_deque
        e_deque = now_e_2nd_deque + e_deque

    return np.around(np.vstack(total_e_DATA_deque), decimals=4)


#%%
n_files = 100
n_primaries_in_file = 100

# E_beam_arr = [20000]
E_beam_arr = [50, 100, 150, 200, 250, 300, 400, 500]
# E_beam_arr = [400, 500, 700, 1000, 1400]
# E_beam_arr = [50, 100, 150, 200, 250, 300, 400, 500, 700, 1000, 1400]

for n in range(n_files):

    print('File #' + str(n))

    for E_beam in E_beam_arr:

        print(E_beam)
        e_DATA = track_all_electrons(n_primaries_in_file, E_beam)

        e_DATA_outer = e_DATA[np.where(e_DATA[:, 6] < 0)]
        np.save('data/2ndaries/0.08/' + str(E_beam) + '/e_DATA_' + str(n) + '.npy', e_DATA_outer)


# %%
fig, ax = plt.subplots(dpi=300)

for e_id in range(int(np.max(e_DATA[:, 0]) + 1)):
    inds = np.where(e_DATA[:, 0] == e_id)[0]

    if len(inds) == 0:
        continue

    ax.plot(e_DATA[inds, 4], e_DATA[inds, 6], '-', linewidth='1')

ax.plot(xx_vac_final, zz_vac_final)
ax.plot(xx_vac_final, np.ones(len(xx_vac_final)) * d_PMMA)

plt.xlim(-1000, 1000)
plt.ylim(-500, 1500)

# ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
# ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.grid()
plt.show()
