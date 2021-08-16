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
arr_size = 1000

# %% energies
Si_E_pl = 16.7
Si_E_cut = Si_E_pl
# Si_E_cut = 30
Si_ee_E_bind = [0, 20.1, 102, 151.1, 1828.9]


# %% load arrays
# elastic_model = 'easy'  # 'easy', 'atomic', 'muffin'
elastic_model = 'muffin'  # 'easy', 'atomic', 'muffin'
elastic_extrap = ''  # '', 'extrap_'

Si_elastic_u = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/Si/Si_'
    + elastic_model + '_u_' + elastic_extrap + 'nm.npy'
)

Si_elastic_u_diff_cumulated = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/Si/Si_diff_cs_cumulated_'
    + elastic_model + '_' + elastic_extrap + '+1.npy'
)

# e-e

Si_electron_u = np.zeros((arr_size, 5))

for j in range(5):
    Si_electron_u[:, j] = np.load(
        '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/u/u_' + str(j) + '_nm_precised.npy'
    )

Si_electron_u_diff_cumulated = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_structure_MC/arrays/Si_electron_u_diff_cumulated.npy'
)

Si_processes_u = np.vstack((Si_elastic_u, Si_electron_u.transpose())).transpose()

Si_processes_u_norm = np.zeros(np.shape(Si_processes_u))

for i in range(len(Si_processes_u)):
    if np.sum(Si_processes_u[i, :]) != 0:
        Si_processes_u_norm[i, :] = Si_processes_u[i, :] / np.sum(Si_processes_u[i, :])

Si_u_total = np.sum(Si_processes_u, axis=1)
Si_process_indexes = list(range(len(Si_processes_u[0, :])))


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


def get_now_z_vac(now_x, layer_ind=0):
    return 0


def track_electron(e_id, par_id, E_0, coords_0, flight_ort_0):
    E = E_0
    coords = coords_0
    flight_ort = flight_ort_0

    layer_ind = 1

    # e_DATA_line: [e_id, par_id, layer_ind, proc_id, x_new, y_new, z_new, E_loss, E_2nd, E_new]
    e_DATA_deque = deque()
    e_DATA_initial_line = [e_id, par_id, layer_ind, -1, *coords_0, 0, 0, E]
    e_DATA_deque.append(e_DATA_initial_line)

    e_2nd_deque = deque()
    next_e_2nd_id = 0

    while True:

        if E <= Si_E_pl:  # check energy
            break

        E_ind = np.argmin(np.abs(grid.EE - E))  # get E_ind

        u1 = np.random.random()
        free_path = -1 / Si_u_total[E_ind] * np.log(u1)  # (1 - u1) !!!
        # free_path = -1 / Si_u_total[E_ind] * np.log(1 - u1)  # (1 - u1) !!!
        delta_r = flight_ort * free_path

        # electron remains in the same layer
        if coords[-1] + delta_r[-1] > 0:
            coords = coords + delta_r

        else:
            coords += delta_r * 3
            break

        proc_ind = np.random.choice(Si_process_indexes, p=Si_processes_u_norm[E_ind, :])

        # elastic scattering
        if proc_ind == 0:
            phi = 2 * np.pi * np.random.random()
            u2 = np.random.random()
            theta_ind = np.argmin(np.abs(Si_elastic_u_diff_cumulated[E_ind, :] - u2))
            theta = grid.THETA_rad[theta_ind]
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

            e_DATA_line = [e_id, par_id, layer_ind, proc_ind, *coords, 0, 0, E]
            e_DATA_deque.append(e_DATA_line)

            E_2nd = 0

        # e-e scattering
        else:
            u2 = np.random.random()

            osc_ind = proc_ind - 1

            if osc_ind == 0:
                hw = Si_E_pl
                Eb = Si_E_pl

                phi = 2 * np.pi * np.random.random()

                sin2 = hw / E
                theta = np.arcsin(np.sqrt(sin2))

                new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

                E_2nd = 0

            else:
                hw_ind = np.argmin(np.abs(Si_electron_u_diff_cumulated[osc_ind, E_ind, :] - u2))
                hw = grid.EE[hw_ind]
                Eb = Si_ee_E_bind[osc_ind]
                delta_E = hw - Eb

                if delta_E < 0:
                    print('delta E < 0 !!!')

                phi = 2 * np.pi * np.random.random()

                sin2 = delta_E / E

                if sin2 > 1:
                    print('sin2 > 1 !!!', sin2)

                theta = np.arcsin(np.sqrt(sin2))

                new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

                E_2nd = delta_E

                phi_2nd = phi - np.pi
                sin2_2nd = 1 - delta_E / E

                if sin2_2nd < 0:
                    print('sin2_2nd < 0 !!!')

                theta_2nd = np.arcsin(np.sqrt(sin2_2nd))
                flight_ort_2nd = get_scattered_flight_ort(flight_ort, phi_2nd, theta_2nd)

                e_2nd_list = [next_e_2nd_id, e_id, E_2nd, *coords, *flight_ort_2nd]
                e_2nd_deque.append(e_2nd_list)
                next_e_2nd_id += 1

            E -= hw

            e_DATA_line = [e_id, par_id, layer_ind, proc_ind, *coords, Eb, E_2nd, E]
            e_DATA_deque.append(e_DATA_line)

        flight_ort = new_flight_ort

    if coords[-1] < 0:  # electron emerges from specimen
        e_DATA_final_line = [e_id, par_id, -1, -2, *coords, 0, 0, E]
        e_DATA_deque.append(e_DATA_final_line)

    elif E < Si_E_pl:  # electron energy lower than E_cut
        e_DATA_final_line = [e_id, par_id, layer_ind, -2, *coords, E, 0, 0]
        e_DATA_deque.append(e_DATA_final_line)

    return e_DATA_deque, e_2nd_deque


def track_all_electrons(n_electrons, E0):
    e_deque = deque()
    total_e_DATA_deque = deque()

    next_e_id = 0

    x_beg = 0
    z_beg = get_now_z_vac(x_beg) + 1e-2

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


# %%
n_files = 100
n_primaries_in_file = 100

E_beam_arr = [1000]
# E_beam_arr = [100]
# E_beam_arr = [50, 100, 150, 200, 250, 300, 400, 500]
# E_beam_arr = [400, 500, 700, 1000, 1400]
# E_beam_arr = [50, 100, 150, 200, 250, 300, 400, 500, 700, 1000, 1400]

# for n in range(1):
for n in range(n_files):

    print('File #' + str(n))

    for E_beam in E_beam_arr:
        print(E_beam)
        e_DATA = track_all_electrons(n_primaries_in_file, E_beam)

        # e_DATA_outer = e_DATA[np.where(e_DATA[:, 6] < 0)]
        # np.save('data/2ndaries/0.08/' + str(E_beam) + '/e_DATA_' + str(n) + '.npy', e_DATA_outer)

        np.save('data/4Akkerman/1keV_pl/e_DATA_' + str(n) + '.npy', e_DATA)

# %%
fig, ax = plt.subplots(dpi=300)

for e_id in range(int(np.max(e_DATA[:, 0]) + 1)):
    inds = np.where(e_DATA[:, 0] == e_id)[0]

    if len(inds) == 0:
        continue

    ax.plot(e_DATA[inds, 4], e_DATA[inds, 6], '-', linewidth='1')

plt.xlim(-1000, 1000)
plt.ylim(-500, 1500)

# plt.xlim(-250, -150)
# plt.ylim(-50, 50)

# ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
# ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.grid()
plt.show()
