# import
import importlib
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import medfilt
from copy import deepcopy
import os
from tqdm import tqdm
from scipy.optimize import curve_fit
from functions import MC_functions as mcf
import grid
import constants as const
from mapping import mapping_3um_500nm as mm
from functions import SE_functions_new as ef
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import reflow_functions as rf
import indexes as ind

af = importlib.reload(af)
const = importlib.reload(const)
ef = importlib.reload(ef)
emf = importlib.reload(emf)
grid = importlib.reload(grid)
ind = importlib.reload(ind)
mcf = importlib.reload(mcf)
mm = importlib.reload(mm)
rf = importlib.reload(rf)


# %% constants
arr_size = 1000
 
Wf_PMMA = 4.68
PMMA_E_cut = 3.3  # Aktary 2006
PMMA_electron_E_bind = [0]

Si_E_pl = 16.7
Si_E_cut = Si_E_pl
Si_electron_E_bind = [Si_E_pl, 20.1, 102, 151.1, 1828.9]

structure_E_cut = [PMMA_E_cut, Si_E_cut]
structure_electron_E_bind = [PMMA_electron_E_bind, Si_electron_E_bind]

elastic_model = 'easy'  # 'easy', 'atomic', 'muffin'
elastic_extrap = ''  # '', 'extrap_'
PMMA_elastic_factor = 0.02
E_10eV_ind = 228

# simulation constants
xx_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/xx_366_zero.npy')
zz_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/zz_366_zero.npy')

# xx_356_200_A = np.load('notebooks/DEBER_simulation/exp_profiles/xx_356_C_slice_1.npy')
# zz_356_200_A = np.load('notebooks/DEBER_simulation/exp_profiles/zz_356_C_slice_1.npy')

# xx_356_200_B = np.load('notebooks/DEBER_simulation/exp_profiles/xx_356_C_slice_3.npy')
# zz_356_200_B = np.load('notebooks/DEBER_simulation/exp_profiles/zz_356_C_slice_3.npy')

# xx_359_100 = np.load('notebooks/DEBER_simulation/exp_profiles/xx_359y_slice_D1.npy')
# zz_359_100 = np.load('notebooks/DEBER_simulation/exp_profiles/zz_359y_slice_D1.npy')

dose_factor = 3.8

exposure_time = 100
It = 1.2e-9 * exposure_time  # C
n_lines = 625

pitch = 3e-4  # cm
ratio = 1.3 / 1
L_line = pitch * n_lines * ratio

It_line = It / n_lines  # C
It_line_l = It_line / L_line

y_depth = mm.ly * 1e-7  # cm

sim_dose = It_line_l * y_depth * dose_factor
n_electrons_required = sim_dose / 1.6e-19
n_electrons_required_s = int(n_electrons_required / exposure_time)  # 1870.77

n_electrons_in_file = 31

T_C = 150
scission_weight = 0.09  # 150 C - 0.088568

d_PMMA = 500
E_beam = 20e+3

time_step = 1

tau = np.load('notebooks/Boyd_kinetic_curves/arrays/tau.npy')
Mn_150 = np.load('notebooks/Boyd_kinetic_curves/arrays/Mn_150.npy') * 100

# PMMA 950K
PD = 2.47
x_0 = 2714
z_0 = (2 - PD)/(PD - 1)
y_0 = x_0 / (z_0 + 1)

# %% load arrays
PMMA_elastic_u = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/PMMA/PMMA_'
    + elastic_model + '_u_' + elastic_extrap + 'nm.npy'
)

PMMA_elastic_u[:E_10eV_ind] = PMMA_elastic_u[E_10eV_ind] * PMMA_elastic_factor

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
PMMA_electron_u_diff_cumulated = np.zeros((1, arr_size, arr_size))

PMMA_electron_u_diff_cumulated[0, :, :] = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_structure_MC/arrays/'
    'PMMA_electron_u_diff_cumulated.npy'
)

Si_electron_u = np.zeros((arr_size, 5))

for j in range(5):
    Si_electron_u[:, j] = np.load(
        '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/u/u_' + str(j) + '_nm_precised.npy'
    )

Si_electron_u_diff_cumulated = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_structure_MC/arrays/'
    'Si_electron_u_diff_cumulated.npy'
)

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
structure_electron_u_diff_cumulated = [PMMA_electron_u_diff_cumulated, Si_electron_u_diff_cumulated]

structure_process_indexes = [PMMA_process_indexes, Si_process_indexes]


# %% MC functions
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


def get_now_z_vac(xx_vac, zz_vac, now_x, layer_ind=0):
    if layer_ind == 1:
        return 0

    xx_vac_final = np.concatenate(
        [np.array([-1e+6]), xx_vac - mm.lx * 2, xx_vac - mm.lx, xx_vac, xx_vac + mm.lx, xx_vac + mm.lx * 2,
         np.array([1e+6])])
    zz_vac_final = np.concatenate(
        [np.array([zz_vac[0]]), zz_vac, zz_vac, zz_vac, zz_vac, zz_vac, np.array([zz_vac[-1]])])

    return mcf.lin_lin_interp(xx_vac_final, zz_vac_final)(now_x)


def get_ee_phi_theta_phi2nd_theta2nd(delta_E, E):

    phi = 2 * np.pi * np.random.random()
    sin2_theta = delta_E / E

    if sin2_theta > 1:
        print('sin2 > 1 !!!', sin2_theta)

    theta = np.arcsin(np.sqrt(sin2_theta))

    sin2_2nd = 1 - sin2_theta

    if sin2_2nd < 0:
        print('sin2_2nd < 0 !!!')

    phi_2nd = phi - np.pi
    theta_2nd = np.arcsin(np.sqrt(sin2_2nd))

    return phi, theta, phi_2nd, theta_2nd


def track_electron(xx_vac, zz_vac, e_id, par_id, E_0, coords_0, flight_ort_0, d_PMMA, z_cut, Pn):
    E = E_0
    coords = coords_0
    flight_ort = flight_ort_0

    outer_flag = False

    if coords[-1] > d_PMMA:  # get layer_ind
        layer_ind = 1
    elif 0 < coords[-1] < d_PMMA:
        layer_ind = 0
    else:
        layer_ind = 2

    # e_DATA_line: [e_id, par_id, layer_ind, proc_id, x_new, y_new, z_new, E_loss, E_2nd, E_new]
    e_DATA_deque = deque()

    e_DATA_initial_line = [e_id, par_id, layer_ind, -1, *coords_0, 0, 0, E]
    e_DATA_deque.append(e_DATA_initial_line)

    e_2nd_deque = deque()
    next_e_2nd_id = 0

    while True:

        if coords[-1] > d_PMMA:  # get layer_ind
            layer_ind = 1
        elif 0 < coords[-1] < d_PMMA:
            layer_ind = 0
        else:
            layer_ind = 2

        if E <= structure_E_cut[layer_ind]:  # check energy
            break

        if coords[-1] > z_cut:
            break

        E_ind = np.argmin(np.abs(grid.EE - E))  # get E_ind

        u1 = np.random.random()
        free_path = -1 / structure_u_total[layer_ind][E_ind] * np.log(1 - u1)  # (1 - u1) !!!
        delta_r = flight_ort * free_path

        # electron remains in the same layer
        if 0 < coords[-1] < d_PMMA and 0 < coords[-1] + delta_r[-1] < d_PMMA or \
                d_PMMA < coords[-1] and d_PMMA < coords[-1] + delta_r[-1]:

            coords = coords + delta_r

        # electron changes layer
        elif 0 < coords[-1] < d_PMMA < coords[-1] + delta_r[-1] or \
                coords[-1] > d_PMMA > coords[-1] + delta_r[-1] > 0:

            d = np.linalg.norm(delta_r) * np.abs(coords[-1] - d_PMMA) / np.abs(delta_r[-1])
            W1 = structure_u_total[layer_ind][E_ind]
            W2 = structure_u_total[1 - layer_ind][E_ind]

            free_path_corr = d + 1 / W2 * (-np.log(1 - u1) - W1 * d)

            if free_path_corr < 0:
                print('free path corr < 0 !!!')

            delta_r_corr = flight_ort * free_path_corr
            coords = coords + delta_r_corr

        # electron is going to emerge from the structure
        elif coords[-1] + delta_r[-1] <= get_now_z_vac(xx_vac, zz_vac, coords[0] + delta_r[0], layer_ind):

            cos2_theta = (-flight_ort[-1]) ** 2

            # electron emerges
            if layer_ind == 1 or np.random.random() < get_T_PMMA(E * cos2_theta):

                if E * cos2_theta < Wf_PMMA and layer_ind == 0:
                    print('Wf problems')

                coords += delta_r * 3

                break

            # electron scatters
            else:
                # print('e surface scattering')
                # now_z_vac = get_now_z_vac(coords[0], layer_ind)
                factor = 0

                coords += delta_r * factor  # reach surface

                if not Pn:
                    e_DATA_line = [e_id, par_id, layer_ind, -5, *coords, 0, 0, E]
                    e_DATA_deque.append(e_DATA_line)

                delta_r[-1] *= -1
                coords += delta_r * (1 - factor)
                flight_ort[-1] *= -1

        else:
            print('WTF else ???')

        proc_ind = np.random.choice(structure_process_indexes[layer_ind], p=structure_u_norm[layer_ind][E_ind, :])

        if coords[-1] <= get_now_z_vac(xx_vac, zz_vac, coords[0], layer_ind):
            outer_flag = True
            break

        # handle scattering

        # elastic scattering
        if proc_ind == 0:
            phi = 2 * np.pi * np.random.random()
            u2 = np.random.random()
            theta_ind = np.argmin(np.abs(structure_elastic_u_diff_cumulated[layer_ind][E_ind, :] - u2))
            theta = grid.THETA_rad[theta_ind]
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

            if not Pn:
                e_DATA_line = [e_id, par_id, layer_ind, proc_ind, *coords, 0, 0, E]
                e_DATA_deque.append(e_DATA_line)

        # e-e scattering
        elif (layer_ind == 0 and proc_ind == 1) or (layer_ind == 1 and proc_ind >= 1):
            u2 = np.random.random()

            if layer_ind == 1 and proc_ind == 1:
                hw = Si_E_pl
                Eb = Si_E_pl

                phi, theta, phi_2nd, theta_2nd = get_ee_phi_theta_phi2nd_theta2nd(hw, E)
                new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)

                E_2nd = 0

            else:
                hw_ind = np.argmin(
                    np.abs(structure_electron_u_diff_cumulated[layer_ind][proc_ind - 1, E_ind, :] - u2)
                )
                hw = grid.EE[hw_ind]
                Eb = structure_electron_E_bind[layer_ind][proc_ind - 1]

                E_2nd = hw - Eb

                phi, theta, phi_2nd, theta_2nd = get_ee_phi_theta_phi2nd_theta2nd(E_2nd, E)

                new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)
                flight_ort_2nd = get_scattered_flight_ort(flight_ort, phi_2nd, theta_2nd)

                e_2nd_list = [next_e_2nd_id, e_id, E_2nd, *coords, *flight_ort_2nd]
                e_2nd_deque.append(e_2nd_list)
                next_e_2nd_id += 1

            E -= hw

            if not (Pn and layer_ind == 1):
                e_DATA_line = [e_id, par_id, layer_ind, proc_ind, *coords, Eb, E_2nd, E]
                e_DATA_deque.append(e_DATA_line)

        # phonon
        elif layer_ind == 0 and proc_ind == 2:
            hw, phi, theta = get_phonon_scat_hw_phi_theta(E)
            new_flight_ort = get_scattered_flight_ort(flight_ort, phi, theta)
            E -= hw

            e_DATA_line = [e_id, par_id, layer_ind, proc_ind, *coords, hw, 0, E]
            e_DATA_deque.append(e_DATA_line)

        # polaron
        elif layer_ind == 0 and proc_ind == 3:
            # e_DATA_line = [e_id, par_id, layer_ind, proc_ind, *coords, E, 0, 0]
            # e_DATA_deque.append(e_DATA_line)
            break

        # any other processes?
        else:
            print('WTF with process_ind')
            new_flight_ort = np.array(([0, 0, 0]))

        flight_ort = new_flight_ort

    if (coords[-1] < 0 or outer_flag) and not Pn:  # electron emerges from specimen
        e_DATA_final_line = [e_id, par_id, -1, -10, *coords, 0, 0, E]
        e_DATA_deque.append(e_DATA_final_line)

    elif E > structure_E_cut[layer_ind]:  # polaron
        e_DATA_final_line = [e_id, par_id, layer_ind, 3, *coords, E, 0, 0]
        e_DATA_deque.append(e_DATA_final_line)

    elif E < structure_E_cut[layer_ind]:  # electron energy lower than E_cut
        e_DATA_final_line = [e_id, par_id, layer_ind, 10, *coords, E, 0, 0]
        e_DATA_deque.append(e_DATA_final_line)

    e_DATA = np.around(np.vstack(e_DATA_deque), decimals=4)

    return e_DATA, e_2nd_deque


def track_all_electrons(xx_vac, zz_vac, n_electrons, E0, beam_sigma, d_PMMA, z_cut, Pn):
    e_deque = deque()
    e_DATA_deque = deque()

    next_e_id = 0

    for _ in range(n_electrons):

        x_beg = np.random.normal(loc=0, scale=beam_sigma)
        z_beg = get_now_z_vac(xx_vac, zz_vac, x_beg) + 1e-2

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

        now_e_DATA, now_e_2nd_deque = track_electron(
            xx_vac,
            zz_vac,
            now_e_id, now_par_id, now_E0,
            np.array([now_x0, now_y0, now_z0]),
            np.array([now_ort_x, now_ort_y, now_ort_z]),
            d_PMMA,
            z_cut,
            Pn
        )

        for e_2nd_line in now_e_2nd_deque:
            e_2nd_line[0] += next_e_id

        next_e_id += len(now_e_2nd_deque)

        # total_e_DATA_deque = total_e_DATA_deque + now_e_DATA_deque

        # for ed in now_e_DATA_deque:
        #     total_e_DATA_deque.append(ed)

        e_DATA_deque.append(now_e_DATA)

        # e_deque = now_e_2nd_deque + e_deque

        for e2d in now_e_2nd_deque:
            e_deque.appendleft(e2d)

    return np.concatenate(e_DATA_deque, axis=0)


# %% other functions
def save_eta():
    plt.figure(dpi=300)
    plt.loglog(MM, ETA)
    plt.title('viscosity graph')
    plt.xlabel('M')
    plt.ylabel(r'$\eta$')
    plt.grid()
    plt.savefig(path + 'ETA.jpg', dpi=300)
    plt.close('all')


def save_ratio():
    plt.figure(dpi=300)
    plt.plot(xx_centers, ratio_array)
    plt.title('ratio')
    plt.xlabel('x, nm')
    plt.ylabel('ratio')
    plt.grid()
    plt.ylim(0, 1.2)

    if not os.path.exists(path + 'ratios/'):
        os.makedirs(path + 'ratios/')

    plt.savefig(path + 'ratios/' + 'ratio_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')


def save_surface_inds():
    plt.figure(dpi=300)
    plt.plot(xx_centers, surface_inds)
    plt.title('surface inds')
    plt.xlabel('x, nm')
    plt.ylabel('surface inds')
    plt.grid()

    if not os.path.exists(path + 's_inds/'):
        os.makedirs(path + 's_inds/')

    plt.savefig(path + 's_inds/' + 's_inds_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')


def save_tau_matrix():
    plt.figure(dpi=300)
    plt.imshow(tau_matrix.transpose())
    plt.colorbar()

    if not os.path.exists(path + 'tau/'):
        os.makedirs(path + 'tau/')

    plt.savefig(path + 'tau/' + 'tau_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')


def save_Mn_matrix():
    plt.figure(dpi=300)
    plt.imshow(Mn_matrix.transpose())
    plt.colorbar()

    if not os.path.exists(path + 'Mn_matrix/'):
        os.makedirs(path + 'Mn_matrix/')

    plt.savefig(path + 'Mn_matrix/' + 'Mn_matrix_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')


def save_zz_vac_centers():
    plt.figure(dpi=300)
    plt.plot(xx_centers, zz_vac_centers, label='zz_vac_centers')
    plt.title('zz_vac_centers, time = ' + str(now_time))
    plt.xlabel('x, nm')
    plt.ylabel('zz_vac_centers')
    plt.xlim(-1500, 1500)
    plt.legend()
    plt.grid()

    if not os.path.exists(path + 'zz_vac_centers/'):
        os.makedirs(path + 'zz_vac_centers/')

    plt.savefig(path + 'zz_vac_centers/' + 'zz_vac_centers_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')


def save_zz_vac_bins():
    plt.figure(dpi=300)
    plt.plot(xx_bins, zz_vac_bins)
    plt.title('zz_vac_bins, time = ' + str(now_time))
    plt.xlabel('x, nm')
    plt.ylabel('zz_vac_bins')
    plt.xlim(-1500, 1500)
    plt.grid()

    if not os.path.exists(path + 'zz_vac_bins/'):
        os.makedirs(path + 'zz_vac_bins/')

    plt.savefig(path + 'zz_vac_bins/' + 'zz_vac_bins_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')


def save_mobilities():
    plt.figure(dpi=300)
    plt.semilogy(xx_centers, mobs_array, label='Mn mobility')
    plt.semilogy(xx_centers, mobs_centers, '--', label='Mn mobility filt')
    plt.title('mobilities, time = ' + str(now_time))
    plt.xlabel('x, nm')
    plt.ylabel('SE mobility')
    plt.xlim(-1500, 1500)
    plt.legend()
    plt.grid()

    if not os.path.exists(path + 'mobs/'):
        os.makedirs(path + 'mobs/')

    plt.savefig(path + 'mobs/' + 'mobilities_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')


def save_monomers():
    plt.figure(dpi=300)
    plt.plot(xx_centers, monomer_array)
    plt.title('monomer array, time = ' + str(now_time))
    plt.xlabel('x, nm')
    plt.ylabel('n_monomers')
    plt.grid()

    if not os.path.exists(path + 'monomers/'):
        os.makedirs(path + 'monomers/')

    plt.savefig(path + '/monomers/monomer_array_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')


def save_scissions():
    plt.figure(dpi=300)

    scission_matrix_plot = deepcopy(now_scission_matrix)

    for i, _ in enumerate(xx_centers):
        scission_matrix_plot[i, surface_inds[i]] = np.max(scission_matrix_plot)

    plt.imshow(scission_matrix_plot.transpose())
    # plt.plot(xx_centers, np.sum(now_scission_matrix, axis=1))

    if not os.path.exists(path + 'scissions/'):
        os.makedirs(path + 'scissions/')

    plt.savefig(path + '/scissions/scissios_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')


def save_profiles(time, is_exposure=True):
    plt.figure(dpi=300)
    # plt.plot(xx_total, zz_total, '.-', color='C0', ms=2, label='SE profile')
    # plt.plot(xx_centers, d_PMMA - zz_inner_centers, '.-', color='C4', ms=2, label='inner interp')
    # plt.plot(xx_bins, d_PMMA - zz_vac_bins, 'r.-', color='C3', ms=2, label='PMMA interp')
    plt.plot(xx_bins, d_PMMA - zz_vac_bins, 'ro-', color='C3', ms=5, label='PMMA')

    # plt.plot(xx_366, zz_366 + 75, '--', color='black', label='experiment')
    # plt.plot(xx_366, zz_366 + 100, '--', color='black')

    # if is_exposure:
        # plt.plot(now_x0_array, d_PMMA - now_z0_array, 'm.')
        # plt.plot(-now_x0_array, d_PMMA - now_z0_array, 'm.')

    plt.plot(xx_bins, np.zeros(len(xx_bins)), 'k')

    # plt.title('profiles, time = ' + str(time))
    plt.title('t = ' + str(time) + ' s', fontsize=18)
    plt.xlabel('x, нм', fontsize=18)
    plt.ylabel('z, нм', fontsize=18)
    # plt.legend()
    plt.grid()
    plt.xlim(-1500, 1500)
    # plt.ylim(-300, 600)
    plt.ylim(0, 600)
    plt.savefig(path + 'profiles_' + str(time) + '_s.jpg', dpi=300)
    plt.close('all')


def get_pos_enter(now_x0):
    now_x0_bin = np.argmin(np.abs(xx_centers - now_x0))
    now_z0 = zz_vac_centers[now_x0_bin]

    position_enter = 0

    for ne in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
        if ne - 50 <= now_z0 < ne:
            position_enter = ne - 50

    return int(position_enter)


def make_SE_iteration(zz_vac_bins, zz_inner_centers, mobs_centers, time_step):

    xx_SE = np.zeros(len(xx_bins) + len(xx_centers) - 1)
    zz_vac_SE = np.zeros(len(xx_bins) + len(xx_centers) - 1)
    mobs_SE = np.zeros(len(xx_bins) + len(xx_centers) - 1)

    for i in range(len(xx_centers)):
        xx_SE[2 * i] = xx_bins[i]
        xx_SE[2 * i + 1] = xx_centers[i]
        zz_vac_SE[2 * i] = zz_vac_bins[i]
        zz_vac_SE[2 * i + 1] = zz_inner_centers[i]

    mobs_SE[0] = mobs_centers[0]
    mobs_SE[-1] = mobs_centers[-1]
    mobs_SE[1:-1] = mcf.lin_lin_interp(xx_centers, mobs_centers)(xx_SE[1:-1])

    zz_PMMA_SE = d_PMMA - zz_vac_SE
    zz_PMMA_SE = zz_PMMA_SE + d_PMMA
    zz_PMMA_SE[np.where(zz_PMMA_SE < 0)] = 1

    xx_SE_final = np.concatenate((xx_SE - mm.lx, xx_SE, xx_SE + mm.lx))
    zz_PMMA_SE_final = np.concatenate((zz_PMMA_SE, zz_PMMA_SE, zz_PMMA_SE))
    mobs_SE_final = np.concatenate((mobs_SE, mobs_SE, mobs_SE))

    path_name = 'notebooks/SE/datafiles/datafile_' + str(beam_sigma) + '_' +\
                str(zip_length) + '_' + str(power_low) + '_' + '.fe'

    ef.create_datafile_latest_um(
        yy=xx_SE_final * 1e-3,
        zz=zz_PMMA_SE_final * 1e-3,
        width=mm.ly * 1e-3,
        mobs=mobs_SE_final,
        path=path_name
    )

    ef.run_evolver(
        file_full_path='/Users/fedor/PycharmProjects/MC_simulation/' + path_name,
        commands_full_path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/commands/commands_2022_' +
                           str(time_step) + 's.txt'
    )

    profile_surface = ef.get_evolver_profile(
        path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_surface.txt'
    ) * 1000

    new_xx_surface, new_zz_surface = profile_surface[:, 0], profile_surface[:, 1]
    new_zz_surface -= d_PMMA
    new_zz_surface_final = mcf.lin_lin_interp(new_xx_surface, new_zz_surface)(xx_bins)

    profile_inner = ef.get_evolver_profile(  # inner
        path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_inner.txt'
    ) * 1000

    new_xx_inner, new_zz_inner = profile_inner[:, 0], profile_inner[:, 1]
    new_zz_inner -= d_PMMA
    new_zz_inner_final = mcf.lin_lin_interp(new_xx_inner, new_zz_inner)(xx_centers)

    profile_total = ef.get_evolver_profile(  # total
        path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_total.txt'
    ) * 1000

    new_xx_total, new_zz_total = profile_total[::2, 0], profile_total[::2, 1]
    new_zz_total -= d_PMMA

    new_zz_vac_bins = d_PMMA - new_zz_surface_final
    new_zz_inner_centers = d_PMMA - new_zz_inner_final
 
    return new_xx_total, new_zz_total, new_zz_vac_bins, new_zz_inner_centers


# %% SIMULATION
kernel_size = 3
Mn_edge = 42000
power_high = 3.4

# PARAMETERS #
# beam_sigma = 300
beam_sigma = 5
zip_length = 150
power_low = 1.4
# PARAMETERS #

x_step, z_step = 100, 5
xx_bins, zz_bins = mm.x_bins_100nm, mm.z_bins_5nm
xx_centers, zz_centers = mm.x_centers_100nm, mm.z_centers_5nm

bin_volume = x_step * mm.ly * z_step
bin_n_monomers = bin_volume / const.V_mon_nm3

zz_vac_bins = np.zeros(len(xx_bins))
zz_vac_centers = np.zeros(len(xx_centers))
surface_inds = np.zeros(len(xx_centers)).astype(int)

zz_inner_centers = np.zeros(len(xx_centers))

tau_matrix = np.zeros((len(xx_centers), len(zz_centers)))
Mn_matrix = np.ones((len(xx_centers), len(zz_centers))) * Mn_150[0]
Mn_centers = np.zeros(len(xx_centers))
mob_matrix = np.zeros((len(xx_centers), len(zz_centers)))
mobs_array = np.zeros(len(xx_centers))

path = '/Volumes/Transcend/SIM_DEBER/150C_100s_beam_sigma_test/new_s' + str(beam_sigma) + '_z' +\
       str(zip_length) + '_pl' + str(power_low) + '/'

if not os.path.exists(path):
    os.makedirs(path)

MM = np.logspace(2, 6, 10)
ETA = np.zeros(len(MM))

for i in range(len(ETA)):
    ETA[i] = rf.get_viscosity_experiment_Mn(T_C, MM[i], power_high, power_low, Mn_edge=Mn_edge)

save_eta()

now_time = 0

while now_time < exposure_time:

    print('Now time =', now_time)

    zz_vac_centers = mcf.lin_lin_interp(xx_bins, zz_vac_bins)(xx_centers)

    for i in range(len(xx_centers)):

        where_inds = np.where(zz_centers > zz_vac_centers[i])[0]

        if len(where_inds) > 0:
            surface_inds[i] = where_inds[0]
        else:
            surface_inds[i] = len(zz_centers) - 1

    now_x0_array = np.zeros(30)
    now_z0_array = np.zeros(30)
    now_scission_matrix = np.zeros((len(xx_centers), len(zz_centers)))

    # GET SCISSION MATRIX
    for n in range(30):

        n_file = np.random.choice(600)

        now_x0 = np.random.normal(loc=0, scale=beam_sigma)
        pos_enter = get_pos_enter(now_x0)

        now_x0_array[n] = now_x0
        now_z0_array[n] = pos_enter

        now_e_DATA_Pv = np.load(
            '/Volumes/Transcend/e_DATA_500nm_point/' + str(pos_enter) + '/e_DATA_Pv_' +
            str(n_file) + '.npy'
        )

        scission_inds = np.where(np.random.random(len(now_e_DATA_Pv)) < scission_weight)[0]
        now_e_DATA_sci = now_e_DATA_Pv[scission_inds, :]

        now_e_DATA_sci[:, ind.e_DATA_x_ind] += now_x0

        af.snake_array(
            array=now_e_DATA_sci,
            x_ind=ind.e_DATA_x_ind,
            y_ind=ind.e_DATA_y_ind,
            z_ind=ind.e_DATA_z_ind,
            xyz_min=[mm.x_min, mm.y_min, -np.inf],
            xyz_max=[mm.x_max, mm.y_max, np.inf]
        )

        now_scission_matrix += np.histogramdd(
            sample=now_e_DATA_sci[:, [ind.e_DATA_x_ind, ind.e_DATA_z_ind]],
            bins=[xx_bins, zz_bins]
        )[0]

    for i in range(len(xx_centers)):
        now_scission_matrix[i, :surface_inds[i]] = 0

    # x2 HACK
    now_scission_matrix += now_scission_matrix[::-1, :]

    for i in range(len(xx_centers)):
        for j in range(len(zz_centers)):
            now_k_s = now_scission_matrix[i, j] / time_step / bin_n_monomers
            tau_matrix[i, j] += y_0 * now_k_s * time_step
            Mn_matrix[i, j] = mcf.lin_log_interp(tau, Mn_150)(tau_matrix[i, j])
            mob_matrix[i, j] = rf.move_Mn_to_mobs(
                Mn=Mn_matrix[i, j],
                T_C=T_C,
                power_high=power_high,
                power_low=power_low,
                Mn_edge=42000
            )

    zz_PMMA_centers = d_PMMA - zz_vac_centers
    zz_PMMA_inner = d_PMMA - zz_inner_centers

    ratio_array = (zz_PMMA_inner + (zz_PMMA_centers - zz_PMMA_inner) / 2) / zz_PMMA_centers

    # save_ratio()
    # save_scissions()
    # save_tau_matrix()
    # save_surface_inds()
    # save_Mn_matrix()

    zip_length_matrix = np.ones(np.shape(now_scission_matrix)) * zip_length
    # zip_length_matrix = Mn_matrix / 100 * Mn_factor

    monomer_matrix = now_scission_matrix * zip_length_matrix
    monomer_array = np.sum(monomer_matrix, axis=1)
    monomer_array *= ratio_array
    # save_monomers()

    delta_h_array = monomer_array * const.V_mon_nm3 / x_step / mm.ly * 2  # triangle!
    new_zz_inner_centers = zz_inner_centers + delta_h_array

    for i in range(len(xx_centers)):
        # Mn_centers[i] = np.average(Mn_matrix[i, surface_inds[i]:surface_inds[i] + 10])
        # mobs_array[i] = np.average(mob_matrix[i, surface_inds[i]:surface_inds[i] + 10])
        Mn_centers[i] = np.average(Mn_matrix[i, surface_inds[i]:])
        mobs_array[i] = np.average(mob_matrix[i, surface_inds[i]:])

    mobs_centers = medfilt(mobs_array, kernel_size=kernel_size)

    xx_total, zz_total, zz_vac_bins, zz_inner_centers = make_SE_iteration(
        zz_vac_bins=zz_vac_bins,
        zz_inner_centers=new_zz_inner_centers,
        mobs_centers=mobs_centers,
        time_step=1
    )

    if now_time % 1 == 0:
        save_mobilities()
        save_profiles(now_time, is_exposure=True)

    now_time += time_step


# % cooling reflow
TT = np.array([150,
               149, 148, 147, 146, 145, 144, 143, 142, 141, 140,
               139, 138, 137, 136, 135, 134, 133, 132, 131, 130,
               129, 128, 127, 126, 125, 124, 123, 122, 121, 120,
               119, 118, 117, 116, 115, 114, 113, 112, 111, 110,
               109, 108, 107, 106, 105, 104, 103, 102, 101, 100,
               99, 98, 97, 96, 95, 94, 93, 92, 91, 90,
               89, 88, 87, 86, 85, 84, 83, 82, 81, 80
               ])

tt = np.array([8,
               4, 4, 3, 2, 5, 2, 4, 3, 3, 3,
               4, 2, 4, 3, 3, 3, 4, 3, 3, 4,
               3, 3, 4, 4, 3, 3, 4, 4, 4, 4,
               3, 4, 4, 5, 4, 4, 4, 5, 4, 4,
               5, 5, 4, 6, 4, 5, 5, 5, 5, 5,
               6, 6, 5, 6, 6, 5, 6, 6, 6, 7,
               7, 6, 7, 6, 8, 7, 7, 6, 9, 9
               ])

# Mn_centers = np.load('notebooks/DEBER_simulation/Mn_centers_test.npy')
# zz_vac_bins = np.load('notebooks/DEBER_simulation/zz_vac_bins_test.npy')
# zz_inner_centers = np.load('notebooks/DEBER_simulation/zz_inner_centers_test.npy')

# %
for n_cooling_step, time_cooling_step in enumerate(tt):

    print(now_time)

    mobs_centers = np.zeros(len(Mn_centers))

    for j in range(len(Mn_centers)):
        mobs_centers[j] = rf.move_Mn_to_mobs(
            Mn=Mn_centers[j],
            T_C=TT[n_cooling_step],
            power_high=power_high,
            power_low=power_low
        )

    xx_total, zz_total, zz_vac_bins, zz_inner_centers = make_SE_iteration(
        zz_vac_bins=zz_vac_bins,
        zz_inner_centers=zz_inner_centers,
        mobs_centers=mobs_centers,
        time_step=time_cooling_step
    )

    save_mobilities()
    save_profiles(now_time, is_exposure=False)

    now_time += time_cooling_step

    if now_time > 300:
        break
