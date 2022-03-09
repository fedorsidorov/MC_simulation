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
# PMMA_E_cut = Wf_PMMA
PMMA_electron_E_bind = [0]

Si_E_pl = 16.7
Si_E_cut = Si_E_pl
Si_electron_E_bind = [Si_E_pl, 20.1, 102, 151.1, 1828.9]

structure_E_cut = [PMMA_E_cut, Si_E_cut]
structure_electron_E_bind = [PMMA_electron_E_bind, Si_electron_E_bind]

elastic_model = 'easy'  # 'easy', 'atomic', 'muffin'
# elastic_model = 'muffin'  # 'easy', 'atomic', 'muffin'
elastic_extrap = ''  # '', 'extrap_'
PMMA_elastic_factor = 0.02
E_10eV_ind = 228


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


# %% experiment constants
xx_366 = np.load('notebooks/DEBER_simulation/exp_profile_366/xx.npy')
zz_366 = np.load('notebooks/DEBER_simulation/exp_profile_366/zz.npy')

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

n_electrons_in_file = 94

E0 = 20e+3
T_C = 150
scission_weight = 0.09  # 150 C - 0.088568

d_PMMA = 500
E_beam = 20e+3
beam_sigma = 500

time_step = 1

# %% SIMULATION
tau = np.load('notebooks/Boyd_kinetic_curves/arrays/tau.npy')
Mn_150 = np.load('notebooks/Boyd_kinetic_curves/arrays/Mn_150.npy') * 100
Mw_150 = np.load('notebooks/Boyd_kinetic_curves/arrays/Mw_150.npy') * 100

# PMMA 950K
PD = 2.47
x_0 = 2714
z_0 = (2 - PD)/(PD - 1)
y_0 = x_0 / (z_0 + 1)

# bin size !!!
x_step, z_step = 10, 10
x_bins, z_bins = mm.x_bins_10nm, mm.z_bins_10nm
x_centers, z_centers = mm.x_centers_10nm, mm.z_centers_10nm

# x_step, z_step = 25, 25
# x_bins, z_bins = mm.x_bins_25nm, mm.z_bins_25nm
# x_centers, z_centers = mm.x_centers_25nm, mm.z_centers_25nm

# x_step, z_step = 50, 50
# x_bins, z_bins = mm.x_bins_50nm, mm.z_bins_50nm
# x_centers, z_centers = mm.x_centers_50nm, mm.z_centers_50nm

# x_step, z_step = 100, 100
# x_bins, z_bins = mm.x_bins_100nm, mm.z_bins_100nm
# x_centers, z_centers = mm.x_centers_100nm, mm.z_centers_100nm

bin_volume = x_step * mm.ly * z_step
bin_n_monomers = bin_volume / const.V_mon_nm3

xx = x_bins
zz_vac = np.zeros(len(xx))

xx_inner = x_centers
zz_inner = np.zeros(len(xx_inner))

tau_matrix = np.zeros((len(x_centers), len(z_centers)))
Mn_matrix = np.ones((len(x_centers), len(z_centers))) * Mn_150[0]
Mw_matrix = np.ones((len(x_centers), len(z_centers))) * Mw_150[0]
eta_Mn_matrix = np.zeros((len(x_centers), len(z_centers)))
eta_Mw_matrix = np.zeros((len(x_centers), len(z_centers)))
mob_Mn_matrix = np.zeros((len(x_centers), len(z_centers)))
mob_Mw_matrix = np.zeros((len(x_centers), len(z_centers)))

# Mn_factor = 5
zip_length = 200

power_high = 3.4
# power_high = 4

power_low = 1.4
# power_low = 4

# kernel_size = 5
kernel_size = 3

path = 'notebooks/DEBER_simulation/zip_' + str(zip_length) + '_kern_' + str(kernel_size) + '_pows_' + \
       str(power_high) + '_' + str(power_low) + '_' + str(x_step) + 'nm/'

if not os.path.exists(path):
    os.makedirs(path)

MM = np.logspace(2, 6, 10)
ETA = np.zeros(len(MM))

for i in range(len(ETA)):
    ETA[i] = rf.get_viscosity_experiment_Mn(T_C, MM[i], power_high, power_low)

plt.figure(dpi=300)
plt.loglog(MM, ETA)
plt.xlabel('M')
plt.ylabel(r'$\eta$')
plt.grid()
# plt.show()
plt.savefig(path + 'ETA.jpg', dpi=300)

now_time = 0

# while now_time < 75:
while now_time < exposure_time:

    print('Now time =', now_time)

    now_scission_matrix_1nm = np.load(
        'notebooks/DEBER_simulation/scission_matrixes_sigma_500nm/scission_matrix_1nm_' + str(now_time) + '.npy'
    )
    zz_vac_1nm = mcf.lin_lin_interp(xx, zz_vac)(mm.x_centers_1nm)

    for i in range(len(mm.x_centers_1nm)):
        now_vac_inds = np.where(mm.z_centers_1nm < zz_vac_1nm[i])[0]

        if len(now_vac_inds) > 0:
            now_scission_matrix_1nm[i, :now_vac_inds[-1]] = 0

    now_scission_matrix = np.zeros((len(x_centers), len(z_centers)))

    x_new_inds = ((mm.x_centers_1nm - mm.x_min) // x_step).astype(int)
    z_new_inds = ((mm.z_centers_1nm - mm.z_min) // z_step).astype(int)

    for i in range(len(mm.x_centers_1nm)):
        for k in range(len(mm.z_centers_1nm)):
            now_scission_matrix[x_new_inds[i], z_new_inds[k]] += now_scission_matrix_1nm[i, k]

    zz_vac_centers = mcf.lin_lin_interp(xx, zz_vac)(xx_inner)
    zz_PMMA_centers = d_PMMA - zz_vac_centers

    ratio_arr = (d_PMMA - zz_inner) / (((d_PMMA - zz_vac_centers) + (d_PMMA - zz_inner)) / 2)

    plt.figure(dpi=300)
    plt.plot(xx_inner, ratio_arr)
    plt.title('ratio array, time = ' + str(now_time))
    plt.grid()
    # plt.show()
    plt.savefig(path + 'ratio_array_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')

    zip_length_matrix = np.ones(np.shape(now_scission_matrix)) * zip_length
    # zip_length_matrix = Mn_matrix / 100 / Mn_factor

    plt.figure(dpi=300)
    plt.imshow(now_scission_matrix.transpose())
    plt.colorbar()
    plt.title('scission matrix, time = ' + str(now_time))
    # plt.show()
    plt.savefig(path + 'scission_matrix_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')

    plt.figure(dpi=300)
    plt.plot(xx_inner, np.sum(now_scission_matrix, axis=1))
    plt.title('scission array, time = ' + str(now_time))
    plt.grid()
    # plt.show()
    plt.savefig(path + 'scission_array_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')

    # plt.figure(dpi=300)
    # plt.imshow(zip_length_matrix.transpose())
    # plt.title('zip_len matrix, time = ' + str(now_time))
    # plt.show()
    # plt.savefig(path + 'zip_len_matrix_' + str(now_time) + '_s.jpg', dpi=300)
    # plt.close('all')

    surface_inds = np.zeros(len(xx_inner))

    for i in range(len(x_centers)):
        for j in range(len(z_centers)):

            if z_centers[j] < zz_vac_centers[i]:
                surface_inds[i] = j + 1

            now_k_s = now_scission_matrix[i, j] / time_step / bin_n_monomers
            tau_matrix[i, j] += y_0 * now_k_s * time_step
            Mn_matrix[i, j] = mcf.lin_log_interp(tau, Mn_150)(tau_matrix[i, j])
            Mw_matrix[i, j] = mcf.lin_log_interp(tau, Mw_150)(tau_matrix[i, j])

            eta_Mn_matrix[i, j] = rf.get_viscosity_experiment_Mn(
                T_C=150,
                Mn=Mn_matrix[i, j],
                power_high=power_high,
                power_low=power_low
            )

            eta_Mw_matrix[i, j] = rf.get_viscosity_experiment_Mw(150, Mw_matrix[i, j], power=power_high)

            mob_Mn_matrix[i, j] = rf.get_SE_mobility(eta_Mn_matrix[i, j])
            mob_Mw_matrix[i, j] = rf.get_SE_mobility(eta_Mw_matrix[i, j])

    monomer_matrix = now_scission_matrix * zip_length_matrix

    monomer_array = np.sum(monomer_matrix, axis=1)
    monomer_array *= ratio_arr

    plt.figure(dpi=300)
    plt.plot(xx_inner, surface_inds)
    plt.title('surface inds, time = ' + str(now_time))
    # plt.show()
    plt.savefig(path + 'surface_inds_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')

    plt.figure(dpi=300)
    plt.plot(xx_inner, monomer_array)
    plt.title('monomer array, time = ' + str(now_time))
    # plt.show()
    plt.savefig(path + 'monomer_array_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')

    delta_h_array = monomer_array * const.V_mon_nm3 / x_step / mm.ly * 2  # triangle!

    new_zz_vac_inner = zz_inner + delta_h_array

    # zz_vac_cen = mcf.lin_lin_interp(xx, zz_vac)(mm.x_centers_1nm)

    # mob_array = np.average(mob_Mn_matrix, axis=1)
    # mob_array = np.average(mob_Mw_matrix, axis=1)

    mobs_array = np.zeros(len(x_centers))

    for i in range(len(mobs_array)):
        mobs_array[i] = mob_Mn_matrix[i, int(surface_inds[i])]

    mobs_array_filt = medfilt(mobs_array, kernel_size=kernel_size)

    zero_level = (mobs_array_filt[0] + mobs_array_filt[-1]) / 2
    var_Mn = np.sum(xx_inner**2 * (mobs_array_filt - zero_level)) / np.sum(mobs_array_filt - zero_level)

    A = np.max(mobs_array_filt - zero_level)

    gauss_mobs = A * np.exp(-xx_inner**2 / (2 * var_Mn)) + zero_level

    mobs_final = gauss_mobs

    plt.figure(dpi=300)
    plt.plot(xx_inner, mobs_array, label='Mn mobility')
    plt.plot(xx_inner, mobs_array_filt, label='Mn mobility filt')
    plt.plot(xx_inner, gauss_mobs, label='Mn gauss')
    plt.title('mobilities, time = ' + str(now_time))
    plt.xlim(-1500, 1500)
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(path + 'mobilities_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')

    xx_evolver = np.zeros(len(x_bins) + len(x_centers) + 2)
    zz_vac_evolver = np.zeros(len(x_bins) + len(x_centers) + 2)

    xx_evolver[0] = x_centers[0] - x_step
    # zz_vac_evolver[0] = new_zz_vac_inner[0]
    zz_vac_evolver[0] = zz_vac[0]

    xx_evolver[1] = x_bins[0]
    zz_vac_evolver[1] = zz_vac[0]

    xx_evolver[-1] = x_centers[-1] + x_step
    zz_vac_evolver[-1] = zz_vac[-1]

    for i in range(len(x_centers)):
        xx_evolver[2 + 2 * i] = x_centers[i]
        xx_evolver[2 + 2 * i + 1] = x_bins[i + 1]

        zz_vac_evolver[2 + 2 * i] = new_zz_vac_inner[i]
        zz_vac_evolver[2 + 2 * i + 1] = zz_vac[i + 1]

    mobs_evolver = np.zeros(len(x_bins) + len(x_centers) + 2)

    mobs_evolver[0] = mobs_final[0]
    mobs_evolver[1] = mobs_final[0]
    mobs_evolver[-2] = mobs_final[-1]
    mobs_evolver[-1] = mobs_final[-1]

    mobs_evolver[2:-2] = mcf.lin_lin_interp(xx_inner, mobs_final)(xx_evolver[2:-2])
    mobs_evolver[2:-2] = mcf.lin_lin_interp(xx_inner, mobs_final)(xx_evolver[2:-2])

    zz_PMMA_evolver = d_PMMA - zz_vac_evolver

    ef.create_datafile_latest_um(
        yy=xx_evolver * 1e-3,
        zz=zz_PMMA_evolver * 1e-3,
        width=mm.ly * 1e-3,
        mobs=mobs_evolver,
        path='notebooks/SE/datafile_DEBER_2022.fe'
    )

    ef.run_evolver(
        file_full_path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/datafile_DEBER_2022.fe',
        commands_full_path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/commands_2022.txt'
    )

    profile_surface = ef.get_evolver_profile(  # surface
        path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_surface.txt'
    ) * 1000

    new_xx_surface, new_zz_surface = profile_surface[:, 0], profile_surface[:, 1]
    new_xx_surface[0], new_xx_surface[-1] = x_bins[0], x_bins[-1]
    new_zz_surface[0] = new_zz_surface[1]
    new_zz_surface[-1] = new_zz_surface[-2]

    np.save(path + 'SE_xx_surface_' + str(now_time) + '.npy', new_xx_surface)
    np.save(path + 'SE_zz_surface_' + str(now_time) + '.npy', new_zz_surface)

    new_zz_surface_final = mcf.lin_lin_interp(new_xx_surface, new_zz_surface)(x_bins)

    np.save(path + 'SE_zz_surface_final' + str(now_time) + '.npy', new_zz_surface_final)

    profile_inner = ef.get_evolver_profile(  # inner
            path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_inner.txt'
        ) * 1000

    new_xx_inner, new_zz_inner = profile_inner[:, 0], profile_inner[:, 1]
    new_xx_inner[0], new_xx_inner[-1] = x_centers[0], x_centers[-1]
    new_zz_inner[0] = new_zz_inner[1]
    new_zz_inner[-1] = new_zz_inner[-2]

    np.save(path + 'SE_xx_inner_' + str(now_time) + '.npy', new_xx_inner)
    np.save(path + 'SE_zz_inner_' + str(now_time) + '.npy', new_zz_inner)

    new_zz_inner_final = mcf.lin_lin_interp(new_xx_inner, new_zz_inner)(x_centers)

    np.save(path + 'SE_zz_inner_final' + str(now_time) + '.npy', new_zz_inner_final)

    profile_total = ef.get_evolver_profile(  # total
        path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_total.txt'
    ) * 1000

    new_xx_total, new_zz_total = profile_total[::2, 0], profile_total[::2, 1]

    np.save(path + 'new_xx_total_' + str(now_time) + '.npy', new_xx_total)
    np.save(path + 'new_zz_total_' + str(now_time) + '.npy', new_zz_total)

    plt.figure(dpi=300)

    plt.plot(new_xx_total, new_zz_total, '.-', ms=2, label='SE profile')
    plt.plot(new_xx_surface, new_zz_surface, '.-', ms=2, label='PMMA')
    plt.plot(new_xx_inner, new_zz_inner, '.-', ms=2, label='inner')
    plt.plot(xx, new_zz_surface_final, '.-', ms=2, label='PMMA interp')
    plt.plot(xx_inner, new_zz_inner_final, '.-', ms=2, label='inner interp')

    plt.plot(xx_366, zz_366, '--', label='final profile')

    plt.title('profiles, time = ' + str(now_time))
    plt.legend()
    plt.grid()
    plt.xlim(-1500, 1500)
    plt.ylim(0, 600)

    # plt.show()
    plt.savefig(path + 'profiles_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')

    zz_vac = d_PMMA - new_zz_surface_final
    zz_inner = d_PMMA - new_zz_inner_final

    now_time += time_step


# %
# plt.figure(dpi=300)
# plt.plot(mm.x_centers_1nm, zz_vac_1nm)
# plt.show()


