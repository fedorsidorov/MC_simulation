# import
import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from copy import deepcopy
import os
from tqdm import tqdm
from functions import MC_functions as mcf
import constants as const
from mapping import mapping_3um_500nm as mm
from functions import SE_functions_new as ef
from functions import array_functions as af
from functions import reflow_functions as rf
import indexes as ind

af = importlib.reload(af)
const = importlib.reload(const)
ef = importlib.reload(ef)
ind = importlib.reload(ind)
mcf = importlib.reload(mcf)
mm = importlib.reload(mm)
rf = importlib.reload(rf)


# %% constants
xx_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/xx_366_zero.npy')
zz_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/zz_366_zero.npy')

d_PMMA = 500

time_step = 1

# PMMA 950K
PD = 2.47
x_0 = 2714
z_0 = (2 - PD)/(PD - 1)
y_0 = x_0 / (z_0 + 1)


# %% other functions
weights = np.load('notebooks/G_value/samples/weights.npy')

TT_1 = np.load('notebooks/G_value/samples/TT_sim_1.npy')
TT_2 = np.load('notebooks/G_value/samples/TT_sim_2.npy')
TT_3 = np.load('notebooks/G_value/samples/TT_sim_3.npy')
TT_4 = np.load('notebooks/G_value/samples/TT_sim_4.npy')
TT_5 = np.load('notebooks/G_value/samples/TT_sim_5.npy')

TT = (TT_1 + TT_2 + TT_3 + TT_4 + TT_5) / 5

# plt.figure(dpi=300)
# plt.plot(TT, weights)
# plt.show()


# def get_ps(Temp_C):
#     return mcf.lin_lin_interp(TT, weights)(Temp_C)


def save_profiles(time, is_exposure=True):
    plt.figure(dpi=300)
    plt.plot(xx_total, zz_total, '.-', color='C0', ms=2, label='SE profile')
    plt.plot(xx_centers, d_PMMA - zz_inner_centers, '.-', color='C4', ms=2, label='inner interp')
    plt.plot(xx_bins, d_PMMA - zz_vac_bins, 'r.-', color='C3', ms=2, label='PMMA interp')

    plt.plot(xx_366, zz_366 + 75, '--', color='black', label='experiment')
    plt.plot(xx_366, zz_366 + 100, '--', color='black')

    if is_exposure:
        plt.plot(now_x0_array, d_PMMA - now_z0_array, 'm.')
        plt.plot(-now_x0_array, d_PMMA - now_z0_array, 'm.')

    plt.plot(xx_bins, np.zeros(len(xx_bins)), 'k')

    plt.title('profiles, time = ' + str(time))
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.legend()
    plt.grid()
    plt.xlim(-1500, 1500)
    plt.ylim(-300, 600)
    plt.savefig(path + 'profiles_' + str(time) + '_s.jpg', dpi=300)
    plt.close('all')


def get_pos_enter(now_x0):
    now_x0_bin = np.argmin(np.abs(xx_centers - now_x0))
    now_z0 = zz_vac_centers[now_x0_bin]

    position_enter = 0

    for ne in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
        if ne - 50 <= now_z0 < ne:
            position_enter = ne - 50

    if now_z0 >= 500:
        position_enter = 500

    return int(position_enter)


def get_zip_len(T_C):
    zl_130 = 100
    zl_150 = 150

    return zl_130 + (zl_150 - zl_130) / (150 - 130) * (T_C - 130)


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

    path_name = 'notebooks/SE/datafiles/datafile_T.fe'

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
        path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_surface.txt',
        y_max=mm.y_max
    ) * 1000

    new_xx_surface, new_zz_surface = profile_surface[:, 0], profile_surface[:, 1]
    new_zz_surface -= d_PMMA
    new_zz_surface_final = mcf.lin_lin_interp(new_xx_surface, new_zz_surface)(xx_bins)

    profile_inner = ef.get_evolver_profile(  # inner
        path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_inner.txt',
        y_max=mm.y_max
    ) * 1000

    new_xx_inner, new_zz_inner = profile_inner[:, 0], profile_inner[:, 1]
    new_zz_inner -= d_PMMA
    new_zz_inner_final = mcf.lin_lin_interp(new_xx_inner, new_zz_inner)(xx_centers)

    profile_total = ef.get_evolver_profile(  # total
        path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_total.txt',
        y_max=mm.y_max
    ) * 1000

    new_xx_total, new_zz_total = profile_total[::2, 0], profile_total[::2, 1]
    new_zz_total -= d_PMMA

    new_zz_vac_bins = d_PMMA - new_zz_surface_final
    new_zz_inner_centers = d_PMMA - new_zz_inner_final

    return new_xx_total, new_zz_total, new_zz_vac_bins, new_zz_inner_centers


# %% SIMULATION
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
n_electrons_required_s = int(n_electrons_required / exposure_time)

n_electrons_in_file = 31

n_files_required_s = int(n_electrons_required_s / n_electrons_in_file / 2)

kernel_size = 3
Mn_edge = 42000
power_high = 3.4


# %%
# for T_C in [147, 148, 149, 151, 152, 153]:
for T_C in [148, 149, 151, 152, 153]:

    tau = np.load('notebooks/Boyd_kinetic_curves/arrays/tau.npy')
    Mn_T = np.load('notebooks/Boyd_kinetic_curves/for_vary_T/Mn_' + str(T_C) + '_term.npy') * 100

    E_beam = 20e+3
    beam_sigma = 250
    zip_length = get_zip_len(T_C)
    power_low = 1.4

    n_e_DATA_files = 600

    # scission_weight = get_ps(T_C)
    scission_weight = mcf.lin_lin_interp(TT, weights)(T_C)

    x_step, z_step = 100, 5
    xx_bins, zz_bins = mm.x_bins_100nm, mm.z_bins_5nm
    xx_centers, zz_centers = mm.x_centers_100nm, mm.z_centers_5nm

    bin_volume = x_step * mm.ly * z_step
    bin_n_monomers = bin_volume / const.V_mon_nm3

    for n_try in range(10):

        zz_vac_bins = np.zeros(len(xx_bins))
        zz_vac_centers = np.zeros(len(xx_centers))
        surface_inds = np.zeros(len(xx_centers)).astype(int)

        zz_inner_centers = np.zeros(len(xx_centers))

        tau_matrix = np.zeros((len(xx_centers), len(zz_centers)))
        Mn_matrix = np.ones((len(xx_centers), len(zz_centers))) * Mn_T[0]
        Mn_centers = np.zeros(len(xx_centers))
        mob_matrix = np.zeros((len(xx_centers), len(zz_centers)))
        mobs_array = np.zeros(len(xx_centers))

        path = '/Volumes/Transcend/SIM_DEBER/366_vary_params/T_' + str(T_C) + '/try_' + str(n_try) + '/'

        if not os.path.exists(path):
            os.makedirs(path)

        exposure_time = 100
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

            now_x0_array = np.zeros(n_files_required_s)
            now_z0_array = np.zeros(n_files_required_s)
            now_scission_matrix = np.zeros((len(xx_centers), len(zz_centers)))

            # GET SCISSION MATRIX
            for n in range(n_files_required_s):

                n_file = np.random.choice(n_e_DATA_files)

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

            # X2 HACK !!!
            now_scission_matrix += now_scission_matrix[::-1, :]

            for i in range(len(xx_centers)):
                for j in range(len(zz_centers)):
                    now_k_s = now_scission_matrix[i, j] / time_step / bin_n_monomers
                    tau_matrix[i, j] += y_0 * now_k_s * time_step
                    Mn_matrix[i, j] = mcf.lin_log_interp(tau, Mn_T)(tau_matrix[i, j])
                    mob_matrix[i, j] = rf.move_Mn_to_mobs(
                        Mn=Mn_matrix[i, j],
                        T_C=T_C,
                        power_high=power_high,
                        power_low=power_low,
                        Mn_edge=Mn_edge
                    )

            zz_PMMA_centers = d_PMMA - zz_vac_centers
            zz_PMMA_inner = d_PMMA - zz_inner_centers

            ratio_array = (zz_PMMA_inner + (zz_PMMA_centers - zz_PMMA_inner) / 2) / zz_PMMA_centers

            zip_length_matrix = np.ones(np.shape(now_scission_matrix)) * zip_length

            monomer_matrix = now_scission_matrix * zip_length_matrix
            monomer_array = np.sum(monomer_matrix, axis=1)
            monomer_array *= ratio_array

            delta_h_array = monomer_array * const.V_mon_nm3 / x_step / mm.ly * 2  # triangle!
            new_zz_inner_centers = zz_inner_centers + delta_h_array

            for i in range(len(xx_centers)):
                Mn_centers[i] = np.average(Mn_matrix[i, surface_inds[i]:])
                mobs_array[i] = np.average(mob_matrix[i, surface_inds[i]:])

            mobs_centers = medfilt(mobs_array, kernel_size=kernel_size)

            xx_total, zz_total, zz_vac_bins, zz_inner_centers = make_SE_iteration(
                zz_vac_bins=zz_vac_bins,
                zz_inner_centers=new_zz_inner_centers,
                mobs_centers=mobs_centers,
                time_step=1
            )

            if now_time % 5 == 0:
                save_profiles(now_time, is_exposure=True)

            now_time += time_step

        # % cooling reflow
        TT_cooling = np.array([150,
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

        for n_cooling_step, time_cooling_step in enumerate(tt):

            print(now_time)

            mobs_centers = np.zeros(len(Mn_centers))

            for j in range(len(Mn_centers)):
                mobs_centers[j] = rf.move_Mn_to_mobs(
                    Mn=Mn_centers[j],
                    T_C=TT_cooling[n_cooling_step],
                    power_high=power_high,
                    power_low=power_low
                )

            xx_total, zz_total, zz_vac_bins, zz_inner_centers = make_SE_iteration(
                zz_vac_bins=zz_vac_bins,
                zz_inner_centers=zz_inner_centers,
                mobs_centers=mobs_centers,
                time_step=time_cooling_step
            )

            save_profiles(now_time, is_exposure=False)

            now_time += time_cooling_step

            if now_time > 300:
                break

        np.save(path + 'xx_total.npy', xx_total)
        np.save(path + 'zz_total.npy', zz_total)

        np.save(path + 'xx_bins.npy', xx_bins)
        np.save(path + 'zz_vac_bins.npy', d_PMMA - zz_vac_bins)

        np.save(path + 'xx_centers.npy', xx_centers)
        np.save(path + 'zz_inner_centers.npy', d_PMMA - zz_inner_centers)


# %%
tau = np.load('notebooks/Boyd_kinetic_curves/arrays/tau.npy')
Mn_147 = np.load('notebooks/Boyd_kinetic_curves/for_vary_T/Mn_147_term.npy')
Mn_148 = np.load('notebooks/Boyd_kinetic_curves/for_vary_T/Mn_148_term.npy')
Mn_149 = np.load('notebooks/Boyd_kinetic_curves/for_vary_T/Mn_149_term.npy')
Mn_151 = np.load('notebooks/Boyd_kinetic_curves/for_vary_T/Mn_151_term.npy')
Mn_152 = np.load('notebooks/Boyd_kinetic_curves/for_vary_T/Mn_152_term.npy')
Mn_153 = np.load('notebooks/Boyd_kinetic_curves/for_vary_T/Mn_153_term.npy')

plt.figure(dpi=300)
plt.semilogy(tau, Mn_147)
plt.semilogy(tau, Mn_148)
plt.semilogy(tau, Mn_149)
plt.semilogy(tau, Mn_151)
plt.semilogy(tau, Mn_152)
plt.semilogy(tau, Mn_153)
plt.show()


