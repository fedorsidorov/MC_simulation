# import
import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from copy import deepcopy
import os
from tqdm import tqdm
from functions import MC_functions as mcf
import constants as const
# from mapping import mapping_2um_500nm as mm
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
pitch_nm = mm.lx  # nm
d_PMMA = mm.d_PMMA


# def func_cos(x_arr, h, A):
#     return h - A * np.cos(2 * np.pi / pitch_nm * x_arr)


xx = mm.x_bins_100nm
zz = mm.d_PMMA * (1 - np.cos(2 * np.pi / pitch_nm * xx)) / 2

# E_beam = 25e+3
# E_beam = 20e+3
# E_beam = 5e+3
# E_beam = 1e+3

time_step = 1

tau = np.load('notebooks/Boyd_Schulz_Zimm/arrays/tau.npy')
Mn_150 = np.load('notebooks/Boyd_Schulz_Zimm/arrays/Mn_150.npy') * 100

# PMMA 950K
PD = 2.47
x_0 = 2714
z_0 = (2 - PD)/(PD - 1)
y_0 = x_0 / (z_0 + 1)

# plt.figure(dpi=300)
# plt.plot(xx, func_cos(xx, 400, 100))
# plt.show()


# %% other functions
def save_profiles(time, is_exposure=True):
    plt.figure(dpi=300)
    plt.plot(xx_total, zz_total, '.-', color='C0', ms=2, label='SE profile')
    plt.plot(xx_centers, d_PMMA - zz_inner_centers, '.-', color='C4', ms=2, label='inner interp')
    plt.plot(xx_bins, d_PMMA - zz_vac_bins, 'r.-', color='C3', ms=2, label='PMMA interp')

    # plt.plot(xx_bins, func_cos(xx_bins, *popt), 'C1', label='1 - cos fit')

    if is_exposure:
        plt.plot(now_x0_array, d_PMMA - now_z0_array, 'm.')
        plt.plot(-now_x0_array, d_PMMA - now_z0_array, 'm.')

    plt.plot(xx_bins, np.zeros(len(xx_bins)), 'k')

    plt.title('profiles, time = ' + str(int(time)))
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.legend()
    plt.grid()
    plt.xlim(-mm.lx / 2, mm.lx / 2)
    plt.ylim(-300, 600)
    plt.savefig(path + 'profiles_' + str(int(time)) + '_s.jpg', dpi=300)
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

    # path_name = 'notebooks/SE/datafiles/datafile_' + str(beam_sigma) + '_' +\
    #             str(zip_length) + '_' + str(power_low) + '_' + '.fe'
    path_name = 'notebooks/SE/datafiles_holo/datafile.fe'

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

kernel_size = 3
Mn_edge = 42000
power_high = 3.4

# PARAMETERS #
# beam_sigma = 300
beam_sigma = 5

zip_length = 150
power_low = 1.4

# n_e_DATA_files = 600
n_e_DATA_files = 100
# n_e_DATA_files = 200
T_C = 150
scission_weight = 0.09  # 150 C - 0.088568
# PARAMETERS #

x_step, z_step = 100, 5
xx_bins, zz_bins = mm.x_bins_100nm, mm.z_bins_5nm
xx_centers, zz_centers = mm.x_centers_100nm, mm.z_centers_5nm

bin_volume = x_step * mm.ly * z_step
bin_n_monomers = bin_volume / const.V_mon_nm3


# %%
# for T_step in [10, 5, 2, 1]:
# for T_step in [10, 5]:
for T_step in [10]:
# for T_step in [0.001]:

    # for exposure_time in range(20, 101, 10):
    # for exposure_time in [25, 30, 35, 40]:
    for exposure_time in [40, 45]:

        zz_vac_bins = np.zeros(len(xx_bins))
        zz_vac_centers = np.zeros(len(xx_centers))
        surface_inds = np.zeros(len(xx_centers)).astype(int)

        zz_inner_centers = np.zeros(len(xx_centers))

        tau_matrix = np.zeros((len(xx_centers), len(zz_centers)))
        Mn_matrix = np.ones((len(xx_centers), len(zz_centers))) * Mn_150[0]
        Mn_centers = np.zeros(len(xx_centers))
        mob_matrix = np.zeros((len(xx_centers), len(zz_centers)))
        mobs_array = np.zeros(len(xx_centers))

        now_time = 0

        # path = '/Volumes/Transcend/SIM_DEBER/150C_resolution_CORR/sigma_5nm/5_keV/' + str(T_step) + 'C_sec/' \
        #        + str(exposure_time) + '/'
        path = '/Volumes/Transcend/SIM_DEBER/150C_resolution_NO_CORR/sigma_5nm/25_keV/' + str(T_step) + 'C_sec/'\
               + str(exposure_time) + '/'

        if not os.path.exists(path):
            os.makedirs(path)

        It = 1.2e-9 * exposure_time  # C
        n_lines = 625

        pitch = pitch_nm * 1e-7  # cm
        ratio = 1.3 / 1
        L_line = pitch * n_lines * ratio

        It_line = It / n_lines  # C
        It_line_l = It_line / L_line

        y_depth = mm.ly * 1e-7  # cm

        sim_dose = It_line_l * y_depth * dose_factor
        n_electrons_required = sim_dose / 1.6e-19
        n_electrons_required_s = int(n_electrons_required / exposure_time)

        n_electrons_in_file = 31

        n_files_required_s = int(n_electrons_required_s / n_electrons_in_file / 2)  # x2 HACK !!!

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
                    # '/Volumes/Transcend/e_DATA_500nm_point_5keV/' + str(pos_enter) + '/e_DATA_Pv_' +
                    '/Volumes/Transcend/e_DATA_500nm_point_25keV/' + str(pos_enter) + '/e_DATA_Pv_' +
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

            # remove vacuum events
            for i in range(len(xx_centers)):
                now_scission_matrix[i, :surface_inds[i]] = 0

            # x2 HACK
            now_scission_matrix += now_scission_matrix[::-1, :]

            for i in range(len(xx_centers)):
                for j in range(len(zz_centers)):
                    now_k_s = now_scission_matrix[i, j] / time_step / bin_n_monomers
                    tau_matrix[i, j] += y_0 * now_k_s * time_step

                    if tau_matrix[i, j] < tau[-1]:
                        Mn_matrix[i, j] = mcf.lin_log_interp(tau, Mn_150)(tau_matrix[i, j])

                    else:
                        Mn_matrix[i, j] = Mn_150[-1]

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

            # popt, _ = curve_fit(func_cos, xx_bins, d_PMMA - zz_vac_bins)

            zz_vac_bins[np.where(zz_vac_bins > 499)] = 499

            save_profiles(now_time, is_exposure=True)

            now_time += time_step

        # % cooling reflow
        TT_cooling = np.arange(80, 151, T_step)[::-1]
        tt = np.ones(len(TT_cooling))

        for n_cooling_step, time_cooling_step in enumerate(tt):

            print(now_time)
            if now_time > 1000:
                break

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
                time_step=int(time_cooling_step)
            )

            # popt, _ = curve_fit(func_cos, xx_bins, d_PMMA - zz_vac_bins)

            save_profiles(now_time, is_exposure=False)

            now_time += time_cooling_step

        np.save(path + 'xx_total.npy', xx_total)
        np.save(path + 'zz_total.npy', zz_total)

        np.save(path + 'xx_bins.npy', xx_bins)
        np.save(path + 'zz_vac_bins.npy', d_PMMA - zz_vac_bins)

        np.save(path + 'xx_centers.npy', xx_centers)
        np.save(path + 'zz_inner_centers.npy', d_PMMA - zz_inner_centers)
