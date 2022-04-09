import importlib
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy

import DEBER_module

importlib.reload(DEBER_module)


# %% functions
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
    plt.plot(xx_centers, mobs_array, label='Mn mobility')
    plt.plot(xx_centers, mobs_centers, '--', label='Mn mobility filt')
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


def save_profiles():
    plt.figure(dpi=300)
    plt.plot(xx_total, zz_total, '.-', color='C0', ms=2, label='SE profile')
    plt.plot(xx_centers, d_PMMA - zz_inner_centers, '.-', color='C4', ms=2, label='inner interp')
    plt.plot(xx_bins, d_PMMA - zz_vac_bins, 'r.-', color='C3', ms=2, label='PMMA interp')

    plt.plot(xx_366_100, zz_366_100, '--', label='final profile 100s')
    plt.plot(xx_356_200_A, zz_356_200_A, '--', label='final profile 200s A')
    plt.plot(xx_356_200_B, zz_356_200_B, '--', label='final profile 200s B')

    plt.plot(now_x0_array, d_PMMA - now_z0_array, 'm.')
    plt.plot(-now_x0_array, d_PMMA - now_z0_array, 'm.')
    plt.plot(xx_bins, np.zeros(len(xx_bins)), 'k')

    plt.title('profiles, time = ' + str(now_time))
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.legend()
    plt.grid()
    plt.xlim(-1500, 1500)
    plt.ylim(-200, 600)
    plt.savefig(path + 'profiles_' + str(now_time) + '_s.jpg', dpi=300)
    plt.close('all')


def save_profiles_final(new_xx_total, new_zz_total_final, new_zz_surface_final, new_zz_inner_final, time):
    final_fname = '/Volumes/Transcend/SIM_DEBER/s' + str(beam_sigma) + '_z' + \
                  str(zip_length) + '_pl' + str(power_low) + '.jpg'

    plt.figure(dpi=300)
    plt.plot(new_xx_total, zz_total, '.-', color='C0', ms=2, label='SE profile')
    plt.plot(xx_centers, zz_inner_centers, '.-', color='C4', ms=2, label='inner interp')
    plt.plot(xx_bins, zz_vac_bins, 'r.-', color='C3', ms=2, label='PMMA interp')

    plt.plot(xx_366_100, zz_366_100, '--', label='final profile 100s')
    plt.plot(xx_356_200_A, zz_356_200_A, '--', label='final profile 200s A')
    plt.plot(xx_356_200_B, zz_356_200_B, '--', label='final profile 200s B')

    plt.plot(xx_bins, np.zeros(len(xx_bins)), 'k')

    plt.title('profiles FINAL ' + str(time))
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.legend()
    plt.grid()
    plt.xlim(-1500, 1500)
    plt.ylim(-200, 600)
    plt.savefig(path + 'profiles_FINAL_' + str(time) + '.jpg', dpi=300)
    # plt.savefig(final_fname, dpi=300)
    plt.close('all')


def get_pos_enter(now_x0):
    now_x0_bin = np.argmin(np.abs(xx_centers - now_x0))
    now_z0 = zz_vac_centers[now_x0_bin]

    position_enter = 0

    for ne in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
        if ne - 50 <= now_z0 < ne:
            position_enter = ne - 50

    return int(position_enter)


def finalize(zz_vac_bins_final, zz_inner_centers_final, time):

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

    mobs_Mn_final_matrix = np.zeros((len(Mn_centers), len(TT)))

    for i, mn in enumerate(Mn_centers):
        for j in range(len(TT)):

            now_eta_Mn = rf.get_viscosity_experiment_Mn(
                T_C=TT[j],
                Mn=Mn_centers[i],
                power_high=power_high,
                power_low=power_low
            )

            mobs_Mn_final_matrix[i, j] = rf.get_SE_mobility(now_eta_Mn)

    tt_weights = tt / np.sum(tt)

    # mobs_1d_array_final_average = np.average(mobs_Mn_final_matrix, axis=1)
    mobs_1d_array_final_average = np.average(mobs_Mn_final_matrix, axis=1, weights=tt_weights)
    mobs_array_final_filt = medfilt(mobs_1d_array_final_average, kernel_size=kernel_size)
    mobs_final = mobs_array_final_filt

    save_mobilities()

    xx_SE = np.zeros(len(xx_bins) + len(xx_centers) - 1)
    zz_vac_SE = np.zeros(len(xx_bins) + len(xx_centers) - 1)
    mobs_SE = np.zeros(len(xx_bins) + len(xx_centers) - 1)

    for i in range(len(xx_centers)):
        xx_SE[2 * i] = xx_bins[i]
        xx_SE[2 * i + 1] = xx_centers[i]

        zz_vac_SE[2 * i] = zz_vac_bins_final[i]
        zz_vac_SE[2 * i + 1] = zz_inner_centers_final[i]

    mobs_SE[0] = mobs_final[0]
    mobs_SE[-1] = mobs_final[-1]
    mobs_SE[1:-1] = mcf.lin_lin_interp(xx_centers, mobs_final)(xx_SE[1:-1])

    zz_PMMA_SE = d_PMMA - zz_vac_SE
    zz_PMMA_SE = zz_PMMA_SE + d_PMMA
    zz_PMMA_SE[np.where(zz_PMMA_SE < 0)] = 1

    xx_SE_final = np.concatenate((xx_SE - mm.lx, xx_SE, xx_SE + mm.lx))
    zz_PMMA_SE_final = np.concatenate((zz_PMMA_SE, zz_PMMA_SE, zz_PMMA_SE))
    mobs_SE_final = np.concatenate((mobs_SE, mobs_SE, mobs_SE))

    ef.create_datafile_latest_um(
        yy=xx_SE_final * 1e-3,
        zz=zz_PMMA_SE_final * 1e-3,
        width=mm.ly * 1e-3,
        mobs=mobs_SE_final * 3,  # x3 time hack !!!
        path='notebooks/SE/datafile_DEBER_2022_final.fe'
    )

    ef.run_evolver(
        file_full_path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/datafile_DEBER_2022_final.fe',
        commands_full_path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/commands_2022_final.txt'
    )

    profile_surface = ef.get_evolver_profile(  # surface
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

    profile_total_final = ef.get_evolver_profile(  # total
        path='/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_total.txt'
    ) * 1000

    new_xx_total_final, new_zz_total_final = profile_total_final[::2, 0], profile_total_final[::2, 1]
    new_zz_total_final -= d_PMMA

    save_profiles_final(new_xx_total_final, new_zz_total_final, new_zz_surface_final, new_zz_inner_final, time)


def make_SE_1s_iteration(zz_vac_bins, zz_inner_centers, mobs_centers):

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

    ef.create_datafile_latest_um(
        yy=xx_SE_final * 1e-3,
        zz=zz_PMMA_SE_final * 1e-3,
        width=mm.ly * 1e-3,
        mobs=mobs_SE_final,
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


# def simulate_cooling_reflow(zz_vac_bins, zz_inner_centers, Mn_centers):



# %% SIMULATION
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

beam_sigma = 250

kernel_size = 3
zip_length = 190
power_high = 3.4
power_low = 3
Mn_edge = 42000

path = '/Volumes/Transcend/SIM_DEBER/s' + str(beam_sigma) + '_z' +\
       str(zip_length) + '_pl' + str(power_low) + '/'

if not os.path.exists(path):
    os.makedirs(path)

MM = np.logspace(2, 6, 10)
ETA = np.zeros(len(MM))

for i in range(len(ETA)):
    ETA[i] = rf.get_viscosity_experiment_Mn(T_C, MM[i], power_high, power_low, Mn_edge=Mn_edge)

save_eta()

now_time = 0

# while now_time < 1:
while now_time < exposure_time:
# while now_time < 200:

    print('Now time =', now_time)

    zz_vac_centers = mcf.lin_lin_interp(xx_bins, zz_vac_bins)(xx_centers)

    # TODO zz_vac_center_inds == surface_inds ?
    for i in range(len(xx_centers)):
        surface_inds[i] = np.where(zz_centers > zz_vac_centers[i])[0][0]

    now_x0_array = np.zeros(30)
    now_z0_array = np.zeros(30)
    now_scission_matrix = np.zeros((len(xx_centers), len(zz_centers)))

    # GET SCISSION MATRIX
    # for n in range(60):
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

    save_scissions()
    # save_tau_matrix()
    save_surface_inds()
    # save_Mn_matrix()

    zz_PMMA_centers = d_PMMA - zz_vac_centers
    zz_PMMA_inner = d_PMMA - zz_inner_centers

    ratio_array = (zz_PMMA_inner + (zz_PMMA_centers - zz_PMMA_inner) / 2) / zz_PMMA_centers
    save_ratio()

    zip_length_matrix = np.ones(np.shape(now_scission_matrix)) * zip_length
    # zip_length_matrix = Mn_matrix / 100 * Mn_factor

    monomer_matrix = now_scission_matrix * zip_length_matrix
    monomer_array = np.sum(monomer_matrix, axis=1)
    monomer_array *= ratio_array
    # save_monomers()

    delta_h_array = monomer_array * const.V_mon_nm3 / x_step / mm.ly * 2  # triangle!
    new_zz_inner_centers = zz_inner_centers + delta_h_array

    mobs_array = np.zeros(len(xx_centers))

    for i in range(len(mobs_array)):
        mobs_array[i] = np.average(mob_matrix[i, surface_inds[i]:surface_inds[i] + 10])
        Mn_centers[i] = np.average(Mn_matrix[i, surface_inds[i]:surface_inds[i] + 10])

    mobs_centers = medfilt(mobs_array, kernel_size=kernel_size)

    save_mobilities()

    xx_total, zz_total, zz_vac_bins, zz_inner_centers = make_SE_1s_iteration(
        zz_vac_bins=zz_vac_bins,
        zz_inner_centers=new_zz_inner_centers,
        mobs_centers=mobs_centers
    )

    save_profiles()

    now_time += time_step
