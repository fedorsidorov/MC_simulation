import importlib
import matplotlib.pyplot as plt
import numpy as np
import constants as const
from mapping import mapping_5um_900nm as mm
from functions import MC_functions as mcf
from functions import DEBER_functions as df
from functions import SE_functions as ef

const = importlib.reload(const)
mm = importlib.reload(mm)
mcf = importlib.reload(mcf)
df = importlib.reload(df)
ef = importlib.reload(ef)

# %%
profile_0p2 = np.loadtxt('notebooks/DEBER_simulation/exp_curves/exp_900nm_5um_0.2uC_cm2.txt')

profile_0p2[0, 1] = profile_0p2[:, 1].max()
profile_0p2[-1, 1] = profile_0p2[:, 1].max()

xx_vac_0p2 = profile_0p2[:, 0] * 1000 - 6935.08
zz_vac_0p2 = (profile_0p2[:, 1].max() - profile_0p2[:, 1]) * 1000

# plt.figure(dpi=300)
# plt.plot(xx_vac_0p2, zz_vac_0p2)
# plt.plot(xx_vac_0p2, zz_vac_0p2 * 0.7)
# plt.plot(xx_vac_0p2, zz_vac_0p2 * 0.5)
# plt.plot(xx_vac_0p2, zz_vac_0p2 * 0.3)
# plt.grid()
# plt.show()

# %%
# xx_vac = mm.x_centers_5nm  # nm
xx_vac = mm.x_centers_50nm  # nm
# zz_vac = np.ones(len(xx_vac)) * 0  # nm

# n_steps = 59
n_steps = 235
# zip_length = 4000
SE_mobility_arr = np.ones(n_steps) * 0.01
zip_length_arr = np.zeros(n_steps) * 1000

SE_mobility_arr[0:10] = 0.1
zip_length_arr[0:10] = 10000
SE_mobility_arr[10:20] = 1
zip_length_arr[10:20] = 7000
SE_mobility_arr[20:30] = 1
zip_length_arr[20:30] = 10000
SE_mobility_arr[30:40] = 1
zip_length_arr[30:40] = 17000
SE_mobility_arr[40:50] = 1
zip_length_arr[40:50] = 23000
SE_mobility_arr[50:60] = 1
zip_length_arr[50:60] = 35000

now_n_step = 0

if now_n_step == 0:
    zz_vac = np.ones(len(xx_vac)) * 0
else:
    zz_vac = np.load('notebooks/DEBER_simulation/podgon_monomer/' + str(now_n_step - 1) + '_zz_vac_' +
                     str(zip_length_arr[now_n_step - 1]) + '_' + str(SE_mobility_arr[now_n_step - 1]) + '.npy')

# for n_step in range(now_n_step, now_n_step + 1):
for n_step in range(now_n_step, now_n_step + 10):

    zip_length = zip_length_arr[n_step]

    now_scission_matrix = np.load('notebooks/DEBER_simulation/vary_zz_vac_0p2_1s/scission_matrix_'
                                  + str(n_step) + '.npy')

    if n_step == 50:
        now_scission_matrix = np.load('notebooks/DEBER_simulation/vary_zz_vac_0p2_1s/scission_matrix_'
                                      + str(49) + '.npy')

    now_monomers_array = np.sum(now_scission_matrix, axis=1) * zip_length

    now_monomers_array_50 = np.histogram(
        a=mm.x_centers_5nm,
        bins=mm.x_bins_50nm,
        weights=now_monomers_array
    )[0]

    # zz_vac += now_monomers_array * const.V_mon_nm3 / mm.step_5nm / mm.ly
    zz_vac += now_monomers_array_50 * const.V_mon_nm3 / mm.step_50nm / mm.ly
    zz_PMMA = mm.d_PMMA - zz_vac

    # xx_for_evolver = np.concatenate([[mm.x_bins_5nm[0]], mm.x_centers_5nm, [mm.x_bins_5nm[-1]]])
    xx_for_evolver = np.concatenate([[mm.x_bins_50nm[0]], mm.x_centers_50nm, [mm.x_bins_50nm[-1]]])
    zz_for_evolver = np.concatenate([[zz_PMMA[0]], zz_PMMA, [zz_PMMA[-1]]])

    # plt.figure(dpi=300)
    # plt.plot(xx_for_evolver, zz_for_evolver)
    # plt.show()

    file_full_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/datafile_DEBER_2021_5.fe'
    commands_full_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/commands_5.txt'

    ef.create_datafile_latest(
        yy=xx_for_evolver * 1e-3,
        zz=zz_for_evolver * 1e-3,
        width=mm.ly * 1e-3,
        mobs=np.ones(len(xx_for_evolver)) * SE_mobility_arr[n_step],
        path=file_full_path
    )

    ef.run_evolver(file_full_path, commands_full_path)

    vlist_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_single_5.txt'
    zz_after_evolver = df.get_zz_after_evolver_50(vlist_path)

    zz_exp = (zz_after_evolver[0] + zz_after_evolver[-1]) / 2 - zz_vac_0p2 * (n_step + 1) / n_steps

    plt.figure(dpi=300)
    plt.plot(mm.x_centers_50nm, zz_PMMA, label='before SE')
    plt.plot(mm.x_centers_50nm, zz_after_evolver, label='after SE')
    # plt.plot(xx_vac_0p2, mm.d_PMMA - zz_vac_0p2 * (n_step + 1) / n_steps, label='experiment')
    plt.plot(xx_vac_0p2, zz_exp, label='experiment')

    plt.title('profiles ' + str(n_step))
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.legend()
    plt.grid()
    # plt.show()

    plt.savefig('notebooks/DEBER_simulation/podgon_monomer/' + str(n_step) +
                '_zz_vac_' + str(zip_length) + '_' + str(SE_mobility_arr[n_step]) + '.jpg')

    plt.close('all')

    zz_vac = mm.d_PMMA - zz_after_evolver

    np.save('notebooks/DEBER_simulation/podgon_monomer/' + str(n_step) +
            '_zz_vac_' + str(zip_length) + '_' + str(SE_mobility_arr[n_step]) + '.npy', zz_vac)


