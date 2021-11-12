import importlib
import matplotlib.pyplot as plt
import numpy as np
import constants as const
from mapping import mapping_5um_900nm as mm
from functions import MC_functions as mcf
from functions import DEBER_functions as df
from functions import SE_functions as ef
from functions import reflow_functions as rf

const = importlib.reload(const)
mm = importlib.reload(mm)
mcf = importlib.reload(mcf)
df = importlib.reload(df)
ef = importlib.reload(ef)
rf = importlib.reload(rf)

# %%
profile_0p2 = np.loadtxt('notebooks/DEBER_simulation/exp_curves/exp_900nm_5um_0.2uC_cm2.txt')

profile_0p2[0, 1] = profile_0p2[:, 1].max()
profile_0p2[-1, 1] = profile_0p2[:, 1].max()

xx_vac_0p2 = profile_0p2[:, 0] * 1000 - 6935.08
zz_vac_0p2 = (profile_0p2[:, 1].max() - profile_0p2[:, 1]) * 1000

# %%
xx_vac = mm.x_centers_50nm  # nm

zip_length = 4000

n_steps = 235
SE_mobility_mat = np.ones((len(mm.x_centers_50nm), n_steps))
zip_length_mat = np.ones((len(mm.x_centers_50nm), n_steps)) * zip_length

coeff = 100
cos_power = 25
SE_mob_factor = 25
# zip_length_factor = 18
zip_length_factor = 0

for j in range(n_steps):
    coeff_arr = (np.cos(xx_vac * 2 * np.pi / mm.lx) + 1) ** cos_power
    SE_mobility_mat[:, j] = SE_mobility_mat[:, j] * coeff_arr
    SE_mobility_mat[:, j] /= np.max(SE_mobility_mat[:, j])
    SE_mobility_mat[:, j] += 4.059264732063505e-06

    zip_length_mat[:, j] = zip_length + j * zip_length_factor

plt.figure(dpi=300)
# plt.plot(xx_vac, SE_mobility_mat[:, 2])

plt.semilogy(xx_vac, SE_mobility_mat[:, 0])

plt.xlabel('t, s')
plt.ylabel('SE mobility')

plt.legend()
# plt.xlim(0, 250)
plt.ylim(0, 1.2)
plt.grid()
plt.show()
# plt.savefig('_SE_mobility.jpg')

# %%
zz_vac = np.zeros(len(xx_vac))  # nm
now_n_step = 0

# for n_step in range(now_n_step, now_n_step + 100):
for n_step in range(n_steps):

    zip_length = zip_length_mat[0, n_step]

    now_scission_matrix = np.load('notebooks/DEBER_simulation/vary_zz_vac_0p2_1s/scission_matrix_'
                                  + str(n_step) + '.npy')

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

    now_mobs = SE_mobility_mat[:, n_step]
    mobs_for_evolver = np.concatenate([[now_mobs[0]], now_mobs, [now_mobs[-1]]])

    # plt.figure(dpi=300)
    # plt.plot(xx_for_evolver, zz_for_evolver)
    # plt.show()

    file_full_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/datafile_DEBER_2021_5.fe'
    commands_full_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/commands_5.txt'

    ef.create_datafile_latest(
        yy=xx_for_evolver * 1e-3,
        zz=zz_for_evolver * 1e-3,
        width=mm.ly * 1e-3,
        mobs=mobs_for_evolver,
        path=file_full_path
    )

    ef.run_evolver(file_full_path, commands_full_path)

    vlist_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_single_5.txt'
    zz_after_evolver = df.get_zz_after_evolver_50(vlist_path)

    zz_exp = (zz_after_evolver[0] + zz_after_evolver[-1]) / 2 - zz_vac_0p2 * (n_step + 1) / n_steps

    plt.figure(dpi=300)
    plt.plot(mm.x_centers_50nm, zz_PMMA, label='before SE')
    plt.plot(mm.x_centers_50nm, zz_after_evolver, label='after SE')
    plt.plot(xx_vac_0p2, zz_exp, label='experiment')

    plt.title('profiles ' + str(n_step))
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.legend()
    plt.grid()
    # plt.show()

    plt.savefig('notebooks/DEBER_simulation/podgon_monomer/conf_vary_mob_4000_25/' + str(n_step) + '_zz_vac.jpg')

    plt.close('all')

    zz_vac = mm.d_PMMA - zz_after_evolver

    np.save('notebooks/DEBER_simulation/podgon_monomer/conf_vary_mob_4000_25/' + str(n_step) + '_zz_vac.npy', zz_vac)

# %%
eta_arr = rf.get_eta(SE_mobility_mat[:, 0])

plt.figure(dpi=300)
plt.semilogy(mm.x_centers_50nm, eta_arr)
plt.show()

# %%
np.save('xx_eta_2.npy', mm.x_centers_50nm)
np.save('eta_2.npy', eta_arr)

