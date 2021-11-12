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

# %%
zz_vac = np.zeros(len(xx_vac))  # nm
now_n_step = 0

# for n_step in range(now_n_step, now_n_step + 100):
for n_step in range(n_steps):

    now_scission_matrix = np.load('notebooks/DEBER_simulation/vary_zz_vac_0p2_1s/scission_matrix_'
                                  + str(n_step) + '.npy')

    now_monomers_array = np.sum(now_scission_matrix, axis=1) * zip_length

    now_monomers_array_50 = np.histogram(
        a=mm.x_centers_5nm,
        bins=mm.x_bins_50nm,
        weights=now_monomers_array
    )[0]

    zz_vac += now_monomers_array_50 * const.V_mon_nm3 / mm.step_50nm / mm.ly
    zz_PMMA = mm.d_PMMA - zz_vac

    xx_for_evolver = np.concatenate([[mm.x_bins_50nm[0]], mm.x_centers_50nm, [mm.x_bins_50nm[-1]]])
    zz_for_evolver = np.concatenate([[zz_PMMA[0]], zz_PMMA, [zz_PMMA[-1]]])

    global_true_Mn_matrix = np.ones((len(mm.x_centers_50nm), len(mm.z_centers_50nm)))

    # now_eta_array, now_SE_mob_array = df.get_eta_SE_mob_arrays(
    #     true_Mn_matrix=global_true_Mn_matrix,
    #     temp_C=160,
    #     viscosity_power=3.4
    # )

    eta = rf.get_viscosity_experiment_Mn(T_C=160, Mn=251000, power=3.4)

    eta_array = np.ones(len(mm.x_centers_50nm)) * eta
    SE_mob_array = np.ones(len(mm.x_centers_50nm)) * rf.get_SE_mobility(eta)

    now_mobs = SE_mob_array
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

    plt.savefig('notebooks/DEBER_simulation/podgon_monomer/conf_const_mob_4000/' + str(n_step) + '_zz_vac.jpg')

    plt.close('all')

    zz_vac = mm.d_PMMA - zz_after_evolver

    np.save('notebooks/DEBER_simulation/podgon_monomer/conf_const_mob_4000/' + str(n_step) + '_zz_vac.npy', zz_vac)


# %%
# zz_0p2 = np.load('notebooks/DEBER_simulation/podgon_monomer/conf_const_mob_4000/234_zz_vac.npy')
zz_0p2 = np.load('notebooks/DEBER_simulation/podgon_monomer/conf_vary_mob_4000_25/234_zz_vac.npy')

zz_exp = (2 * mm.d_PMMA - zz_0p2[0] - zz_0p2[-1]) / 2 - zz_vac_0p2

fontsize = 10

_, ax = plt.subplots(dpi=300)
fig = plt.gcf()
fig.set_size_inches(4, 3)

# plt.figure(dpi=300)
plt.plot(xx_vac_0p2, zz_exp, label='experiment')
plt.plot(mm.x_centers_50nm, mm.d_PMMA - zz_0p2, label='simulation')

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)

plt.xlabel('x, nm', fontsize=fontsize)
plt.ylabel('z, nm', fontsize=fontsize)
plt.legend(fontsize=fontsize)

plt.xlim(-2500, 2500)
# plt.xlim(-3000, 3000)
plt.ylim(0, 800)

plt.grid()
# plt.show()
plt.savefig('pr_2.jpg', bbox_inches='tight')
