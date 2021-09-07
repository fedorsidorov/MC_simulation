import importlib
import matplotlib.pyplot as plt
import numpy as np
import constants as const
from mapping import mapping_5um_900nm as mm
from functions import array_functions as af
from functions import MC_functions as mcf
from functions import e_matrix_functions as emf
from functions import scission_functions as sf
from functions import reflow_functions as rf
from functions import SE_functions as ef
from functions import e_beam_MC as eb_MC
from functions import diffusion_functions as df
from tqdm import tqdm
import indexes as ind

const = importlib.reload(const)
eb_MC = importlib.reload(eb_MC)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
mcf = importlib.reload(mcf)
mm = importlib.reload(mm)
af = importlib.reload(af)
df = importlib.reload(df)
ef = importlib.reload(ef)
sf = importlib.reload(sf)
rf = importlib.reload(rf)

# %%
T_C = 160
zip_length = 1000
step_time = 5

MMA_weight = 100

y_slice_V = mm.step_5nm * mm.ly * mm.step_5nm

y_0 = 3989

tau_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Boyd_kinetic_curves/arrays/tau.npy')
Mn_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Boyd_kinetic_curves/arrays/Mn_125.npy')*100
Mw_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Boyd_kinetic_curves/arrays/Mw_125.npy')*100

tau_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))
free_monomer_matrix_in_resist = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))


# %%
def get_resist_fraction_matrix(zz_vac):
    resist_fraction_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))

    for i in range(len(mm.x_centers_5nm)):
        beg_ind = np.where(mm.z_bins_5nm > now_zz_vac[i])[0][0]

        now_resist_fraction_matrix[i, beg_ind:] = 1

        if beg_ind > 0:
            now_resist_fraction_matrix[i, :beg_ind - 1] = 0
            trans_resist_fraction = 1 - (now_zz_vac[i] - mm.z_bins_5nm[beg_ind - 1]) / mm.step_5nm
            now_resist_fraction_matrix[i, beg_ind - 1] = trans_resist_fraction

    return resist_fraction_matrix


for n_step in range(34, 35):

    # get new scission matrix
    xx_vac_raw = np.load('notebooks/DEBER_simulation/vary_zz_vac_0p2_5s/xx_vac.npy')
    now_zz_vac_raw = np.load('notebooks/DEBER_simulation/vary_zz_vac_0p2_5s/zz_vac_' + str(n_step) + '.npy')
    now_scission_matrix = \
        np.load('notebooks/DEBER_simulation/vary_zz_vac_0p2_5s/scission_matrix_' + str(n_step) + '.npy')

    now_xx_vac = mm.x_centers_5nm
    now_zz_vac = mcf.lin_lin_interp(xx_vac_raw, now_zz_vac_raw)(now_xx_vac)

    # plt.figure(dpi=300)
    # plt.plot(now_xx_vac, now_zz_vac)
    # plt.show()

    # get resist fraction matrix
    now_resist_fraction_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))

    for i in range(len(mm.x_centers_5nm)):
        beg_ind = np.where(mm.z_bins_5nm > now_zz_vac[i])[0][0]

        now_resist_fraction_matrix[i, beg_ind:] = 1

        if beg_ind > 0:
            now_resist_fraction_matrix[i, :beg_ind - 1] = 0
            trans_resist_fraction = 1 - (now_zz_vac[i] - mm.z_bins_5nm[beg_ind - 1]) / mm.step_5nm
            now_resist_fraction_matrix[i, beg_ind - 1] = trans_resist_fraction
    # now_resist_fraction_matrix = get_resist_fraction_matrix(now_zz_vac)

    plt.figure(dpi=300)
    plt.imshow(now_resist_fraction_matrix.transpose())
    plt.show()

    break

    now_resist_monomer_matrix = now_resist_fraction_matrix * y_slice_V / const.V_mon_nm3

    free_monomer_matrix_in_resist = free_monomer_matrix_in_resist * now_resist_fraction_matrix

    # deal with Mn
    now_ks_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))
    now_Mn_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))
    now_true_Mn_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))
    now_eta_array = np.zeros(len(mm.x_centers_5nm))
    now_SE_mob = np.zeros(len(mm.x_centers_5nm))

    inds_mon = np.where(now_resist_monomer_matrix > 0)
    now_ks_matrix[inds_mon] = now_scission_matrix[inds_mon] / now_resist_monomer_matrix[inds_mon] / step_time

    now_tau_matrix = y_0 * now_ks_matrix * step_time

    tau_matrix += now_tau_matrix

    progress_bar = tqdm(total=len(mm.x_centers_5nm), position=0)

    for i in range(len(mm.x_centers_5nm)):
        for k in range(len(mm.z_centers_5nm)):

            if now_resist_fraction_matrix[i, k] == 0:
                continue

            if tau_matrix[i, k] > tau_125[-1]:
                now_Mn = Mn_125[-1]
            else:
                # now_Mn = mcf.lin_lin_interp(tau_125, Mn_125)(tau_matrix[i, k])
                tau_ind = np.argmin(np.abs(tau_matrix[i, k] - tau_125))
                now_Mn = Mn_125[tau_ind]

            now_Mn_matrix[i, k] = now_Mn

            now_n_chains = y_slice_V / const.V_mon_nm3 / (now_Mn / MMA_weight)

            now_true_Mn = (now_n_chains * now_Mn + free_monomer_matrix_in_resist[i, k] * MMA_weight) /\
                          (now_n_chains + free_monomer_matrix_in_resist[i, k])

            now_true_Mn_matrix[i, k] = now_true_Mn

        progress_bar.update()

    now_eta_array = np.zeros(len(mm.x_centers_5nm))
    now_SE_mob_array = np.zeros(len(mm.x_centers_5nm))

    for i in range(len(mm.x_centers_5nm)):

        inds = np.where(now_true_Mn_matrix[i, :] > 0)[0]
        now_Mn_avg = np.average(now_true_Mn_matrix[i, inds])
        now_eta = rf.get_viscosity_experiment_Mn(T_C, now_Mn_avg, 3.4)

        now_eta_array[i] = now_eta
        now_SE_mob_array[i] = rf.get_SE_mobility(now_eta)



    # now_n_free_monomers = free_monomer_matrix[i, k]
    #
    # if now_n_free_monomers == 0:
    #     continue
    #
    # now_n_chains = now_resist_monomer_matrix[i, k] / (Mn_0 / 1e+2)
    # now_true_Mn = now_resist_monomer_matrix[i, k] * 100 / (now_n_chains + now_n_free_monomers)
    # Mn_true_matrix[i, k] = now_true_Mn
    # eta_matrix[i, k] = rf.get_viscosity_experiment_Mn(T_C, now_true_Mn, 3.4)
    # SE_mob_matrix[i, k] = rf.get_SE_mobility(eta_matrix[i, k])

# %% simulate depolymerization
now_free_monomer_matrix_0 = now_scission_matrix * zip_length

# %% simulate diffusion
now_outer_monomer_array = np.zeros(len(mm.x_centers_5nm))

now_wp_matrix = np.zeros(np.shape(now_free_monomer_matrix_0))
now_D_matrix = np.zeros(np.shape(now_free_monomer_matrix_0))

progress_bar = tqdm(total=len(mm.x_centers_5nm), position=0)

now_out_monomer_array = np.zeros(len(mm.x_centers_5nm))

for i in range(len(mm.x_centers_5nm)):
    for k in range(len(mm.z_centers_5nm)):

        now_n_free_monomers = int(now_free_monomer_matrix_0[i, k] + free_monomer_matrix_in_resist[i, k])

        now_wp = 1 - now_n_free_monomers / now_resist_monomer_matrix[i, k]
        if now_wp <= 0:
            print('now_wp <= 0')
            now_wp = 0

        now_D = df.get_D(T_C, now_wp)

        now_wp_matrix[i, k] = now_wp
        now_D_matrix[i, k] = now_D * 1e-5


for i in range(len(mm.x_centers_5nm)):
    for k in range(len(mm.z_centers_5nm)):

        now_n_free_monomers = int(now_free_monomer_matrix_0[i, k] + free_monomer_matrix_in_resist[i, k])

        if now_n_free_monomers == 0:
            continue

        now_xx0_mon_arr = np.ones(now_n_free_monomers) * mm.x_centers_5nm[i]
        now_zz0_mon_arr = np.ones(now_n_free_monomers) * mm.z_centers_5nm[k]

        if now_resist_monomer_matrix[i, k] == 0:  # events doubling problems
            continue

        now_xx_mon = df.get_final_x_arr(x0_arr=now_xx0_mon_arr, D=now_D_matrix[i, k], delta_t=step_time)
        now_zz_mon = df.get_final_z_arr(z0_arr_raw=now_zz0_mon_arr, d_PMMA=900, D=now_D_matrix[i, k], delta_t=step_time)

        af.snake_coord_1d(array=now_xx_mon, coord_min=mm.x_bins_5nm[0], coord_max=mm.x_bins_5nm[-1])

        now_zz_mon_vac = mcf.lin_lin_interp(now_xx_vac_prec, now_zz_vac_prec)(now_xx_mon)
        now_weights = now_zz_mon > now_zz_mon_vac

        now_xx_zz_mon = np.vstack([now_xx_mon, now_zz_mon]).transpose()

        free_monomer_matrix_in_resist += np.histogramdd(
            sample=now_xx_zz_mon,
            bins=[mm.x_bins_5nm, mm.z_bins_5nm],
            weights=now_weights.astype(int)
        )[0]

        now_out_monomer_array += np.histogram(
            a=now_xx_mon,
            bins=mm.x_bins_5nm,
            weights=np.logical_not(now_weights).astype(int)
        )[0]

    progress_bar.update()

plt.figure(dpi=300)
plt.imshow(free_monomer_matrix_in_resist.transpose())
# plt.imshow(np.log(now_D_matrix.transpose()))
plt.show()
