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
dose_s = 0.05e-6  # C / cm^2
# dose_s = 0.2e-6  # C / cm^2
# dose_s = 0.87e-6  # C / cm^2

dose_l = dose_s * mm.lx_cm   # C / cm
total_time = dose_l / mm.j_exp_l  # 1024 s
n_electrons = dose_s * mm.area_cm2 / const.e_SI  # 54 301
n_electrons_s = int(np.around(n_electrons / total_time))

step_time = 2
n_steps = int(total_time / step_time)
n_electrons_in_file = n_electrons_s * step_time

scission_weight = 0.092  # 160 C

viscosity_power = 3.4

E0 = 20e+3
T_C = 160

# PARAMETERS !!!
zip_length = 3000
k_diff = int(1e+4)
Mn_diff = int(3e+3)

# %%
global_tau_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)))
global_free_monomer_in_resist_matrix = np.zeros((len(mm.x_centers_5nm), len(mm.z_centers_5nm)), dtype=int)
global_Mn_matrix = np.ones((len(mm.x_centers_5nm), len(mm.z_centers_5nm))) * df.Mn_0
global_true_Mn_matrix = np.ones((len(mm.x_centers_5nm), len(mm.z_centers_5nm))) * df.Mn_0

global_outer_monomer_array = np.zeros(len(mm.x_centers_5nm))

xx_vac = mm.x_centers_5nm  # nm
zz_vac = np.ones(len(xx_vac)) * 0  # nm
zz_vac_list = [zz_vac]

total_tau_array = np.zeros(len(mm.x_centers_5nm))

# n_step = 0
for n_step in range(n_steps):

    print('Step #' + str(n_step))

    # get scission_matrix
    now_scission_matrix = df.get_scission_matrix(
        n_electrons_in_file=n_electrons_in_file,
        E0=E0,
        scission_weight=scission_weight,
        xx_vac=xx_vac,
        zz_vac=zz_vac
    )

    # %
    # get resist fraction matrix
    now_resist_fraction_matrix = df.get_resist_fraction_matrix(zz_vac)
    now_resist_monomer_matrix = now_resist_fraction_matrix * mm.y_slice_V_nm3 / const.V_mon_nm3

    plt.figure(dpi=300)
    plt.imshow(now_resist_fraction_matrix.transpose())
    plt.colorbar()
    plt.title('resist fraction matrix ' + str(n_step))
    # plt.show()
    plt.savefig('notebooks/DEBER_simulation/Mn_D/' + str(zip_length) + '_' + str(k_diff) + '_' + str(Mn_diff) + '/' +
                str(n_step) + '_1_resisit_fraction.jpg')

    # correct global_free_monomers_in_resist matrix
    global_free_monomer_in_resist_matrix_corr = \
        (global_free_monomer_in_resist_matrix * now_resist_fraction_matrix).astype(int)

    # add monomers that are cut by new PMMA surface
    global_outer_monomer_array +=\
        np.sum(global_free_monomer_in_resist_matrix - global_free_monomer_in_resist_matrix_corr, axis=1)

    global_free_monomer_in_resist_matrix = global_free_monomer_in_resist_matrix_corr

    # simulate depolymerization
    # constant zip length
    now_free_monomer_matrix_0 = (now_scission_matrix * zip_length * now_resist_fraction_matrix).astype(int)

    # non-uniform zip length
    # now_free_monomer_matrix_0 =\
    #     (now_scission_matrix * global_Mn_matrix / const.MMA_weight * now_resist_fraction_matrix).astype(int)

    global_free_monomer_in_resist_matrix += now_free_monomer_matrix_0

    plt.figure(dpi=300)
    plt.imshow(global_free_monomer_in_resist_matrix.transpose())
    plt.colorbar()
    plt.title('monomers in resist BEFORE diffusion ' + str(n_step))
    # plt.show()
    plt.savefig('notebooks/DEBER_simulation/Mn_D/' + str(zip_length) + '_' + str(k_diff) + '_' + str(Mn_diff) + '/' +
                str(n_step) + '_2_n_free_mon_BEFORE.jpg')

    # update tau_matrix
    now_delta_tau_matrix = df.get_delta_tau_matix(now_resist_monomer_matrix, now_scission_matrix, step_time)
    global_tau_matrix += now_delta_tau_matrix

    # get Mn and true_Mn matrices
    global_Mn_matrix, global_true_Mn_matrix = df.get_Mn_true_Mn_matrix(
        resist_fraction_matrix=now_resist_fraction_matrix,
        free_monomer_in_resist_matrix=global_free_monomer_in_resist_matrix,
        tau_matrix=global_tau_matrix
    )

    # get eta and SE_mob arrays
    now_eta_array, now_SE_mob_array = df.get_eta_SE_mob_arrays(
        true_Mn_matrix=global_Mn_matrix,
        # true_Mn_matrix=global_true_Mn_matrix,
        temp_C=T_C,
        viscosity_power=viscosity_power
    )

    now_SE_mob_array_corr = np.copy(now_SE_mob_array)
    # now_SE_mob_array_corr[460:540] = (now_SE_mob_array_corr[459] + now_SE_mob_array_corr[540]) / 2
    now_SE_mob_array_corr[np.where(now_SE_mob_array_corr > 1)] = 1

    plt.figure(dpi=300)
    plt.imshow(np.log10(global_Mn_matrix).transpose())
    plt.colorbar()
    plt.title('lg(Mn) ' + str(n_step))
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig('notebooks/DEBER_simulation/Mn_D/' + str(zip_length) + '_' + str(k_diff) + '_' + str(Mn_diff) + '/' +
                str(n_step) + '_3a_Mn_matrix.jpg')

    plt.figure(dpi=300)
    plt.semilogy(mm.x_centers_5nm, now_SE_mob_array, label='original')
    plt.semilogy(mm.x_centers_5nm, now_SE_mob_array_corr, label='cut')
    plt.title('SE mobility ' + str(n_step))
    plt.xlabel('x, nm')
    plt.ylabel('SE mobility')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig('notebooks/DEBER_simulation/Mn_D/' + str(zip_length) + '_' + str(k_diff) + '_' + str(Mn_diff) + '/' +
                str(n_step) + '_3b_SE_mobility.jpg')

    # %
    # get wp, D and true_D matrices
    now_wp_matrix, now_D_matrix = df.get_wp_D_matrix(
        global_free_monomer_in_resist_matrix=global_free_monomer_in_resist_matrix,
        resist_monomer_matrix=now_resist_monomer_matrix,
        temp_C=T_C
    )

    now_true_D_matrix = df.get_true_D_matrix(
        D_matrix=now_D_matrix,
        Mn_matrix=global_Mn_matrix,
        k_diff=k_diff,
        Mn_diff=Mn_diff
    )

    plt.figure(dpi=300)
    plt.imshow(np.log10(now_true_D_matrix).transpose())
    plt.colorbar()
    plt.title('lg(true D) ' + str(n_step))
    # plt.show()
    plt.savefig('notebooks/DEBER_simulation/Mn_D/' + str(zip_length) + '_' + str(k_diff) + '_' + str(Mn_diff) + '/' +
                str(n_step) + '_4_true_D_matrix.jpg')

    # %
    # simulate diffusion
    now_free_monomer_in_resist_matrix, now_outer_monomer_array = df.get_free_mon_matrix_mon_out_array_after_diffusion(
        global_free_monomer_in_resist_matrix=global_free_monomer_in_resist_matrix,
        # D_matrix=now_D_matrix,
        D_matrix=now_true_D_matrix,
        xx_vac=xx_vac,
        zz_vac=zz_vac,
        d_PMMA=mm.d_PMMA,
        step_time=step_time
    )

    plt.figure(dpi=300)
    plt.plot(mm.x_centers_5nm, now_outer_monomer_array * const.V_mon_nm3 / mm.step_5nm / mm.ly)
    plt.title('now z out ' + str(n_step))
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.grid()
    # plt.show()
    plt.savefig('notebooks/DEBER_simulation/Mn_D/' + str(zip_length) + '_' + str(k_diff) + '_' + str(Mn_diff) + '/' +
                str(n_step) + '_5_z_out.jpg')

    global_free_monomer_in_resist_matrix += now_free_monomer_in_resist_matrix.astype(int)
    global_outer_monomer_array += now_outer_monomer_array

    global_free_monomer_in_resist_matrix -= now_free_monomer_matrix_0

    plt.figure(dpi=300)
    plt.imshow(global_free_monomer_in_resist_matrix.transpose())
    plt.colorbar()
    plt.title('monomers in resist AFTER diffusion ' + str(n_step))
    # plt.show()
    plt.savefig('notebooks/DEBER_simulation/Mn_D/' + str(zip_length) + '_' + str(k_diff) + '_' + str(Mn_diff) + '/' +
                str(n_step) + '_6_n_free_mon_AFTER.jpg')

    if (np.sum(now_free_monomer_in_resist_matrix) + np.sum(now_outer_monomer_array)) /\
       np.sum(now_free_monomer_matrix_0) != 1:
        print('some monomers are lost!')

    new_zz_vac = global_outer_monomer_array * const.V_mon_nm3 / mm.step_5nm / mm.ly

    # %
    zz_PMMA = mm.d_PMMA - new_zz_vac

    if np.all(zz_PMMA == mm.d_PMMA):
        zz_PMMA[500] -= 0.1

    xx_for_evolver_pre = np.concatenate([[mm.x_bins_5nm[0]], mm.x_centers_5nm, [mm.x_bins_5nm[-1]]])
    zz_for_evolver_pre = np.concatenate([[zz_PMMA[0]], zz_PMMA, [zz_PMMA[-1]]])
    xx_for_evolver = mm.x_bins_20nm
    zz_for_evolver = mcf.lin_lin_interp(xx_for_evolver_pre, zz_for_evolver_pre)(xx_for_evolver)

    # mobs_for_evolver = np.concatenate([[now_SE_mob_array[0]], now_SE_mob_array, [now_SE_mob_array[-1]]])
    mobs_for_evolver = np.concatenate([[now_SE_mob_array_corr[0]], now_SE_mob_array_corr, [now_SE_mob_array_corr[-1]]])

    file_full_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/datafile_DEBER_2021_5.fe'
    commands_full_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/commands_5.txt'

    # go to um and 20nm bin !!!
    ef.create_datafile_latest(
        yy=xx_for_evolver * 1e-3,
        zz=zz_for_evolver * 1e-3,
        width=mm.ly * 1e-3,
        mobs=mobs_for_evolver,
        path=file_full_path
    )

    # %
    ef.run_evolver(file_full_path, commands_full_path)

    vlist_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_single_5.txt'
    zz_after_evolver = df.get_zz_after_evolver(vlist_path)

    plt.figure(dpi=300)
    plt.plot(mm.x_centers_5nm, mm.d_PMMA - zz_vac, label='previous step')
    plt.plot(mm.x_centers_5nm, zz_PMMA, label='before SE')
    plt.plot(mm.x_centers_5nm, zz_after_evolver, label='after SE')

    plt.title('profiles ' + str(n_step))
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig('notebooks/DEBER_simulation/Mn_D/' + str(zip_length) + '_' + str(k_diff) + '_' + str(Mn_diff) + '/' +
                str(n_step) + '_7_profiles.jpg')

    zz_vac_final = mm.d_PMMA - zz_after_evolver
    zz_vac_list.append(zz_vac_final)
    zz_vac = zz_vac_final

    np.save('notebooks/DEBER_simulation/Mn_D/' + str(zip_length) + '_' + str(k_diff) + '_' + str(Mn_diff) + '/' +
            str(n_step) + '_zz_vac_final.npy', zz_vac_final)

    plt.close('all')





