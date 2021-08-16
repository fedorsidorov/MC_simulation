import numpy as np
import matplotlib.pyplot as plt
import grid

# %% load PMMA curves
PMMA_el_u = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/PMMA/'
                    'PMMA_easy_u_nm.npy')

PMMA_el_diff_u_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/'
                                   'final_arrays/PMMA/PMMA_diff_cs_cumulated_easy_+1.npy')

PMMA_val_u = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Dapor_PMMA_Mermin/final_arrays/u_nm.npy')

PMMA_val_diff_u_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Dapor_PMMA_Mermin/'
                                    'final_arrays/diff_u_cumulated.npy')

# PMMA_C_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/'
#                       'arrays_PMMA/PMMA_C_IMFP_E_bind_nm.npy')
# PMMA_C_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_PMMA/'
#                                  'PMMA_C_DIIMFP_cumulated_E_bind.npy')

# PMMA_O_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/'
#                       'arrays_PMMA/PMMA_O_IMFP_E_bind_nm.npy')
# PMMA_O_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_PMMA/'
#                                  'PMMA_O_DIIMFP_cumulated_E_bind.npy')

# phonon
PMMA_ph_u = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_PMMA/'
                       'PMMA_ph_IMFP_nm.npy')

# polaron
PMMA_pol_u = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/'
                        'arrays_PMMA/PMMA_pol_IMFP_nm.npy')

PMMA_IMFP =\
    np.vstack((PMMA_el_u, PMMA_val_u, PMMA_ph_u, PMMA_pol_u)).transpose()

# norm IMFP array
PMMA_IMFP_norm = np.zeros(np.shape(PMMA_IMFP))
for i in range(len(PMMA_IMFP)):
    if np.sum(PMMA_IMFP[i, :]) != 0:
        PMMA_IMFP_norm[i, :] = PMMA_IMFP[i, :] / np.sum(PMMA_IMFP[i, :])

PMMA_total_IMFP = np.sum(PMMA_IMFP, axis=1)
PMMA_process_indexes = list(range(len(PMMA_IMFP[0, :])))

# PMMA_ee_DIMFP_cumulated = np.array([PMMA_val_DIMFP_cumulated, PMMA_C_DIMFP_cumulated, PMMA_O_DIMFP_cumulated])

plt.figure(dpi=300)
plt.loglog(grid.EE, PMMA_el_u)
plt.loglog(grid.EE, PMMA_val_u)
plt.loglog(grid.EE, PMMA_ph_u)
plt.loglog(grid.EE, PMMA_pol_u)

plt.ylim(1e-5, 1e+2)
plt.grid()
plt.show()





# %% load simple_Si_MC curves
# Si_el_IMFP =\
# np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_Si/Si_el_IMFP_nm.npy')
# Si_el_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_Si/'
#                                 'Si_el_DIMFP_cumulated.npy')
#
# Si_ee_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_Si/' +
#                      '5_osc/Si_ee_IMFP_5osc_nm.npy')
# Si_ee_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_Si/' +
#                                 '5_osc/Si_ee_DIMFP_cumulated_5osc.npy')
#
# Si_IMFP = np.vstack((Si_el_IMFP, Si_ee_IMFP.transpose())).transpose()
#
# # norm IMFP array
# Si_IMFP_norm = np.zeros(np.shape(Si_IMFP))
# for i in range(len(Si_IMFP)):
#     if np.sum(Si_IMFP[i, :]) != 0:
#         Si_IMFP_norm[i, :] = Si_IMFP[i, :] / np.sum(Si_IMFP[i, :])
#
# Si_total_IMFP = np.sum(Si_IMFP, axis=1)
# Si_process_indexes = list(range(len(Si_IMFP[0, :])))

# %% combine it all to structure curves
# structure_el_IMFP = [PMMA_el_IMFP, Si_el_IMFP]
# structure_el_DIMFP_cumulated = [PMMA_el_DIMFP_cumulated, Si_el_DIMFP_cumulated]
#
# # structure_IMFP = [PMMA_IMFP, Si_IMFP]
# structure_ee_DIMFP_cumulated = [PMMA_ee_DIMFP_cumulated, Si_ee_DIMFP_cumulated]
#
# structure_total_IMFP = [PMMA_total_IMFP, Si_total_IMFP]
# structure_IMFP_norm = [PMMA_IMFP_norm, Si_IMFP_norm]
#
# structure_process_indexes = [PMMA_process_indexes, Si_process_indexes]
#





