import numpy as np

# %% load PMMA curves
PMMA_el_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_PMMA/PMMA_el_IMFP_nm.npy')
PMMA_el_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_PMMA/'
                                  'PMMA_el_DIMFP_cumulated.npy')

PMMA_val_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_PMMA/'
                        'PMMA_val_IMFP_nm.npy')
PMMA_val_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_PMMA/'
                                   'PMMA_val_DIMFP_cumulated_corr.npy')

PMMA_C_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/'
                      'arrays_PMMA/PMMA_C_IMFP_E_bind_nm.npy')
PMMA_C_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_PMMA/'
                                 'PMMA_C_DIIMFP_cumulated_E_bind.npy')

PMMA_O_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/'
                      'arrays_PMMA/PMMA_O_IMFP_E_bind_nm.npy')
PMMA_O_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_PMMA/'
                                 'PMMA_O_DIIMFP_cumulated_E_bind.npy')

# phonon
PMMA_ph_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_PMMA/'
                       'PMMA_ph_IMFP_nm.npy')

# polaron
PMMA_pol_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/'
                        'arrays_PMMA/PMMA_pol_IMFP_nm.npy')

PMMA_IMFP =\
    np.vstack((PMMA_el_IMFP, PMMA_val_IMFP, PMMA_C_IMFP, PMMA_O_IMFP, PMMA_ph_IMFP, PMMA_pol_IMFP)).transpose()

# norm IMFP array
PMMA_IMFP_norm = np.zeros(np.shape(PMMA_IMFP))
for i in range(len(PMMA_IMFP)):
    if np.sum(PMMA_IMFP[i, :]) != 0:
        PMMA_IMFP_norm[i, :] = PMMA_IMFP[i, :] / np.sum(PMMA_IMFP[i, :])

PMMA_total_IMFP = np.sum(PMMA_IMFP, axis=1)
PMMA_process_indexes = list(range(len(PMMA_IMFP[0, :])))

PMMA_ee_DIMFP_cumulated = np.array([PMMA_val_DIMFP_cumulated, PMMA_C_DIMFP_cumulated, PMMA_O_DIMFP_cumulated])

# %% load simple_Si_MC curves
Si_el_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_Si/Si_el_IMFP_nm.npy')
Si_el_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_Si/'
                                'Si_el_DIMFP_cumulated.npy')

Si_ee_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_Si/' +
                     '5_osc/Si_ee_IMFP_5osc_nm.npy')
Si_ee_DIMFP_cumulated = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_Si/' +
                                '5_osc/Si_ee_DIMFP_cumulated_5osc.npy')

Si_IMFP = np.vstack((Si_el_IMFP, Si_ee_IMFP.transpose())).transpose()

# norm IMFP array
Si_IMFP_norm = np.zeros(np.shape(Si_IMFP))
for i in range(len(Si_IMFP)):
    if np.sum(Si_IMFP[i, :]) != 0:
        Si_IMFP_norm[i, :] = Si_IMFP[i, :] / np.sum(Si_IMFP[i, :])

Si_total_IMFP = np.sum(Si_IMFP, axis=1)
Si_process_indexes = list(range(len(Si_IMFP[0, :])))

# %% combine it all to structure curves
structure_el_IMFP = [PMMA_el_IMFP, Si_el_IMFP]
structure_el_DIMFP_cumulated = [PMMA_el_DIMFP_cumulated, Si_el_DIMFP_cumulated]

# structure_IMFP = [PMMA_IMFP, Si_IMFP]
structure_ee_DIMFP_cumulated = [PMMA_ee_DIMFP_cumulated, Si_ee_DIMFP_cumulated]

structure_total_IMFP = [PMMA_total_IMFP, Si_total_IMFP]
structure_IMFP_norm = [PMMA_IMFP_norm, Si_IMFP_norm]

structure_process_indexes = [PMMA_process_indexes, Si_process_indexes]






