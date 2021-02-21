import importlib
import numpy as np

from functions import MC_functions as mcf
import indexes as inds
import grid

import matplotlib.pyplot as plt
mcf = importlib.reload(mcf)
inds = importlib.reload(inds)
grid = importlib.reload(grid)

# %% elastic
PMMA_el_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/ELSEPA/PMMA/PMMA_muffin_EIMFP.npy') * 1e-7
# PMMA_el_IMFP[:228] = PMMA_el_IMFP[228]  # no extrapolation
PMMA_el_DIMFP_cumulated = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/ELSEPA/PMMA/MMA_diff_cs_cumulated_muffin.npy')

# plt.figure(dpi=300)
# plt.loglog(grid.EE, PMMA_el_IMFP)
# plt.show()

# plt.figure(dpi=300)
# for i in range(1, 1000, 100):
#     plt.semilogx(grid.EE, PMMA_el_DIMFP_norm[i, :])
# plt.show()

# %%
Si_el_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/ELSEPA/Si/Si_muffin_u.npy') * 1e-7
Si_el_IMFP[:inds.Si_E_cut_ind] = 0  # no life below plasmon energy!
Si_el_DIMFP_cumulated = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/ELSEPA/Si/Si_diff_cs_cumulated_muffin.npy')

# Si_el_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/MuElec/elastic_u.npy') * 1e-7
# Si_el_IMFP[:inds.Si_E_cut_ind] = 0  # no life below plasmon energy!

# Si_el_DIMFP_norm = np.load('Resources/MuElec/elastic_diff_sigma_sin_norm.npy')

# plt.figure(dpi=300)
# plt.loglog(grid.EE, Si_el_IMFP_ELSEPA)
# plt.loglog(grid.EE, Si_el_IMFP)
# plt.show()

# plt.figure(dpi=300)
# for i in range(1, 1000, 100):
#     plt.semilogx(grid.EE, Si_el_DIMFP_norm[i, :])
# plt.show()

#%% PMMA e-e
C_K_ee_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/GOS/C_GOS_IIMFP.npy') * 1e-7
C_K_ee_DIMFP_cumulated = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/GOS/C_GOS_DIIMFP_cumulated.npy')

O_K_ee_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/GOS/O_GOS_IIMFP.npy') * 1e-7
O_K_ee_DIMFP_cumulated = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/GOS/O_GOS_DIIMFP_cumulated.npy')

PMMA_val_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/Mermin/IIMFP_Mermin_PMMA.npy') * 1e-7
PMMA_val_DIMFP_cumulated = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/Mermin/DIIMFP_Mermin_PMMA_cumulated.npy')

# plt.figure(dpi=300)
# plt.loglog(grid.EE, PMMA_val_IMFP)
# plt.loglog(grid.EE, C_K_ee_IMFP + O_K_ee_IMFP + PMMA_val_IMFP)
# plt.show()

# plt.figure(dpi=300)
# for i in range(1, 1000, 100):
#     plt.loglog(grid.EE, PMMA_val_DIMFP_norm[i, :])
# plt.show()

PMMA_ee_DIMFP_3_cumulated = np.array((PMMA_val_DIMFP_cumulated, C_K_ee_DIMFP_cumulated, O_K_ee_DIMFP_cumulated))

# %% Si e-e
Si_ee_IMFP_6 = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/MuElec/Si_MuElec_IIMFP.npy') * 1e-7
Si_ee_DIMFP_6_cumulated = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/MuElec/diff_sigma_6_cumulated.npy')

# %% PMMA phonon, polaron
PMMA_ph_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/PhPol/PMMA_phonon_IMFP.npy') * 1e-7

# electron-polaron interaction
PMMA_pol_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/PhPol/PMMA_polaron_IMFP_0p1_0p15.npy') * 1e-7
# PMMA_pol_IMFP = \
#     np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/PhPol/PMMA_polaron_IMFP_1p5_0p14.npy') * 1e-7
# PMMA_pol_IMFP = \
#     np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/PhPol/PMMA_polaron_IMFP_0p07_0p1.npy') * 1e-7

# plt.figure(dpi=300)
# plt.semilogx(grid.EE, PMMA_ph_IMFP)
# plt.semilogx(grid.EE, PMMA_pol_IMFP)
# plt.show()

# %% total IMFP
PMMA_IMFP = np.vstack((PMMA_el_IMFP, PMMA_val_IMFP, C_K_ee_IMFP, O_K_ee_IMFP,
                       PMMA_ph_IMFP, PMMA_pol_IMFP)).transpose()
PMMA_IMFP_norm = mcf.norm_2d_array(PMMA_IMFP)
PMMA_total_IMFP = np.sum(PMMA_IMFP, axis=1)

Si_IMFP = np.vstack((Si_el_IMFP, Si_ee_IMFP_6.transpose())).transpose()
Si_IMFP_norm = mcf.norm_2d_array(Si_IMFP)
Si_total_IMFP = np.sum(Si_IMFP, axis=1)
