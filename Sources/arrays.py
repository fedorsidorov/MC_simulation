import importlib
import numpy as np
import matplotlib.pyplot as plt

from functions import MC_functions as mcf
import constants as const
import indexes as inds
import grid

const = importlib.reload(const)
grid = importlib.reload(grid)
inds = importlib.reload(inds)
mcf = importlib.reload(mcf)

# %% elastic
PMMA_el_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/ELSEPA/PMMA/MMA_muffin_u.npy') * 1e-7
PMMA_el_IMFP[:228] = PMMA_el_IMFP[228]  # no extrapolation
# PMMA_el_DIMFP_cumulated = \
#     np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/ELSEPA/PMMA/PMMA_el_DIMFP_cumulated.npy')

elastic_kind = 'muffin'
PMMA_el_IMFP_new = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/PMMA/PMMA_'
                       + elastic_kind + '_u.npy') * 1e-7

plt.figure(dpi=300)
plt.loglog(grid.EE, PMMA_el_IMFP)
plt.loglog(grid.EE, PMMA_el_IMFP_new)
plt.show()

# plt.figure(dpi=300)
# for i in range(1, 1000, 100):
#     plt.semilogx(grid.EE, PMMA_el_DIMFP_norm[i, :])
# plt.show()

# %%
Si_el_IMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/MuElec/MuElec_elastic_arrays/u.npy') * 1e-7
Si_el_DIMFP_cumulated = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/MuElec/MuElec_elastic_arrays/u_diff_cumulated.npy')

Si_el_IMFP[:inds.Si_E_cut_ind] = 0  # no life below plasmon energy!
Si_el_DIMFP_cumulated[:inds.Si_E_cut_ind, :] = 0  # no life below plasmon energy!

# plt.figure(dpi=300)
# plt.loglog(grid.EE, Si_el_IMFP)
# plt.grid()
# plt.show()

# plt.figure(dpi=300)
# for i in range(1, 1000, 20):
#     plt.semilogx(grid.THETA_rad, Si_el_DIMFP_cumulated[i, :])
# plt.show()

#%% PMMA e-e
C_K_ee_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/GOS/C_IIMFP.npy') * 1e-7
C_K_ee_DIMFP_cumulated = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/GOS/C_DIIMFP_cumulated.npy')

O_K_ee_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/GOS/O_IIMFP.npy') * 1e-7
O_K_ee_DIMFP_cumulated = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/GOS/O_DIIMFP_cumulated.npy')

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

# %% simple_Si_MC e-e
Si_ee_IMFP_6 = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/MuElec/MuElec_inelastic_arrays/u_ee_6.npy') * 1e-7
Si_ee_DIMFP_6_cumulated = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/MuElec/' +
            'MuElec_inelastic_arrays/u_diff_cumulated_6.npy')

Si_E_bind = [16.65, 6.52, 13.63, 107.98, 151.55, 1828.5]

# Si_ee_IMFP_5 = np.zeros([len(grid.EE), 5])
# for n in range(5):
#     Si_ee_IMFP_5[:, n] = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/OLF_Si/IIMFP_5osc/u_' +
#                                  str(n) + '.npy') * 1e-7

# Si_ee_IMFP_5[:inds.Si_E_cut_ind] = 0  # no life below plasmon energy!

# Si_ee_DIMFP_5_cumulated = np.zeros([5, len(grid.EE), len(grid.EE)])
# for n in range(5):
#     Si_ee_DIMFP_5_cumulated[n, :, :] =\
#         np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/OLF_Si/DIIMFP_5osc_cumulated/DIIMFP_' +
#                 str(n) + '_cumulated.npy')

# Si_ee_DIMFP_5_cumulated[:, inds.Si_E_cut_ind, :] = 0  # no life below plasmon energy!

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
# Si_IMFP = np.vstack((Si_el_IMFP, Si_ee_IMFP_5.transpose())).transpose()
Si_IMFP_norm = mcf.norm_2d_array(Si_IMFP)
Si_total_IMFP = np.sum(Si_IMFP, axis=1)
