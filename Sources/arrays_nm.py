import importlib
import numpy as np
from functions import MC_functions as util
import indexes as inds
import matplotlib.pyplot as plt
util = importlib.reload(util)
inds = importlib.reload(inds)


#%% grid
EE = np.logspace(0, 4.4, 1000)
EE_prec = np.logspace(-1, 4.4, 1000)

THETA_deg = np.linspace(0, 180, 1000)
THETA_rad = np.deg2rad(THETA_deg)

# %%
# elastic interactions - OK
PMMA_el_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/ELSEPA/PMMA/PMMA_muffin_EIMFP.npy') * 1e-7
PMMA_el_IMFP[:228] = PMMA_el_IMFP[228]  # no extrapolation
PMMA_el_DIMFP_norm = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/ELSEPA/PMMA/MMA_muffin_DEMFP_plane_norm.npy')

Si_el_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/ELSEPA/Si/Si_muffin_u.npy') * 1e-7
Si_el_IMFP[:inds.Si_E_cut_ind] = 0  # no life below plasmon energy!
Si_el_DIMFP_norm = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/ELSEPA/Si/Si_muffin_diff_cs_plane_norm.npy')

# electron-electron interaction
C_K_ee_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/GOS/C_GOS_IIMFP.npy') * 1e-7
C_K_ee_DIMFP_norm = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/GOS/C_GOS_DIIMFP_norm.npy')

O_K_ee_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/GOS/O_GOS_IIMFP.npy') * 1e-7
O_K_ee_DIMFP_norm = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/GOS/O_GOS_DIIMFP_norm.npy')

PMMA_val_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/Mermin/IIMFP_Mermin_PMMA.npy') * 1e-7
PMMA_val_DIMFP_norm = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/Mermin/DIIMFP_Mermin_PMMA_norm.npy')

PMMA_ee_DIMFP_norm_3 = np.array((PMMA_val_DIMFP_norm, C_K_ee_DIMFP_norm, O_K_ee_DIMFP_norm))

Si_ee_IMFP_6 = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/MuElec/Si_MuElec_IIMFP.npy') * 1e-7
Si_ee_DIMFP_norm_6 = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/MuElec/Si_MuElec_DIIMFP_norm.npy')

# electron-phonon interaction
PMMA_ph_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/PhPol/PMMA_phonon_IMFP.npy') * 1e-7
PMMA_pol_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/PhPol/PMMA_polaron_IMFP_0p1_0p15.npy') * 1e-7

# total IMFP
PMMA_IMFP = np.vstack((PMMA_el_IMFP, PMMA_val_IMFP, C_K_ee_IMFP, O_K_ee_IMFP,
                       PMMA_ph_IMFP, PMMA_pol_IMFP)).transpose()
PMMA_IMFP_norm = util.norm_2d_array(PMMA_IMFP)
PMMA_total_IMFP = np.sum(PMMA_IMFP, axis=1)

Si_IMFP = np.vstack((Si_el_IMFP, Si_ee_IMFP_6.transpose())).transpose()
Si_IMFP_norm = util.norm_2d_array(Si_IMFP)
Si_total_IMFP = np.sum(Si_IMFP, axis=1)