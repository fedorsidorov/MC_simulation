import importlib

import matplotlib.pyplot as plt
import numpy as np

import grid as grid
import indexes as inds
from functions import MC_functions as util

grid = importlib.reload(grid)
inds = importlib.reload(inds)
util = importlib.reload(util)

# %%
# elastic interactions - OK
PMMA_el_IMFP = np.load('Resources/ELSEPA/PMMA/PMMA_muffin_EIMFP.npy')
PMMA_el_IMFP[:228] = PMMA_el_IMFP[228]  # no extrapolation
PMMA_el_DIMFP_norm = np.load('Resources/ELSEPA/PMMA/MMA_muffin_DEMFP_plane_norm.npy')

Si_el_IMFP = np.load('Resources/ELSEPA/Si/Si_muffin_u.npy')
Si_el_IMFP[:inds.Si_E_cut_ind] = 0  # no life below plasmon energy!
Si_el_DIMFP_norm = np.load('Resources/ELSEPA/Si/Si_muffin_diff_cs_plane_norm.npy')

# electron-electron interaction
C_K_ee_IMFP = np.load('Resources/GOS/C_GOS_IIMFP.npy')
C_K_ee_DIMFP_norm = np.load('Resources/GOS/C_GOS_DIIMFP_norm.npy')  # C - OK

O_K_ee_IMFP = np.load('Resources/GOS/O_GOS_IIMFP.npy')
O_K_ee_DIMFP_norm = np.load('Resources/GOS/O_GOS_DIIMFP_norm.npy')  # O - OK

PMMA_val_IMFP = np.load('Resources/Mermin/IIMFP_Mermin_PMMA.npy')
PMMA_val_DIMFP_norm = np.load('Resources/Mermin/DIIMFP_Mermin_PMMA_norm.npy')  # valence - OK

PMMA_ee_DIMFP_norm_3 = np.array((PMMA_val_DIMFP_norm, C_K_ee_DIMFP_norm, O_K_ee_DIMFP_norm))

Si_ee_IMFP_6 = np.load('Resources/MuElec/Si_MuElec_IIMFP.npy')
Si_ee_DIMFP_norm_6 = np.load('Resources/MuElec/Si_MuElec_DIIMFP_norm.npy')

# electron-phonon interaction
PMMA_ph_IMFP = np.load('Resources/PhPol/PMMA_phonon_IMFP.npy')
PMMA_pol_IMFP = np.load('Resources/PhPol/PMMA_polaron_IMFP_0p1_0p15.npy')

# total IMFP
PMMA_IMFP = np.vstack((PMMA_el_IMFP, PMMA_val_IMFP, C_K_ee_IMFP, O_K_ee_IMFP,
                       PMMA_ph_IMFP, PMMA_pol_IMFP)).transpose()
PMMA_IMFP_norm = util.norm_2d_array(PMMA_IMFP)
PMMA_total_IMFP = np.sum(PMMA_IMFP, axis=1)

Si_IMFP = np.vstack((Si_el_IMFP, Si_ee_IMFP_6.transpose())).transpose()
Si_IMFP_norm = util.norm_2d_array(Si_IMFP)
Si_total_IMFP = np.sum(Si_IMFP, axis=1)

# %%
_, _ = plt.subplots(dpi=600)
fig = plt.gcf()
fig.set_size_inches(3, 3)

font_size = 8

plt.loglog(grid.EE, PMMA_el_IMFP, label='упр. расс.')
plt.loglog(grid.EE, PMMA_val_IMFP, label='вал. эл.')
plt.loglog(grid.EE, C_K_ee_IMFP, label='внутр. эл. C')
plt.loglog(grid.EE, O_K_ee_IMFP, label='внутр. эл. O')
plt.loglog(grid.EE, PMMA_ph_IMFP, label='расс. на фон.')
plt.loglog(grid.EE, PMMA_pol_IMFP, label='ген. полярона')

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

plt.xlabel(r'$E$, эВ', fontsize=font_size)
plt.ylabel(r'$\mu$, cm$^{-1}$', fontsize=font_size)

plt.legend(fontsize=6, loc='upper right')
plt.xlim(2, 2e+4)
plt.ylim(1e+2, 1e+9)
plt.grid()
# plt.show()

# %%
plt.savefig('IMFP_PMMA.tiff', bbox_inches='tight', dpi=600)

