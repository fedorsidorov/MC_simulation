import importlib

import numpy as np

from SimClasses import utilities as u, constants as c

c = importlib.reload(c)
u = importlib.reload(u)


#%%
# elastic interactions - OK
PMMA_el_IMFP = np.load('Resources/ELSEPA/PMMA/PMMA_muffin_EIMFP.npy')
PMMA_el_IMFP[:228] = PMMA_el_IMFP[228]  # no extrapolation
PMMA_el_DIMFP_norm = np.load('Resources/ELSEPA/PMMA/MMA_muffin_DEMFP_plane_norm.npy')

Si_el_IMFP = np.load('Resources/ELSEPA/Si/Si_muffin_u.npy')
Si_el_IMFP[:278] = 0  # no life below plasmon energy!
Si_el_DIMFP_norm = np.load('Resources/ELSEPA/Si/Si_muffin_diff_cs_plane_norm.npy')

# electron-electron interaction
C_K_ee_IMFP = np.load('Resources/GOS/C_GOS_IIMFP.npy')
C_K_ee_DIMFP_norm = np.load('Resources/GOS/C_GOS_DIIMFP_norm.npy')  # C - OK

O_K_ee_IMFP = np.load('Resources/GOS/O_GOS_IIMFP.npy')
O_K_ee_DIMFP_norm = np.load('Resources/GOS/O_GOS_DIIMFP_norm.npy')  # O - OK

PMMA_val_IMFP = np.load('Resources/Mermin/IIMFP_Mermin_PMMA.npy')
PMMA_val_DIMFP_norm = np.load('Resources/Mermin/DIIMFP_Mermin_PMMA_norm.npy')  # valence - OK

PMMA_ee_DIMFP_norm_3 = np.array((PMMA_val_DIMFP_norm, C_K_ee_DIMFP_norm, O_K_ee_DIMFP_norm))

Si_ee_IMFP_6 = np.load('Resources/MuElec/Si_MuElec_IIMFP.npy')*1e-18*c.n_Si
Si_ee_DIMFP_norm_6 = np.load('Resources/MuElec/Si_MuElec_DIIMFP_norm.npy')

# electron-phonon interaction
PMMA_ph_IMFP = np.load('Resources/PhPol/PMMA_phonon_IMFP.npy')
PMMA_pol_IMFP = np.load('Resources/PhPol/PMMA_polaron_IMFP_0p1_0p15.npy')

# total IMFP
PMMA_IMFP = np.vstack((PMMA_el_IMFP, PMMA_val_IMFP, C_K_ee_IMFP, O_K_ee_IMFP,
                       PMMA_ph_IMFP, PMMA_pol_IMFP)).transpose()
PMMA_IMFP_norm = u.norm_2d_array(PMMA_IMFP)
PMMA_total_IMFP = np.sum(PMMA_IMFP, axis=1)

Si_IMFP = np.vstack((Si_el_IMFP, Si_ee_IMFP_6.transpose())).transpose()
Si_IMFP_norm = u.norm_2d_array(Si_IMFP)
Si_total_IMFP = np.sum(Si_IMFP, axis=1)

# structure lists
structure_IMFP_norm = [PMMA_IMFP_norm, Si_IMFP_norm]  # !!!!!!!!!!!!!!!
structure_total_IMFP = [PMMA_total_IMFP, Si_total_IMFP]
structure_elastic_DIMFP = [PMMA_el_DIMFP_norm, Si_el_DIMFP_norm]
structure_ee_DIMFP_norm = [PMMA_ee_DIMFP_norm_3, Si_ee_DIMFP_norm_6]

# cutoff energies
structure_cutoff_E = [1, 16.7]

# binding energies
structure_E_bind = [
    [c.val_E_bind_PMMA, c.K_Ebind_C, c.K_Ebind_O],  # dapor2017.pdf
    c.Si_MuElec_E_bind  # MuElec Geant4 library
]

# indexes
structure_proc_inds = [
    [i for i in range(len(structure_IMFP_norm[0][0]))],
    [i for i in range(len(structure_IMFP_norm[1][0]))]
]
