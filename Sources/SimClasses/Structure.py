import importlib

import numpy as np
from numpy import random

import constants as c
import grid as g
import utilities as u

c = importlib.reload(c)
g = importlib.reload(g)
u = importlib.reload(u)


class Structure:
    # elastic interactions
    PMMA_el_IMFP = np.load('Resources/ELSEPA/PMMA/PMMA_muffin_EIMFP.npy')
    PMMA_el_DIMFP_norm = np.load('Resources/ELSEPA/PMMA/MMA_muffin_DEMFP_plane_norm.npy')

    Si_el_IMFP = np.load('Resources/ELSEPA/Si/Si_muffin_u.npy')
    Si_el_DIMFP_norm = np.load('Resources/ELSEPA/Si/Si_muffin_diff_cs_plane_norm.npy')

    # electron-electron interaction
    C_K_ee_IMFP = np.load('Resources/GOS/C_GOS_IIMFP.npy')
    O_K_ee_IMFP = np.load('Resources/GOS/O_GOS_IIMFP.npy')
    C_K_ee_DIMFP_norm = np.load('Resources/GOS/C_GOS_DIIMFP_norm.npy')
    O_K_ee_DIMFP_norm = np.load('Resources/GOS/O_GOS_DIIMFP_norm.npy')
    PMMA_val_IMFP = np.load('Resources/Mermin/IIMFP_Mermin_PMMA.npy')
    PMMA_val_DIMFP_norm = np.load('Resources/Mermin/DIIMFP_Mermin_PMMA_norm.npy')
    PMMA_ee_IMFP = C_K_ee_IMFP + O_K_ee_IMFP + PMMA_val_IMFP

    PMMA_ee_DIMFP_norm_3 = np.array((PMMA_val_DIMFP_norm, C_K_ee_DIMFP_norm, O_K_ee_DIMFP_norm))

    Si_ee_IMFP_6 = np.load('Resources/MuElec/Si_MuElec_IIMFP.npy')
    Si_ee_DIMFP_norm_6 = np.load('Resources/MuElec/Si_MuElec_DIIMFP_norm.npy')

    # electron-phonon interaction
    PMMA_ph_IMFP = np.load('Resources/PhPol/PMMA_phonon_IMFP.npy')
    PMMA_pol_IMFP = np.load('Resources/PhPol/PMMA_polaron_IMFP_0p1_0p15.npy')

    # total IMFP
    PMMA_IMFP = np.vstack((PMMA_el_IMFP, PMMA_val_IMFP, C_K_ee_IMFP, O_K_ee_IMFP,
                           PMMA_ph_IMFP, PMMA_pol_IMFP)).transpose()
    PMMA_IMFP_norm = u.norm_2d_array(PMMA_IMFP, axis=1)
    PMMA_total_IMFP = np.sum(PMMA_IMFP, axis=1)

    Si_IMFP = np.vstack((Si_el_IMFP, Si_ee_IMFP_6.transpose())).transpose()
    Si_IMFP_norm = u.norm_2d_array(Si_IMFP, axis=1)
    Si_total_IMFP = np.sum(Si_IMFP, axis=1)

    # structure lists
    structure_IMFP_norm = [PMMA_IMFP_norm, Si_IMFP_norm]
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

    elastic_ind = 0
    PMMA_ph_ind = 4
    PMMA_pol_ind = 5

    PMMA_ind = 0
    Si_ind = 1

    def __init__(self, d_PMMA_nm):
        self.d_PMMA_nm = d_PMMA_nm

    def get_layer_ind(self, electron):
        return int(electron.get_coords()[2] > self.d_PMMA_nm)

    def get_mfp(self, layer_ind, E_ind):
        mfp = -1 / self.structure_total_IMFP[E_ind][layer_ind] * np.log(random.random())
        return mfp

    def get_process_ind(self, layer_ind, E_ind):
        inds = self.structure_proc_inds[layer_ind]
        return random.choice(inds, p=self.structure_IMFP_norm[layer_ind][E_ind])

    def get_elastic_scat_phi_theta(self, layer_ind, E_ind):
        probs = self.structure_elastic_DIMFP[layer_ind][E_ind, :]
        return random.choice(g.THETA_rad, p=probs)

    def get_E_cutoff(self, layer_ind):
        return self.structure_cutoff_E[layer_ind]

    @staticmethod
    def get_phonon_scat_phi_theta_W(electron):
        W = c.hw_phonon
        phi = 2 * np.pi * random.random()

        E = electron.get_E()
        E_prime = E - W
        B = (E + E_prime + 2 * np.sqrt(E * E_prime)) / (E + E_prime - 2 * np.sqrt(E * E_prime))

        u5 = random.random()
        cos_theta = (E + E_prime) / (2 * np.sqrt(E * E_prime)) * (1 - B ** u5) + B ** u5
        theta = np.arccos(cos_theta)

        return phi, theta, W

    @staticmethod
    def T_PMMA(electron):
        E_cos2_theta = electron.get_E_cos2_theta()

        if E_cos2_theta >= c.Wf_PMMA:
            T_PMMA = 4 * np.sqrt(1 - c.Wf_PMMA / E_cos2_theta) / \
                     (1 + np.sqrt(1 - c.Wf_PMMA / E_cos2_theta)) ** 2
            return T_PMMA
        else:
            return 0

    def get_ee_scat_phi_theta_hw_phi2_theta2(self, layer_ind, ss_ind, E, E_ind):
        phi = 2 * np.pi * random.random()

        if layer_ind == self.Si_ind and ss_ind == 0:  # plasmon
            hw = c.Si_MuElec_E_plasmon
            theta = np.arcsin(np.sqrt(hw / E))
            phi_2nd = 2 * np.pi * random.random()
        else:
            probs = self.structure_ee_DIMFP_norm[layer_ind][ss_ind, E_ind, :]
            hw = random.choice(g.EE, p=probs)
            theta = np.arcsin(np.sqrt(hw / E))
            phi_2nd = phi + np.pi

        theta_2nd = np.pi * random.random()

        return phi, theta, hw, phi_2nd, theta_2nd
