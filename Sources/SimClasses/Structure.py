import importlib

import numpy as np
from numpy import random

import grid as g
from SimClasses import utilities as u, constants as c, arrays as a

c = importlib.reload(c)
g = importlib.reload(g)
u = importlib.reload(u)


#%%
class Structure:

    def __init__(self, d_PMMA_nm):
        self.d_PMMA_nm = d_PMMA_nm

    def get_layer_ind(self, electron):
        return int(electron.get_coords()[2] > self.d_PMMA_nm)

    @staticmethod
    def get_mfp(layer_ind, E_ind):
        mfp = -1 / a.structure_total_IMFP[layer_ind][E_ind] * np.log(random.random())
        return mfp

    @staticmethod
    def get_process_ind(layer_ind, E_ind):
        inds = a.structure_proc_inds[layer_ind]
        return random.choice(inds, p=a.structure_IMFP_norm[layer_ind][E_ind])

    @staticmethod
    def get_elastic_scat_phi_theta(layer_ind, E_ind):
        probs = a.structure_elastic_DIMFP[layer_ind][E_ind, :]
        return random.choice(g.THETA_rad, p=probs)

    @staticmethod
    def get_E_cutoff(layer_ind):
        return a.structure_cutoff_E[layer_ind]

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
            return 0.

    @staticmethod
    def get_ee_scat_phi_theta_hw_phi2_theta2(layer_ind, ss_ind, E, E_ind):
        phi = 2 * np.pi * random.random()

        if layer_ind == a.Si_ind and ss_ind == 0:  # plasmon
            hw = c.Si_MuElec_E_plasmon
            theta = np.arcsin(np.sqrt(hw / E))
            phi_2nd = 2 * np.pi * random.random()
        else:
            probs = a.structure_ee_DIMFP_norm[layer_ind][ss_ind, E_ind, :]
            hw = random.choice(g.EE, p=probs)
            theta = np.arcsin(np.sqrt(hw / E))
            phi_2nd = phi + np.pi

        theta_2nd = np.pi * random.random()

        return phi, theta, hw, phi_2nd, theta_2nd
