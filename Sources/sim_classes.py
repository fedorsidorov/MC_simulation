import importlib

import numpy as np
from numpy import random

import grid as g
import constants as c
import utilities as u

c = importlib.reload(c)


# %%
class Electron:

    x0 = np.mat([[0], [0], [1]])

    def __init__(self, E, coords, O_matrix):
        self.E = E
        self.coords = coords
        self.O_matrix = O_matrix

    def set_E(self, value):
        self.E = value

    def get_E(self):
        return self.E

    def get_E_ind(self):
        return np.argmin(np.abs(g.EE - self.E))

    def get_coords(self):
        return self.coords

    def get_z(self):
        return self.coords[2]

    def set_O_matrix(self, O_matrix):
        self.O_matrix = O_matrix

    def get_O_matrix(self):
        return self.O_matrix

    def scatter(self, phi, theta):
        W = np.mat([[np.cos(phi), np.sin(phi), 0],
                    [-np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta), np.sin(theta)],
                    [np.sin(phi) * np.sin(theta), -np.cos(phi) * np.sin(theta), np.cos(theta)]])
        self.O_matrix = np.matmul(W, self.O_matrix)

    def get_flight_vector(self):
        return np.matmul(self.O_matrix.transpose(), self.x0)

    def make_step(self, step_length):
        self.coords += np.matmul(self.O_matrix.transpose(), self.x0) * step_length


# %%
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

    def __init__(self, d_PMMA_nm):
        self.d_PMMA_nm = d_PMMA_nm

    def get_layer_ind(self, electron):
        return int(electron.get_coords()[2] > self.d_PMMA_nm)

    def get_total_IMFP(self, layer_ind, E_ind):
        return self.structure_total_IMFP[E_ind][layer_ind]

    def get_process_ind(self, layer_ind, E_ind):
        inds = np.arange(np.shape(self.structure_IMFP_norm)[layer_ind][1], dtype=int)
        return random.choice(inds, p=self.structure_IMFP_norm[layer_ind][E_ind])

    def get_elastic_scattering_angles(self, layer_ind, E_ind):
        probs = self.structure_elastic_DIMFP[layer_ind][E_ind]
        return random.choice(g.THETA_rad, p=probs)


# %%
s = Structure(1000)
ans = s.structure_total_IMFP




