import copy
import importlib
from collections import deque

import numpy as np
from numpy import random

import arrays
import constants as const
import grid as grid
import indexes as indxs
import utilities as utils

arrays = importlib.reload(arrays)
const = importlib.reload(const)
grid = importlib.reload(grid)
indxs = importlib.reload(indxs)
utils = importlib.reload(utils)


class Electron:

    def __init__(self, e_id, parent_e_id, E, coords, O_matrix, d_PMMA):
        self.e_id = e_id
        self.parent_e_id = parent_e_id
        self.E = E
        self.coords = coords
        self.O_matrix = O_matrix
        self.history = deque()
        self.x0 = np.mat([[0.], [0.], [1.]])
        self.d_PMMA = d_PMMA
        self.layer_ind = indxs.vacuum_ind
        self.E_ind = 0
        self.n_steps = 0

    def get_coords_list(self):
        return self.coords[0, 0], self.coords[1, 0], self.coords[2, 0]

    def get_coords_matrix(self):
        return self.coords

    def get_E(self):
        return self.E

    def get_E_cos2_theta(self):
        cos_theta = np.dot(np.mat([[0.], [0.], [-1.]]), self.get_flight_vector())
        return self.E * cos_theta ** 2

    def get_e_id(self):
        return self.e_id

    def get_E_ind(self):
        return self.E_ind

    def get_flight_vector(self):
        return np.matmul(self.O_matrix.transpose(), self.x0)

    def get_history(self):
        history_array = np.asarray(self.history)
        return np.asarray(history_array)

    def get_layer_ind(self):
        return self.layer_ind

    def get_O_matrix(self):
        return self.O_matrix

    def get_scattered_O_matrix(self, phi, theta):
        W = np.mat([[np.cos(phi), np.sin(phi), 0.],
                    [-np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta), np.sin(theta)],
                    [np.sin(phi) * np.sin(theta), -np.cos(phi) * np.sin(theta), np.cos(theta)]])
        return np.matmul(W, self.O_matrix)

    def make_step(self, step_length):
        delta_r = np.matmul(self.O_matrix.transpose(), self.x0) * step_length
        self.coords += delta_r
        self.update_layer_ind()
        self.n_steps += 1

    def scatter_with_hw(self, phi, theta, hw):
        self.E -= hw
        self.update_E_ind()
        self.O_matrix = self.get_scattered_O_matrix(phi, theta)

    def set_e_id(self, e_id):
        self.e_id = e_id

    def set_O_matrix(self, O_matrix):
        self.O_matrix = O_matrix

    def start(self):
        self.update_layer_ind()
        self.update_E_ind()
        self.write_state_to_history(-1, 0, 0)

    def stop(self):
        self.write_state_to_history(-1, self.E, 0)

    def update_E_ind(self):
        self.E_ind = np.argmin(np.abs(grid.EE - self.E))

    def update_layer_ind(self):
        if self.coords[2, 0] >= self.d_PMMA:
            self.layer_ind = indxs.Si_ind
        elif self.coords[2, 0] >= 0:
            self.layer_ind = indxs.PMMA_ind
        else:
            self.layer_ind = indxs.vacuum_ind

    def write_state_to_history(self, proc_ind, hw, E_2nd):
        E_deposited = hw - E_2nd
        history_line = [self.e_id, self.parent_e_id, self.layer_ind, proc_ind,
                        *self.get_coords_list(), E_deposited, E_2nd, self.E]
        self.history.append(history_line)


class Structure:

    def __init__(self, d_PMMA):
        self.d_PMMA = d_PMMA

        self.IMFP_norm = [arrays.PMMA_IMFP_norm, arrays.Si_IMFP_norm]
        self.total_IMFP = [arrays.PMMA_total_IMFP, arrays.Si_total_IMFP]
        self.elastic_DIMFP = [arrays.PMMA_el_DIMFP_norm, arrays.Si_el_DIMFP_norm]
        self.ee_DIMFP_norm = [arrays.PMMA_ee_DIMFP_norm_3, arrays.Si_ee_DIMFP_norm_6]

        self.E_bind = [
            [const.val_E_bind_PMMA, const.K_Ebind_C, const.K_Ebind_O],
            const.Si_MuElec_E_bind
        ]

        self.proc_inds = [
            [i for i in range(len(self.IMFP_norm[0][0]))],
            [i for i in range(len(self.IMFP_norm[1][0]))]
        ]

        self.E_cutoff_ind = [0, 0, 0]
        self.E_cutoff_ind[indxs.vacuum_ind] = len(grid.EE) - 1
        self.E_cutoff_ind[indxs.PMMA_ind] = indxs.PMMA_E_cut_ind
        self.E_cutoff_ind[indxs.Si_ind] = indxs.Si_E_cut_ind

        self.W_phonon = const.W_phonon
        self.Wf_PMMA = const.Wf_PMMA

    def get_mfp(self, electron):
        mfp = -1 / self.total_IMFP[electron.get_layer_ind()][electron.get_E_ind()] * np.log(random.random())
        return mfp

    def get_process_ind(self, electron):
        inds = self.proc_inds[electron.get_layer_ind()]
        probs = self.IMFP_norm[electron.get_layer_ind()][electron.get_E_ind()]
        return random.choice(inds, p=probs)

    def get_elastic_scat_phi_theta(self, electron):
        probs = self.elastic_DIMFP[electron.get_layer_ind()][electron.get_E_ind(), :]
        return 2 * np.pi * random.random(), random.choice(grid.THETA_rad, p=probs)

    def get_phonon_scat_phi_theta_W(self, electron):
        W = self.W_phonon
        phi = 2 * np.pi * random.random()
        E = electron.get_E()
        E_prime = E - W
        B = (E + E_prime + 2 * np.sqrt(E * E_prime)) / (E + E_prime - 2 * np.sqrt(E * E_prime))
        u5 = random.random()
        cos_theta = (E + E_prime) / (2 * np.sqrt(E * E_prime)) * (1 - B ** u5) + B ** u5
        theta = np.arccos(cos_theta)
        return phi, theta, W

    def T_PMMA(self, electron):
        E_cos2_theta = electron.get_E_cos2_theta()

        if E_cos2_theta >= const.Wf_PMMA:
            T_PMMA = 4 * np.sqrt(1 - self.Wf_PMMA / E_cos2_theta) / \
                     (1 + np.sqrt(1 - self.Wf_PMMA / E_cos2_theta)) ** 2
            return T_PMMA
        else:
            return 0.

    def get_ee_scat_phi_theta_hw_phi2_theta2(self, electron, subshell_ind):
        phi = 2 * np.pi * random.random()

        if electron.get_layer_ind == indxs.Si_ind and subshell_ind == indxs.sim_MuElec_plasmon_ind:  # plasmon
            hw = const.Si_MuElec_E_plasmon
            phi_2nd = 2 * np.pi * random.random()
        else:
            probs = self.ee_DIMFP_norm[electron.get_layer_ind()][subshell_ind, electron.get_E_ind(), :]
            hw = random.choice(grid.EE, p=probs)
            phi_2nd = phi + np.pi

        theta = np.arcsin(np.sqrt(hw / electron.get_E()))
        theta_2nd = np.pi * random.random()

        return phi, theta, hw, phi_2nd, theta_2nd

    def get_d_PMMA(self):
        return self.d_PMMA


class Event:

    def __init__(self, electron, structure):

        self.layer_ind = electron.get_layer_ind()
        self.E_ind = electron.get_E_ind()
        self.phi, self.theta, self.hw = 0., 0., 0.
        self.E_2nd = 0.
        self.secondary_electron = None
        self.stop = False

        if self.E_ind < structure.E_cutoff_ind[self.layer_ind]:
            self.process_ind = -1
            self.stop = True
            return

        self.process_ind = structure.get_process_ind(electron)

        if self.layer_ind == indxs.PMMA_ind and self.process_ind == indxs.sim_PMMA_polaron_ind:
            self.stop = True

        elif self.process_ind == indxs.sim_elastic_ind:
            self.phi, self.theta = structure.get_elastic_scat_phi_theta(electron)

        elif self.layer_ind == indxs.PMMA_ind and self.process_ind == indxs.sim_PMMA_phonon_ind:
            self.phi, self.theta, self.hw = structure.get_phonon_scat_phi_theta_W(electron)

        else:  # electron-electron interaction
            subshell_ind = self.process_ind - 1
            self.phi, self.theta, self.hw, phi_2nd, theta_2nd = \
                structure.get_ee_scat_phi_theta_hw_phi2_theta2(electron, subshell_ind)
            E_bind = structure.E_bind[self.layer_ind][subshell_ind]

            if self.hw > E_bind:  # secondary generation
                self.E_2nd = self.hw - E_bind
                self.secondary_electron = Electron(
                    e_id=-1,
                    parent_e_id=electron.get_e_id(),
                    E=self.E_2nd,
                    coords=copy.deepcopy(electron.get_coords_matrix()),
                    O_matrix=copy.deepcopy(electron.get_scattered_O_matrix(phi_2nd, theta_2nd)),
                    d_PMMA=structure.get_d_PMMA()
                )

    def secondary_generated(self):
        return self.secondary_electron is not None

    def get_primary_phi_theta_hw(self):
        return self.phi, self.theta, self.hw

    def get_process_ind_hw_E2nd(self):
        return self.process_ind, self.hw, self.E_2nd

    def get_secondary_electron(self):
        return self.secondary_electron

    def is_stop(self):
        return self.stop


class Simulator:

    def __init__(self, d_PMMA, n_electrons, E0_eV):
        self.d_PMMA = d_PMMA
        self.n_electrons = n_electrons
        self.E0 = E0_eV
        self.e_cnt = -1
        self.electrons_deque = deque()
        self.total_history = deque()

    def get_new_e_id(self):
        self.e_cnt += 1
        return self.e_cnt

    def prepare_e_deque(self):
        for _ in range(self.n_electrons):
            electron = Electron(
                e_id=self.get_new_e_id(),
                parent_e_id=-1,
                E=self.E0,
                coords=np.mat([[0.], [0.], [0.]]),
                O_matrix=np.mat(np.eye(3)),
                d_PMMA=self.d_PMMA
            )
            self.electrons_deque.append(electron)

    def track_electron(self, electron, structure):
        electron.start()

        while True:
            event = Event(electron, structure)

            if event.is_stop():
                electron.stop()
                break

            electron.make_step(structure.get_mfp(electron))  # shimizu1992.pdf
            electron.scatter_with_hw(*event.get_primary_phi_theta_hw())
            electron.write_state_to_history(*event.get_process_ind_hw_E2nd())

            if event.secondary_generated():
                new_electron = event.get_secondary_electron()
                new_electron.set_e_id(self.get_new_e_id())
                self.electrons_deque.append(new_electron)

    def start_simulation(self):
        struct = Structure(self.d_PMMA)

        while self.electrons_deque:
            now_electron = self.electrons_deque.popleft()
            self.track_electron(now_electron, struct)
            self.total_history.append(now_electron.get_history())

    def get_total_history(self):
        history = np.vstack(self.total_history)
        history[:, 4:7] *= 1e+7  # cm to nm
        return np.around(np.vstack(history), decimals=5)
