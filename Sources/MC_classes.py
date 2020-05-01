import copy
import importlib
from collections import deque
import numpy as np
from tqdm import tqdm
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

    def get_delta_s_for_step_considering_interface(self, IMFP_PMMA, IMFP_Si, u1, free_path, z1, z2):
        p1 = [IMFP_PMMA, IMFP_Si][self.layer_ind]
        p2 = [IMFP_PMMA, IMFP_Si][1 - self.layer_ind]
        scale_factor = np.abs(self.d_PMMA - z1) / np.abs(z2 - z1)
        d = free_path * scale_factor

        if u1 < (1 - np.exp(-p1 * d)) or self.E_ind < indxs.Si_E_cut_ind:
            delta_s = 1 / p1 * (-np.log(1 - u1))
        else:
            delta_s = d + (1 / p2) * (-np.log(1 - u1) - p1 * d)

        return delta_s

    def get_E(self):
        return self.E

    def get_E_cos2_theta(self):
        cos_theta = np.dot(np.mat([[0.], [0.], [-1.]]), self.get_flight_vector_list())
        return self.E * cos_theta ** 2

    def get_e_id(self):
        return self.e_id

    def get_E_ind(self):
        return self.E_ind

    def get_flight_vector_list(self):
        flight_vector_mat = np.matmul(self.O_matrix.transpose(), self.x0)
        return flight_vector_mat[0, 0], flight_vector_mat[1, 0], flight_vector_mat[2, 0]

    def get_history(self):
        history_array = np.asarray(self.history)
        return np.asarray(history_array)

    def get_layer_ind(self):
        return self.layer_ind

    def get_O_matrix(self):
        return self.O_matrix

    def get_parent_e_id(self):
        return self.parent_e_id

    def get_scattered_O_matrix(self, phi, theta):
        W = np.mat([[np.cos(phi), np.sin(phi), 0.],
                    [-np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta), np.sin(theta)],
                    [np.sin(phi) * np.sin(theta), -np.cos(phi) * np.sin(theta), np.cos(theta)]])
        return np.matmul(W, self.O_matrix)

    @staticmethod
    def get_T_PMMA(E_cos2_theta):
        if E_cos2_theta >= const.Wf_PMMA:
            T_PMMA = 4 * np.sqrt(1 - const.Wf_PMMA / E_cos2_theta) / \
                     (1 + np.sqrt(1 - const.Wf_PMMA / E_cos2_theta)) ** 2
            return T_PMMA
        else:
            return 0.

    def make_simple_step(self, step_length):
        delta_r = np.matmul(self.O_matrix.transpose(), self.x0) * step_length
        self.coords += delta_r
        self.update_layer_ind()
        self.n_steps += 1

    def make_step_considering_interface(self, IMFP_PMMA, IMFP_Si):  # Dapor Springer book
        now_IMFP = [IMFP_PMMA, IMFP_Si][self.layer_ind]
        u1 = np.random.random()
        free_path = -1 / now_IMFP * np.log(u1)
        delta_r = np.matmul(self.O_matrix.transpose(), self.x0) * free_path
        next_coords = self.coords + delta_r
        z1, z2 = self.coords[2, 0], next_coords[2, 0]

        if (z1 < self.d_PMMA) ^ (z2 < self.d_PMMA):  # interface crossing
            self.make_simple_step(
                self.get_delta_s_for_step_considering_interface(IMFP_PMMA, IMFP_Si, u1, free_path, z1, z2)
            )
        elif z2 < 0:
            cos_theta = np.dot(self.get_flight_vector_list(), [0., 0., -1.])

            if np.random.random() < self.get_T_PMMA(self.E * cos_theta ** 2):  # electron emerges
                self.make_simple_step(free_path)
            else:  # electron scatters from PMMA-vacuum surface
                scale_factor = self.coords[2, 0] / np.abs(delta_r[2, 0])  # z / dz
                self.make_simple_step(free_path * scale_factor)  # go to surface
                self.write_state_to_history(-1, 0, 0)  # write state
                self.O_matrix[:, 2] *= -1  # change flight direction - CHECK!!!
                self.make_simple_step(free_path * (1 - scale_factor))  # make the rest of step

        else:
            self.make_simple_step(free_path)

    def scatter_with_hw(self, phi, theta, hw):
        self.E -= hw
        self.update_E_ind()
        self.O_matrix = self.get_scattered_O_matrix(phi, theta)

    def set_e_id(self, e_id):
        self.e_id = e_id

    def start(self):
        self.update_layer_ind()
        self.update_E_ind()
        self.write_state_to_history(-1, 0, 0)

    def stop(self, is_polaron):
        hw = self.E
        self.E = 0
        if is_polaron:
            self.write_state_to_history(indxs.sim_PMMA_polaron_ind, hw, 0)
        else:
            self.write_state_to_history(-1, hw, 0)

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

    def get_d_PMMA(self):
        return self.d_PMMA

    def get_ee_scat_phi_theta_hw_phi2_theta2(self, electron, subshell_ind):
        phi = 2 * np.pi * np.random.random()

        if electron.get_layer_ind() == indxs.Si_ind and subshell_ind == indxs.sim_MuElec_plasmon_ind:
            hw = const.Si_MuElec_E_plasmon
            phi_2nd = 2 * np.pi * np.random.random()
        else:
            probs = self.ee_DIMFP_norm[electron.get_layer_ind()][subshell_ind, electron.get_E_ind(), :]
            hw = np.random.choice(grid.EE, p=probs)
            phi_2nd = phi + np.pi

        theta = np.arcsin(np.sqrt(hw / electron.get_E()))
        theta_2nd = np.pi * np.random.random()

        return phi, theta, hw, phi_2nd, theta_2nd

    def get_elastic_scat_phi_theta(self, electron):
        probs = self.elastic_DIMFP[electron.get_layer_ind()][electron.get_E_ind(), :]
        return 2 * np.pi * np.random.random(), np.random.choice(grid.THETA_rad, p=probs)

    def get_free_path(self, electron):
        free_path = -1 / self.total_IMFP[electron.get_layer_ind()][electron.get_E_ind()] * \
                    np.log(np.random.random())
        return free_path

    def get_PMMA_Si_IMFP(self, electron):
        return self.total_IMFP[0][electron.get_E_ind()], self.total_IMFP[1][electron.get_E_ind()]

    def get_phonon_scat_phi_theta_W(self, electron):
        W = self.W_phonon
        phi = 2 * np.pi * np.random.random()
        E = electron.get_E()
        E_prime = E - W
        B = (E + E_prime + 2 * np.sqrt(E * E_prime)) / (E + E_prime - 2 * np.sqrt(E * E_prime))
        u5 = np.random.random()
        cos_theta = (E + E_prime) / (2 * np.sqrt(E * E_prime)) * (1 - B ** u5) + B ** u5
        theta = np.arccos(cos_theta)
        return phi, theta, W

    def get_process_ind(self, electron):
        inds = self.proc_inds[electron.get_layer_ind()]
        probs = self.IMFP_norm[electron.get_layer_ind()][electron.get_E_ind()]
        return np.random.choice(inds, p=probs)


class Event:

    def __init__(self, electron, structure):

        self.layer_ind = electron.get_layer_ind()
        self.E_ind = electron.get_E_ind()
        self.phi, self.theta, self.hw = 0., 0., 0.
        self.E_2nd = 0.
        self.secondary_electron = None
        self.polaron = False
        self.stop = False

        if self.E_ind < structure.E_cutoff_ind[self.layer_ind]:
            self.process_ind = -1
            self.stop = True
            return

        self.process_ind = structure.get_process_ind(electron)

        if self.layer_ind == indxs.PMMA_ind and self.process_ind == indxs.sim_PMMA_polaron_ind:
            self.polaron = True
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

            if self.hw > E_bind and not(self.layer_ind == indxs.Si_ind and
                                        subshell_ind == indxs.sim_MuElec_plasmon_ind):  # secondary generation
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

    def is_polaron(self):
        return self.polaron

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

    def get_total_history(self):
        history = np.vstack(self.total_history)
        history[:, 4:7] *= 1e+7  # cm to nm
        return np.around(np.vstack(history), decimals=5)

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

    def start_simulation(self):
        struct = Structure(self.d_PMMA)
        progress_bar = tqdm(total=self.n_electrons, position=0)

        while self.electrons_deque:
            now_electron = self.electrons_deque.popleft()

            if now_electron.get_parent_e_id() == -1:
                progress_bar.update(1)

            self.track_electron(now_electron, struct)
            self.total_history.append(now_electron.get_history())

        progress_bar.close()

    def track_electron(self, electron, structure):
        electron.start()

        while True:
            event = Event(electron, structure)

            if event.is_stop():
                electron.stop(event.is_polaron())
                break

            electron.make_step_considering_interface(*structure.get_PMMA_Si_IMFP(electron))  # shimizu1992.pdf
            electron.scatter_with_hw(*event.get_primary_phi_theta_hw())
            electron.write_state_to_history(*event.get_process_ind_hw_E2nd())

            if event.secondary_generated():
                new_electron = event.get_secondary_electron()
                new_electron.set_e_id(self.get_new_e_id())
                self.electrons_deque.appendleft(new_electron)
