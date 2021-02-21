import copy
import importlib
from collections import deque
import numpy as np
from tqdm import tqdm
import arrays_new as arrays
import constants
import grid as grid
import indexes as indexes
from functions import MC_functions as utils
from scipy import interpolate

arrays = importlib.reload(arrays)
constants = importlib.reload(constants)
indexes = importlib.reload(indexes)
utils = importlib.reload(utils)


class Electron:

    def __init__(self, e_id, parent_e_id, E, coords, flight_ort, structure):
        self.e_id = e_id
        self.parent_e_id = parent_e_id
        self.E = E
        self.coords = coords
        self.flight_ort = flight_ort
        self.history = deque()
        self.structure = structure
        self.layer_ind = indexes.vacuum_ind
        self.E_ind = 0
        self.n_steps = 0

    def get_coords(self):
        return copy.deepcopy(self.coords)

    def get_delta_s_for_step_considering_interface(self, IMFP_PMMA, IMFP_Si, u1, free_path, z1, z2):
        p1 = [IMFP_PMMA, IMFP_Si][self.layer_ind]
        p2 = [IMFP_PMMA, IMFP_Si][1 - self.layer_ind]
        scale_factor = np.abs(self.structure.d_PMMA - z1) / np.abs(z2 - z1)
        d = free_path * scale_factor

        if u1 < (1 - np.exp(-p1 * d)) or self.E_ind < indexes.Si_E_cut_ind:
            delta_s = 1 / p1 * (-np.log(1 - u1))
        else:
            delta_s = d + (1 / p2) * (-np.log(1 - u1) - p1 * d)

        return delta_s

    def get_E(self):
        return self.E

    def get_E_cos2_theta(self):
        cos_theta = np.dot(np.array((0., 0., -1.)), self.flight_ort)
        return self.E * cos_theta ** 2

    def get_e_id(self):
        return self.e_id

    def get_E_ind(self):
        return self.E_ind

    def get_history(self):
        history_array = np.asarray(self.history)
        return np.asarray(history_array)

    def get_layer_ind(self):
        return self.layer_ind

    def get_parent_e_id(self):
        return self.parent_e_id

    def get_scattered_flight_ort(self, phi_scat, theta_scat):
        u, v, w = self.flight_ort

        if w == 1:
            w -= 1e-10
        elif w == -1:
            w += 1e-10

        u_new = u * np.cos(theta_scat) +\
            np.sin(theta_scat) / np.sqrt(1 - w**2) * (u * w * np.cos(phi_scat) - v * np.sin(phi_scat))

        v_new = v * np.cos(theta_scat) +\
            np.sin(theta_scat) / np.sqrt(1 - w**2) * (v * w * np.cos(phi_scat) + u * np.sin(phi_scat))

        w_new = w * np.cos(theta_scat) - np.sqrt(1 - w**2) * np.sin(theta_scat) * np.cos(phi_scat)

        return np.array((u_new, v_new, w_new))

    def scatter(self, phi_scat, theta_scat):
        self.flight_ort = self.get_scattered_flight_ort(phi_scat, theta_scat)

    @staticmethod
    def get_T_PMMA(E_cos2_theta):
        if E_cos2_theta >= constants.Wf_PMMA:
            T_PMMA = 4 * np.sqrt(1 - constants.Wf_PMMA / E_cos2_theta) / \
                     (1 + np.sqrt(1 - constants.Wf_PMMA / E_cos2_theta)) ** 2
            return T_PMMA
        else:
            return 0.

    def make_simple_step(self, step_length):
        delta_r = self.flight_ort * step_length
        self.coords += delta_r
        self.update_layer_ind()
        self.n_steps += 1

    def make_step_considering_interface(self, IMFP_PMMA, IMFP_Si):  # Dapor Springer book
        now_IMFP = [IMFP_PMMA, IMFP_Si][self.layer_ind]
        u1 = np.random.random()
        free_path = -1 / now_IMFP * np.log(u1)
        delta_r = self.flight_ort * free_path
        next_coords = self.coords + delta_r
        z1 = self.coords[2]
        x2, z2 = next_coords[0], next_coords[2]

        if (z1 < self.structure.d_PMMA) ^ (z2 < self.structure.d_PMMA):  # interface crossing
            self.make_simple_step(
                self.get_delta_s_for_step_considering_interface(IMFP_PMMA, IMFP_Si, u1, free_path, z1, z2)
            )
        elif z2 < self.structure.get_z_vac_for_x(x2):
            cos_theta = np.dot(self.flight_ort, [0., 0., -1.])

            if np.random.random() < self.get_T_PMMA(self.E * cos_theta ** 2):  # electron emerges
                if self.E * cos_theta ** 2 < constants.Wf_PMMA:
                    print('Wf problems', self.E, cos_theta)
                self.make_simple_step(free_path)

            else:  # electron scatters from PMMA-vacuum surface  # TODO
                # scale_factor = self.coords[2] / np.abs(delta_r[2])  # z / dz
                scale_factor = 0  # hot fix lol
                # self.make_simple_step(free_path * scale_factor)  # go to surface
                # self.write_state_to_history(-1, 0, 0)  # write state

                u, v, w = self.flight_ort
                self.flight_ort = np.array((u, v, -w))

                self.make_simple_step(free_path * (1 - scale_factor))  # make the rest of step

        else:
            self.make_simple_step(free_path)

    def scatter_with_hw(self, phi_scat, theta_scat, hw):
        self.E -= hw
        self.update_E_ind()
        self.scatter(phi_scat, theta_scat)

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
            self.write_state_to_history(indexes.sim_PMMA_polaron_ind, hw, 0)
        else:
            self.write_state_to_history(-1, hw, 0)

    def update_E_ind(self):
        self.E_ind = np.argmin(np.abs(grid.EE - self.E))

    def update_layer_ind(self):
        if self.coords[2] >= self.structure.d_PMMA:
            self.layer_ind = indexes.Si_ind
        else:
            z_interface = self.structure.get_z_vac_for_x(self.coords[0])
            if self.coords[2] >= z_interface:
                self.layer_ind = indexes.PMMA_ind
            else:
                self.layer_ind = indexes.vacuum_ind

    def write_state_to_history(self, proc_ind, hw, E_2nd):
        E_deposited = hw - E_2nd
        history_line = [self.e_id, self.parent_e_id, self.layer_ind, proc_ind,
                        *self.get_coords(), E_deposited, E_2nd, self.E]
        self.history.append(history_line)


class Structure:

    def __init__(self, d_PMMA, xx, zz_vac, ly):  # now in nanometers !!!
        self.d_PMMA = d_PMMA
        self.xx = xx
        self.zz_vac = zz_vac
        self.ly = ly

        self.IMFP_norm = [arrays.PMMA_IMFP_norm, arrays.Si_IMFP_norm]
        self.total_IMFP = [arrays.PMMA_total_IMFP, arrays.Si_total_IMFP]
        self.elastic_DIMFP_cumulated = [arrays.PMMA_el_DIMFP_cumulated, arrays.Si_el_DIMFP_cumulated]
        self.ee_DIMFP_cumulated = [arrays.PMMA_ee_DIMFP_3_cumulated, arrays.Si_ee_DIMFP_6_cumulated]

        self.E_bind = [
            [constants.val_E_bind_PMMA, constants.K_Ebind_C, constants.K_Ebind_O],
            constants.Si_MuElec_E_bind
        ]

        self.proc_inds = [
            [i for i in range(len(self.IMFP_norm[0][0]))],
            [i for i in range(len(self.IMFP_norm[1][0]))]
        ]

        self.E_cutoff_ind = [0, 0, 0]
        self.E_cutoff_ind[indexes.vacuum_ind] = len(grid.EE) - 1
        self.E_cutoff_ind[indexes.PMMA_ind] = indexes.PMMA_E_cut_ind
        self.E_cutoff_ind[indexes.Si_ind] = indexes.Si_E_cut_ind

        self.W_phonon = constants.W_phonon
        self.Wf_PMMA = constants.Wf_PMMA

    def get_d_PMMA(self):
        return self.d_PMMA

    def get_ee_scat_phi_theta_hw_phi2_theta2(self, electron, subshell_ind):
        phi = 2 * np.pi * np.random.random()

        if electron.get_layer_ind() == indexes.Si_ind and subshell_ind == indexes.sim_MuElec_plasmon_ind:
            hw = constants.Si_MuElec_E_plasmon
            phi_2nd = 2 * np.pi * np.random.random()
        else:
            now_rnd = np.random.random()
            hw_ind = np.argmin(np.abs(
                self.ee_DIMFP_cumulated[electron.get_layer_ind()][subshell_ind, electron.get_E_ind(), :] - now_rnd))
            hw = grid.EE[hw_ind]
            phi_2nd = phi + np.pi

        theta = np.arcsin(np.sqrt(hw / electron.get_E()))
        # TODO
        theta_2nd = np.pi * np.random.random()

        return phi, theta, hw, phi_2nd, theta_2nd

    def get_elastic_scat_phi_theta(self, electron):
        now_rnd = np.random.random()
        theta_ind = np.argmin(np.abs(
            self.elastic_DIMFP_cumulated[electron.get_layer_ind()][electron.get_E_ind(), :] - now_rnd))
        return 2 * np.pi * np.random.random(), grid.THETA_rad[theta_ind]

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

    def get_z_vac_for_x(self, x):
        if x > np.max(self.xx):
            return self.zz_vac[-1]
        elif x < np.min(self.xx):
            return self.zz_vac[0]
        else:
            return interpolate.interp1d(self.xx, self.zz_vac)(x)

    def set_xx_zz(self, xx, zz_vac):
        self.xx = xx
        self.zz_vac = zz_vac


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

        if self.layer_ind == indexes.PMMA_ind and self.process_ind == indexes.sim_PMMA_polaron_ind:
            self.polaron = True
            self.stop = True

        elif self.process_ind == indexes.sim_elastic_ind:
            self.phi, self.theta = structure.get_elastic_scat_phi_theta(electron)

        elif self.layer_ind == indexes.PMMA_ind and self.process_ind == indexes.sim_PMMA_phonon_ind:
            self.phi, self.theta, self.hw = structure.get_phonon_scat_phi_theta_W(electron)

        else:  # electron-electron interaction
            subshell_ind = self.process_ind - 1
            self.phi, self.theta, self.hw, phi_2nd, theta_2nd = \
                structure.get_ee_scat_phi_theta_hw_phi2_theta2(electron, subshell_ind)
            E_bind = structure.E_bind[self.layer_ind][subshell_ind]

            if self.hw > E_bind and not(self.layer_ind == indexes.Si_ind and
                                        subshell_ind == indexes.sim_MuElec_plasmon_ind):  # secondary generation
                self.E_2nd = self.hw - E_bind

                flight_ort_2nd = np.array((
                    np.sin(theta_2nd) * np.cos(phi_2nd), np.sin(theta_2nd) * np.sin(phi_2nd), np.cos(theta_2nd)
                ))

                self.secondary_electron = Electron(
                    e_id=-1,
                    parent_e_id=electron.get_e_id(),
                    E=self.E_2nd,
                    coords=electron.get_coords(),
                    flight_ort=flight_ort_2nd,
                    structure=structure
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

    def __init__(self, structure, n_electrons, E0_eV, r_beam_x, r_beam_y):
        self.structure = structure
        self.n_electrons = n_electrons
        self.E0 = E0_eV
        self.r_beam_x = r_beam_x
        self.r_beam_y = r_beam_y
        self.e_cnt = -1
        self.electrons_deque = deque()
        self.total_history = deque()

    def get_new_e_id(self):
        self.e_cnt += 1
        return self.e_cnt

    def get_total_history(self):
        history = np.vstack(self.total_history)
        return np.around(np.vstack(history), decimals=5)

    def prepare_e_deque(self):
        for _ in range(self.n_electrons):

            x0 = np.random.uniform(-self.r_beam_x, self.r_beam_y)
            # x0 = np.random.normal(loc=0, scale=self.r_beam)
            y0 = np.random.uniform(-self.r_beam_y, self.r_beam_y)
            z0 = self.structure.get_z_vac_for_x(x0)

            electron = Electron(
                e_id=self.get_new_e_id(),
                parent_e_id=-1,
                E=self.E0,
                coords=np.array((x0, y0, z0)),
                flight_ort=np.array((0, 0, 1)),
                structure=self.structure
            )
            self.electrons_deque.append(electron)

    def start_simulation(self):
        progress_bar = tqdm(total=self.n_electrons, position=0)

        while self.electrons_deque:
            now_electron = self.electrons_deque.popleft()

            if now_electron.get_parent_e_id() == -1:
                progress_bar.update(1)

            self.track_electron(now_electron, self.structure)
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
