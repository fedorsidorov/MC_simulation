# %%
import importlib
from collections import deque

import numpy as np

import SimClasses.Electron as Electron
import SimClasses.Structure as Structure
import grid as g
from SimClasses import utilities as u, constants as c, arrays as a

c = importlib.reload(c)
g = importlib.reload(g)
u = importlib.reload(u)
Electron = importlib.reload(Electron)
Structure = importlib.reload(Structure)


# %%
class Simulator:

    def __init__(self, d_PMMA_nm, n_electrons, E0_eV):
        self.d_PMMA_nm = d_PMMA_nm
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
            now_electron = Electron.Electron(
                e_id=self.get_new_e_id(),
                parent_e_id=-1,
                E=self.E0,
                coords=np.mat([[0.], [0.], [0.]]),
                O_matrix=np.mat(np.eye(3))
            )
            self.electrons_deque.append(now_electron)

    def add_2nd_e(self, e_parent, phi_2nd, theta_2nd, E_2nd):
        e_2nd = Electron.Electron(
            e_id=self.get_new_e_id(),
            parent_e_id=e_parent.get_e_id(),
            E=E_2nd,
            coords=e_parent.get_coords_matrix(),
            O_matrix=e_parent.get_scattered_O_matrix(phi_2nd, theta_2nd)
        )
        self.electrons_deque.append(e_2nd)

    def track_electron(self, now_e, struct):
        now_e.start(struct.get_layer_ind(now_e))

        while True:
            layer_ind = struct.get_layer_ind(now_e)
            E_ind = now_e.get_E_ind()

            # if now_e.get_E() < struct.get_E_cutoff(layer_ind) or layer_ind == c.vacuum_ind:
            #     break
            if layer_ind == c.PMMA_ind and g.EE[E_ind] < c.PMMA_E_cutoff or \
                    layer_ind == c.Si_ind and g.EE[E_ind] < c.Si_MuElec_E_plasmon or \
                    layer_ind == c.vacuum_ind:
                break

            # first!!!
            now_e.make_step(struct.get_mfp(layer_ind, E_ind))  # write new coordinates

            proc_ind = struct.get_process_ind(layer_ind, E_ind)
            E_2nd = 0

            if proc_ind == c.elastic_ind:  # elastic scattering
                phi, theta = struct.get_elastic_scat_phi_theta(layer_ind, E_ind)
                hw = 0

            elif layer_ind == c.PMMA_ind and proc_ind == c.PMMA_ph_ind:  # PMMA phonons
                phi, theta, hw = struct.get_phonon_scat_phi_theta_W(now_e)

            elif layer_ind == c.PMMA_ind and proc_ind == c.PMMA_pol_ind:  # PMMA polarons
                break

            else:  # electron-electron interaction
                ss_ind = proc_ind - 1
                phi, theta, hw, phi_2nd, theta_2nd = \
                    struct.get_ee_scat_phi_theta_hw_phi2_theta2(layer_ind, ss_ind, now_e.get_E(), E_ind)
                E_bind = a.structure_E_bind[layer_ind][ss_ind]

                if hw > E_bind:  # secondary generation
                    E_2nd = hw - E_bind
                    self.add_2nd_e(now_e, phi_2nd, theta_2nd, E_2nd)

            now_e.scatter_with_E_loss(phi, theta, hw)
            now_e.write_state_to_history(layer_ind, proc_ind, hw, E_2nd)

        now_e.stop(struct.get_layer_ind(now_e))

    def start_simulation(self):
        struct = Structure.Structure(self.d_PMMA_nm * 1e-7)

        # the main simulation cycle
        while self.electrons_deque:
            now_e = self.electrons_deque.popleft()
            self.track_electron(now_e, struct)
            self.total_history.append(now_e.get_history())

    def get_total_history(self):
        history = np.vstack(self.total_history)
        history[:, 4:7] *= 1e+7  # cm to nm
        return np.vstack(history)
