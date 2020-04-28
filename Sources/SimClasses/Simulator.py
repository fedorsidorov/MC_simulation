# %%
import importlib
from collections import deque

import numpy as np
from numpy import random
from numpy import matlib

import constants as c
import grid as g
import utilities as u

import SimClasses.Electron as Electron
import SimClasses.Structure as Structure

c = importlib.reload(c)
g = importlib.reload(g)
u = importlib.reload(u)
Electron = importlib.reload(Electron)
Structure = importlib.reload(Structure)


# %%
class Simulator:

    electrons_deque = deque()
    total_history = deque()

    def __init__(self, d_PMMA_nm, n_electrons, E0):
        self.d_PMMA_nm = d_PMMA_nm
        self.n_electrons = n_electrons
        self.E0 = E0

    def prepare_e_deque(self):
        for i in range(self.n_electrons):
            now_electron = Electron(
                e_id=i,
                parent_e_id=-1,
                E=self.E0,
                coords=np.mat([[0], [0], [1]]),
                O_matrix=matlib.eye(3)
            )
            self.electrons_deque.append(now_electron)

    def track_electron(self, now_e, struct):
        E_ind = now_e.get_E_ind()
        layer_ind = struct.get_layer_ind(now_e)
        now_e.start(struct.get_layer_ind(now_e))

        while now_e.get_E() > struct.get_E_cutoff(layer_ind):
            mfp = struct.get_mfp(layer_ind, E_ind)
            now_e.make_step(mfp)  # write new coordinates

            proc_ind = struct.get_process_ind(layer_ind, E_ind)
            E_2nd = 0

            if proc_ind == struct.elastic_ind:  # elastic scattering
                phi, theta = struct.get_elastic_scat_phi_theta(layer_ind, E_ind)
                now_e.scatter_with_E_loss(phi, theta, 0)
                hw = 0

            elif layer_ind == struct.PMMA_ind and proc_ind == struct.PMMA_ph_ind:  # PMMA phonons
                phi, theta, W = struct.get_phonon_scat_phi_theta_W(now_e)
                now_e.scatter_with_E_loss(phi, theta, W)
                hw = W

            elif layer_ind == struct.PMMA_ind and proc_ind == struct.PMMA_pol_ind:  # PMMA polarons
                break

            else:  # electron-electron interaction
                ss_ind = proc_ind - 1
                phi, theta, hw, phi_2nd, theta_2nd = \
                    struct.get_ee_scat_phi_theta_hw_phi2_theta2(layer_ind, ss_ind, now_e.get_E, E_ind)

                E_bind = struct.structure_E_bind[layer_ind][ss_ind]

                if hw > E_bind:  # secondary generation
                    E_2nd = hw - E_bind
                    e_2nd = Electron(
                        e_id=len(self.electrons_deque),
                        parent_e_id=now_e.get_e_id(),
                        E=E_2nd,
                        coords=now_e.get_coords(),
                        O_matrix=now_e.get_scattered_O_matrix(phi_2nd, theta_2nd)
                    )
                    self.electrons_deque.append(e_2nd)

                now_e.scatter_with_E_loss(phi, theta, hw)

            now_e.write_state_to_history(layer_ind, proc_ind, hw, E_2nd)

        now_e.stop(struct.get_layer_ind(now_e))

    def start_simulation(self):
        struct = Structure(self.d_PMMA_nm)

        # the main simulation cycle
        while self.electrons_deque:
            now_e = self.electrons_deque.popleft()
            self.track_electron(now_e, struct)
            self.total_history.append(now_e.get_history())

    def get_total_history(self):
        np.asarray(self.total_history)
