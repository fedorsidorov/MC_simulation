import importlib
from collections import deque

import numpy as np
from numpy import random
from numpy import matlib

import constants as c
import grid as g
import utilities as u

c = importlib.reload(c)
g = importlib.reload(g)
u = importlib.reload(u)


class Electron:
    x0 = np.mat([[0], [0], [1]])
    history = deque()

    def __init__(self, e_id, parent_e_id, E, coords, O_matrix):
        self.e_id = e_id
        self.parent_e_id = parent_e_id
        self.E = E
        self.coords = coords
        self.O_matrix = O_matrix
        self.history.append(coords)

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

    def get_e_id(self):
        return self.e_id

    def get_scattered_O_matrix(self, phi, theta):
        W = np.mat([[np.cos(phi), np.sin(phi), 0],
                    [-np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta), np.sin(theta)],
                    [np.sin(phi) * np.sin(theta), -np.cos(phi) * np.sin(theta), np.cos(theta)]])
        return np.matmul(W, self.O_matrix)

    def scatter_with_E_loss(self, phi, theta, E_loss):
        self.E -= E_loss
        self.O_matrix = self.get_scattered_O_matrix(phi, theta)

    def get_flight_vector(self):
        return np.matmul(self.O_matrix.transpose(), self.x0)

    def get_E_cos2_theta(self):
        cos_theta = np.dot(np.mat([[0], [0], [-1]]), self.get_flight_vector())
        return self.E * cos_theta**2

    def make_step(self, step_length):
        self.coords += np.matmul(self.O_matrix.transpose(), self.x0) * step_length

    def write_state_to_history(self, layer_ind, proc_ind, hw, E_2nd):
        E_dep = hw - E_2nd
        history_line = [self.e_id, self.parent_e_id, layer_ind, proc_ind,
                        *list(self.coords), E_dep, E_2nd, self.E]
        self.history.append(history_line)

    def start(self, layer_ind):
        history_line = [self.e_id, self.parent_e_id, layer_ind, -1,
                        *list(self.coords), 0, 0, self.E]
        self.history.append(history_line)

    def stop(self, layer_ind):
        history_line = [self.e_id, self.parent_e_id, layer_ind, -1,
                        *list(self.coords), self.E, 0, 0]
        self.history.append(history_line)

    def get_history(self):
        return np.asarray(self.history)
