import importlib

import numpy as np

import constants as c

c = importlib.reload(c)


# %%
class Electron:

    __x0 = np.mat([[0], [0], [1]])

    def __init__(self, E, coords, O_matrix):
        self.__E = E
        self.__coords = coords
        self.__O_matrix = O_matrix

    def set_E(self, value):
        self.__E = value

    def get_E(self):
        return self.__E

    def get_coords(self):
        return self.__coords

    def get_z(self):
        return self.__coords[2]

    def set_O_matrix(self, O_matrix):
        self.__O_matrix = O_matrix

    def get_O_matrix(self):
        return self.__O_matrix

    def scatter(self, phi, theta):
        W = np.mat([[np.cos(phi), np.sin(phi), 0],
                    [-np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta), np.sin(theta)],
                    [np.sin(phi) * np.sin(theta), -np.cos(phi) * np.sin(theta), np.cos(theta)]])
        self.__O_matrix = np.matmul(W, self.__O_matrix)

    def get_flight_vector(self):
        return np.matmul(self.__O_matrix.transpose(), self.__x0)

    def make_step(self, step_length):
        self.__coords += np.matmul(self.__O_matrix.transpose(), self.__x0) * step_length


class Structure:
    # elastic interactions
    __PMMA_EIMFP = np.load('')

    # electron-electron interaction
    __C_IIMFP = np.load('Resources/GOS/IIMFP_GOS_C.npy')
    __O_IIMFP = np.load('Resources/GOS/IIMFP_GOS_O.npy')
    __PMMA_IIMFP = np.load('Resources/GOS/IIMFP_GOS_PMMA.npy')
    __Si_IIMFP = np.load('Resources/GOS/IIMFP_GOS_PMMA.npy')

    __C_DIIMFP_norm = np.load('Resources/GOS/DIIMFP_GOS_C_norm.npy')
    __O_DIIMFP_norm = np.load('Resources/GOS/DIIMFP_GOS_O_norm.npy')
    __PMMA_DIIMFP_norm = np.load('Resources/GOS/DIIMFP_GOS_PMMA_norm.npy')
    __Si_DIIMFP_norm = np.load('Resources/GOS/DIIMFP_GOS_PMMA_norm.npy')

    def __init__(self, d_PMMA):
        self.__d_PMMA = d_PMMA


