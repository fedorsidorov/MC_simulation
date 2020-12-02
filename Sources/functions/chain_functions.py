import importlib
import numpy as np
from numpy import sin, cos
# import os
import matplotlib.pyplot as plt
import constants as const

from mpl_toolkits.mplot3d import Axes3D


# %%
def get_chain_len(mass_array, molecular_weight_array):
    mw_norm = molecular_weight_array / np.sum(molecular_weight_array)
    return int(np.random.choice(mass_array, p=mw_norm) / const.u_MMA)


def rotate_chain(chain, a, b, g):
    M = np.mat([
        [cos(a) * cos(g) - sin(a) * cos(b) * sin(g), -cos(a) * sin(g) - sin(a) * cos(b) * cos(g), sin(a) * sin(b)],
        [sin(a) * cos(g) + cos(a) * cos(b) * sin(g), -sin(a) * sin(g) + cos(a) * cos(b) * cos(g), -cos(a) * sin(b)],
        [sin(b) * sin(g), sin(b) * cos(g), cos(b)]
    ])
    return np.array(np.matmul(M, chain.transpose()).transpose())
