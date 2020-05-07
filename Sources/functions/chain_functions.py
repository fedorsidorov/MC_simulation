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

