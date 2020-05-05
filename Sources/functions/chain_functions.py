import importlib
import numpy as np
from numpy import sin, cos
# import os
import matplotlib.pyplot as plt
import constants_physics as const

from mpl_toolkits.mplot3d import Axes3D


# %%
def get_chain_len(mass_array, molecular_weight_array):
    mw_norm = molecular_weight_array / np.sum(molecular_weight_array)
    return int(np.random.choice(mass_array, p=mw_norm) / const.M0)


# def rnd_ang():
#     return 2 * np.pi * np.random.random()


# def rotate_chain(chain):
#     a = rnd_ang()
#     b = rnd_ang()
#     g = rnd_ang()
#
#     M = np.mat([
#         [cos(a) * cos(g) - sin(a) * cos(b) * sin(g), -cos(a) * sin(g) - sin(a) * cos(b) * cos(g), sin(a) * sin(b)],
#         [sin(a) * cos(g) + cos(a) * cos(b) * sin(g), -sin(a) * sin(g) + cos(a) * cos(b) * cos(g), -cos(a) * sin(b)],
#         [sin(b) * sin(g), sin(b) * cos(g), cos(b)]
#     ])
#
#     return np.matmul(M, chain.transpose()).transpose()


# def angle(v1, v2):
#        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    # return np.arccos(np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# def check_chain_bonds(chain_arr):
#     step = 0.28
#
#     for i in range(len(chain_arr) - 2):
#
#         vector_1 = np.array(chain_arr[i + 1] - chain_arr[i])
#         vector_2 = np.array(-(chain_arr[i + 2] - chain_arr[i + 1]))
#
               # print(vector_1.shape)
               # print(vector_2.shape)
        #
        # if np.abs(np.linalg.norm(vector_1) - step) > 1e-4:
        #     print(i, 'bond length error')
        #     return False
        #
        # now_angle = np.rad2deg(angle(vector_1, vector_2))
        #
        # if np.abs(now_angle - 109) > 1e-4:
        #     print(i, 'bond angle error')
        #     return False
    #
    # return True


