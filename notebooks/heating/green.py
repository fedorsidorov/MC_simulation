import importlib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

import grid
grid = importlib.reload(grid)

# %%
k_1, k_2 = 0.002, 0.014  # W / cm / K
rho_1, rho_2 = 1.2, 2.2  # g / cm^3
Cp_1, Cp_2 = 1.5, 0.75  # J / g / K
alpha_1, alpha_2 = k_1 / (rho_1 * Cp_1), k_2 / (rho_2 * Cp_2)
lambda_12 = (k_1 * np.sqrt(alpha_2) - k_2 * np.sqrt(alpha_1)) / (k_1 * np.sqrt(alpha_2) + k_2 * np.sqrt(alpha_1))

L = 400e-7  # cm
n_terms = 100


def get_green_x(x, xp, t, tp, alpha):
    return 1 / np.sqrt(4 * np.pi * alpha * (t - tp)) * np.exp(-(x - xp)**2 / (4 * alpha * (t - tp)))


def get_green_y(y, yp, t, tp, alpha):
    return 1 / np.sqrt(4 * np.pi * alpha * (t - tp)) * np.exp(-(y - yp)**2 / (4 * alpha * (t - tp)))


def get_green_RR(z, zp, t, tp):
    beg_mult = 1 / np.sqrt(4 * np.pi * alpha_1 * (t - tp))
    term_1 = np.exp(-(z - zp)**2 / (4 * alpha_1 * (t - tp)))
    term_2 = np.exp(-(z + zp)**2 / (4 * alpha_1 * (t - tp)))

    total_sum = 0

    for n in range(1, n_terms + 1):
        total_sum += lambda_12**n * (
            np.exp(-(z - zp + 2 * n * L) ** 2 / (4 * alpha_1 * (t - tp))) +
            np.exp(-(z + zp - 2 * n * L) ** 2 / (4 * alpha_1 * (t - tp))) +
            np.exp(-(z - zp - 2 * n * L) ** 2 / (4 * alpha_1 * (t - tp))) +
            np.exp(-(z + zp + 2 * n * L) ** 2 / (4 * alpha_1 * (t - tp)))
        )

    return beg_mult * (term_1 + term_2 + total_sum)


def get_green_RS(z, zp, t, tp):
    beg_mult = (1 - lambda_12) / np.sqrt(4 * np.pi * alpha_2 * (t - tp))

    total_sum = 0

    for n in range(1, n_terms + 1):
        total_sum += lambda_12 ** n * (
                np.exp(-(np.sqrt(alpha_2 / alpha_1) * (-z + L + 2 * n * L) + (zp - L)) ** 2) +
                np.exp(-(np.sqrt(alpha_2 / alpha_1) * (z + L + 2 * n * L) + (zp - L)) ** 2)
        )

    return beg_mult * total_sum


def get_green_z(z, zp, t, tp):

    if 0 <= z < L and 0 <= zp <= L:
        return get_green_RR(z, zp, t, tp)

    if 0 <= z <= L < zp:
        return get_green_RS(z, zp, t, tp)


def get_green_xyz(x, xp, y, yp, z, zp, t, tp):

    green_x = 0
    green_y = 0
    green_z = 0

    if 0 <= z < L and 0 <= zp <= L:
        green_x = get_green_x(x, xp, t, tp, alpha_1)
        green_y = get_green_x(y, yp, t, tp, alpha_1)
        green_z = get_green_RR(z, zp, t, tp)

    if 0 <= z <= L < zp:
        green_x = get_green_x(x, xp, t, tp, alpha_2)
        green_y = get_green_x(y, yp, t, tp, alpha_2)
        green_z = get_green_RS(z, zp, t, tp)

    return green_x * green_y * green_z








