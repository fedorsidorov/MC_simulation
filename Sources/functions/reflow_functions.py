import importlib

import numpy as np
from scipy.integrate import quad

from functions import MC_functions as mf

mf = importlib.reload(mf)


def get_PMMA_surface_tension(T_C):  # wu1970.pdf
    gamma_CGS = 41.1 - 0.076 * (T_C - 20)
    gamma_SI = gamma_CGS * 1e-3
    return gamma_SI


def get_PMMA_viscosity(T_C):  # jones2006.pdf
    T0 = 86  # C
    C1 = 70.1
    C2 = -12.21
    eta0 = 1e+12  # Pa s
    eta = eta0 * (T_C - T0) / (C2 + (T_C - T0))
    return eta


def get_tau_n(n, A, eta, gamma, h0, l0):
    if n == 0:
        return np.inf
    part_1 = (n * 2 * np.pi / l0) ** 2 * A / (6 * np.pi * h0 * eta)
    part_2 = (n * 2 * np.pi / l0) ** 4 * gamma * h0 ** 3 / (3 * eta)
    return 1 / (part_1 + part_2)


def get_tau_n_easy(n, eta, gamma, h0, l0):
    if n == 0:
        return np.inf
    return 3 * eta / (gamma * h0 ** 3) * (l0 / (2 * np.pi * n)) ** 4


def get_tau_n_array(eta, gamma, h0, l0, N=1000):
    tau_n_array = np.zeros(1000)
    for n in range(N):
        tau_n_array[n] = get_tau_n_easy(n, eta, gamma, h0, l0)
    return tau_n_array


def get_An(func, n, l0):
    def get_Y(x):
        return func(x) * np.cos(2 * np.pi * n * x / l0)

    return 2 / l0 * quad(get_Y, -l0 / 2, l0 / 2)[0]


def get_An_array(xx, zz, l0, N=1000):
    def func(x):
        return mf.lin_lin_interp(xx, zz)(x)

    An_array = np.zeros(N)
    An_array[0] = get_An(func, 0, l0) / 2
    return An_array


def get_h_at_t(xx, An_array, tau_n_array, l0, t):
    result = np.zeros(len(xx))
    result += An_array[0]
    for n in range(1, len(An_array)):
        result += An_array[n] * np.exp(-t / tau_n_array[n]) * np.cos(2 * np.pi * n * xx / l0)
    return result
