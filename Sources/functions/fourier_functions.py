import importlib
from tqdm import tqdm
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
# from functions import MC_functions as mcf

# mcf = importlib.reload(mcf)


# %%
def get_tau_n(n, eta, gamma, h0, l0):
    A = 1e-19
    if n == 0:
        return np.inf
    part_1 = (n * 2 * np.pi / l0) ** 2 * A / (6 * np.pi * h0 * eta)
    part_2 = (n * 2 * np.pi / l0) ** 4 * gamma * h0 ** 3 / (3 * eta)
    return 1 / (part_1 + part_2)


def get_tau_n_easy(n, eta, gamma, h0, l0):
    if n == 0:
        return np.inf
    return 3 * eta / (gamma * h0 ** 3) * (l0 / (2 * np.pi * n)) ** 4


def get_tau_n_array(eta, gamma, h0, l0, N):
    tau_n_array = np.zeros(N)
    for n in range(N):
        tau_n_array[n] = get_tau_n(n, eta, gamma, h0, l0)
    return tau_n_array


def get_tau_n_easy_array(eta, gamma, h0, l0, N):
    tau_n_array = np.zeros(N)
    for n in range(N):
        tau_n_array[n] = get_tau_n_easy(n, eta, gamma, h0, l0)
    return tau_n_array


def get_An(func, n, l0):
    def get_Y(x):
        return func(x) * np.cos(2 * np.pi * n * x / l0)
    return 2 / l0 * quad(get_Y, -l0 / 2, l0 / 2)[0]
    # return 2 / l0 * quad(get_Y, 0, l0)[0]


def get_Bn(func, n, l0):
    def get_Y(x):
        return func(x) * np.sin(2 * np.pi * n * x / l0)
    return 2 / l0 * quad(get_Y, -l0 / 2, l0 / 2)[0]
    # return 2 / l0 * quad(get_Y, 0, l0)[0]


def get_An_array(xx, zz, l0, N):
    def func(x):
        return mcf.lin_lin_interp(xx, zz)(x)

    An_array = np.zeros(N)
    An_array[0] = get_An(func, 0, l0) / 2

    progress_bar = tqdm(total=N, position=0)

    for n in range(1, N):
        An_array[n] = get_An(func, n, l0)
        progress_bar.update()

    return An_array


def get_Bn_array(xx, zz, l0, N):
    def func(x):
        return mcf.lin_lin_interp(xx, zz)(x)

    Bn_array = np.zeros(N)

    progress_bar = tqdm(total=N, position=0)

    for n in range(1, N):
        Bn_array[n] = get_Bn(func, n, l0)
        progress_bar.update()

    return Bn_array


def get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, t):
    result = np.zeros(len(xx))
    result += An_array[0]
    for n in range(1, len(An_array)):
        result += An_array[n] * np.exp(-t / tau_n_array[n]) * np.cos(2 * np.pi * n * xx / l0) + \
                  Bn_array[n] * np.exp(-t / tau_n_array[n]) * np.sin(2 * np.pi * n * xx / l0)
    return result
