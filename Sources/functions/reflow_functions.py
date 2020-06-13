import importlib
from tqdm import tqdm
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from functions import MC_functions as mf

mf = importlib.reload(mf)


# %%
# PMMA_950K_viscosity = np.loadtxt('data/reflow/PMMA_996K_viscosity.txt')


def get_PMMA_950K_viscosity(T_C):  # hirai2003.pdf
    # if T_C < PMMA_950K_viscosity[0, 0]:
    #     return PMMA_950K_viscosity[0, 1]
    # elif T_C > PMMA_950K_viscosity[0, -1]:
    #     return PMMA_950K_viscosity[-1, 1]
    # else:
    #     return mf.lin_log_interp(PMMA_950K_viscosity[:, 0], PMMA_950K_viscosity[:, 1])(T_C)
    return 5e+6


def get_PMMA_surface_tension(T_C):  # wu1970.pdf
    gamma_CGS = 41.1 - 0.076 * (T_C - 20)
    gamma_SI = gamma_CGS * 1e-3
    return gamma_SI


# def get_PMMA_viscosity(T_C):  # jones2006.pdf - ???
#     T0 = 86  # C
#     C1 = 70.1
#     C2 = -12.21
#     C2 = 12.21
#     eta0 = 1e+12  # Pa s
#     eta = eta0 * np.exp(C1 * (T_C - T0) / (C2 + (T_C - T0)))
#     eta = eta0 * np.exp(-C1 * (T_C - T0) / (C2 + (T_C - T0)))
#     return eta


# TT = np.linspace(80, 160, 100)
# etas = np.zeros(len(TT))
#
# for i, T in enumerate(TT):
#     etas[i] = get_PMMA_950K_viscosity(T)
#
# plt.figure(dpi=300)
# plt.semilogy(TT, etas)
# plt.show()


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


def get_An_array(xx, zz, l0, N):
    def func(x):
        return mf.lin_lin_interp(xx, zz)(x)

    An_array = np.zeros(N)

    An_array[0] = get_An(func, 0, l0) / 2

    progress_bar = tqdm(total=N, position=0)

    for n in range(1, N):
        An_array[n] = get_An(func, n, l0)
        progress_bar.update()

    return An_array


def get_h_at_t(xx, An_array, tau_n_array, l0, t):
    result = np.zeros(len(xx))
    result += An_array[0]
    for n in range(1, len(An_array)):
        result += An_array[n] * np.exp(-t / tau_n_array[n]) * np.cos(2 * np.pi * n * xx / l0)
    return result
