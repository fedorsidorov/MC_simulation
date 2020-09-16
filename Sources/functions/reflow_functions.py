import importlib
from tqdm import tqdm
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from functions import MC_functions as mf

mf = importlib.reload(mf)


# %%
def get_PMMA_surface_tension(T_C):  # wu1970.pdf
    gamma_CGS = 41.1 - 0.076 * (T_C - 20)
    gamma_SI = gamma_CGS * 1e-3
    return gamma_SI


def get_viscosity_PMMA_6N(T_C):  # aho2008.pdf
    eta_0 = 13450
    T0 = 200
    C1 = 7.6682
    C2 = 210.76
    log_aT = -C1 * (T_C - T0) / (C2 + (T_C - T0))
    eta = eta_0 * np.exp(log_aT)
    return eta


def get_viscosity_W(T_C, Mw):  # aho2008.pdf, bueche1955.pdf
    Mw_0 = 9e+4
    eta = get_viscosity_PMMA_6N(T_C)
    eta_final = eta * (Mw / Mw_0)**3.4
    return eta_final


def get_SE_mobility(eta):
    k = 1.0043796246664092
    b = -3.8288826121816815
    time2scale = np.exp(np.log(eta) * k + b)
    mobility = 1/time2scale
    return mobility

# temp = np.linspace(120, 170)
#
# plt.figure(dpi=300)
# plt.semilogy(temp, get_viscosity_W(temp, 669e+3))
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
    # return 2 / l0 * quad(get_Y, 0, l0)[0]


def get_Bn(func, n, l0):
    def get_Y(x):
        return func(x) * np.sin(2 * np.pi * n * x / l0)
    return 2 / l0 * quad(get_Y, -l0 / 2, l0 / 2)[0]
    # return 2 / l0 * quad(get_Y, 0, l0)[0]


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


def get_Bn_array(xx, zz, l0, N):
    def func(x):
        return mf.lin_lin_interp(xx, zz)(x)

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
