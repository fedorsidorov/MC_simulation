import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %%
G_k = -454.01242092837134
G_b = 2.0079453212345793


def linear_func(xx, k, b):
    return k * xx + b


# def fit_gamma_G_value():
#     data = np.loadtxt('data/G_values/gamma.txt')


def fit_e_G_value():
    TT_C = np.array((-78, 0, 20, 100))
    TT_inv = 1 / (TT_C + 273)
    GG = np.array((0.74, 1.4, 1.5, 2.3))
    GG_log = np.log(GG)
    popt, pcov = curve_fit(linear_func, TT_inv, GG_log)

    # plt.figure(dpi=300)
    # plt.plot(TT_inv * 1e+3, GG_log, 'o')
    # TT_test_inv = 1 / (np.linspace(-200, 10000, 1000) + 273)
    # plt.plot(TT_test_inv * 1e+3, linear_func(TT_test_inv, *popt))
    # plt.show()

    return popt


def get_G(T_C):
    T_inv = 1 / (T_C + 273)
    return np.exp(linear_func(T_inv, G_k, G_b))


def get_T(G_value):
    T = G_k / (np.log(G_value) - G_b)
    return T - 273


# %%
# TT = np.linspace(0, 170, 100)
# GG = get_G(TT)
#
# plt.figure(dpi=300)
# plt.plot(TT, get_T(GG), 'o-')
# plt.grid()
# plt.show()
