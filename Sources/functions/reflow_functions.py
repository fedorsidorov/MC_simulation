import importlib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from functions import MC_functions as mcf
from scipy.optimize import curve_fit
from mapping import mapping_3p3um_80nm as mapping

mapping = importlib.reload(mapping)
mcf = importlib.reload(mcf)


# %%
# def gauss(xx, a, b, c):
#     return a + b * np.exp(-xx**2 / c**2)


# def gauss_zero(xx, a, b):
#     return a * np.exp(-xx**2 / b**2)


# def get_PMMA_surface_tension(T_C):  # wu1970.pdf
#     gamma_CGS = 41.1 - 0.076 * (T_C - 20)
#     gamma_SI = gamma_CGS * 1e-3
#     return gamma_SI


# def get_viscosity_PMMA_6N(T_C):  # aho2008.pdf
#     eta_0 = 13450
#     T0 = 200
#     C1 = 7.6682
#     C2 = 210.76
#     log_aT = -C1 * (T_C - T0) / (C2 + (T_C - T0))
#     eta = eta_0 * np.exp(log_aT)
#     return eta


def get_viscosity_experiment_const_M(T_C):  # aho2008.pdf
    eta_0 = 1.54e+6
    T0 = 200
    C1 = 7.6682
    C2 = 210.76
    log_aT = -C1 * (T_C - T0) / (C2 + (T_C - T0))
    eta = eta_0 * np.exp(log_aT)
    return eta


def get_SE_mobility(eta):
    C = 0.0381
    k = 0.9946
    time2scale = C * eta**k
    mobility = 1 / time2scale
    return mobility


# def move_Mw_to_mob(T_C, Mw):
#     eta = get_viscosity_W(T_C, Mw)
#     mob = get_SE_mobility(eta)
#     return mob


# def get_viscosity_W(T_C, Mw):  # aho2008.pdf, bueche1955.pdf - ???????????
#     Mw_0 = 9e+4
#     eta = get_viscosity_PMMA_6N(T_C)
#     eta_final = eta * (Mw / Mw_0)**3.4
#     return eta_final


def get_viscosity_experiment_Mn(T_C, Mn):  # aho2008.pdf, bueche1955.pdf - ???????????
    Mn_0 = 271374
    eta_pre = get_viscosity_experiment_const_M(T_C)
    eta_final = eta_pre * (Mn / Mn_0)**3.4
    return eta_final


# def get_fit_params_sci_arr(xx, sci_arr):
#     p0 = [0, 5, 500]
#     popt = curve_fit(gauss, xx, sci_arr, p0=p0)[0]
    # return popt


# %%
# sci_arr = np.load('scission_array_100nm.npy')
# mobs = move_sci_to_mobs(sci_arr, 120, 1500)
#
# plt.figure(dpi=300)
# plt.semilogy(mapping.x_centers_100nm, mobs)
# plt.semilogy(mapping.x_centers_100nm, get_fitted_mobs(mapping.x_centers_100nm, sci_arr, 120, 1000))
# plt.show()
