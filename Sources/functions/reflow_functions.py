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
sci_500 = np.load('notebooks/viscosity/final_1/sci_500.npy') * 8
sci_700 = np.load('notebooks/viscosity/final_1/sci_700.npy') * 8
sci_1000 = np.load('notebooks/viscosity/final_1/sci_1000.npy') * 8
sci_1500 = np.load('notebooks/viscosity/final_1/sci_1500.npy') * 8

sci_500[0] = 0
sci_700[0] = 0
sci_1000[0] = 0
sci_1500[0] = 0

mobs_120_500 = np.load('notebooks/viscosity/final_1/mobs_120/mobs_500.npy')
mobs_120_700 = np.load('notebooks/viscosity/final_1/mobs_120/mobs_700.npy')
mobs_120_1000 = np.load('notebooks/viscosity/final_1/mobs_120/mobs_1000.npy')
mobs_120_1500 = np.load('notebooks/viscosity/final_1/mobs_120/mobs_1500.npy')

mobs_140_500 = np.load('notebooks/viscosity/final_1/mobs_140/mobs_500.npy')
mobs_140_700 = np.load('notebooks/viscosity/final_1/mobs_140/mobs_700.npy')
mobs_140_1000 = np.load('notebooks/viscosity/final_1/mobs_140/mobs_1000.npy')
mobs_140_1500 = np.load('notebooks/viscosity/final_1/mobs_140/mobs_1500.npy')

mobs_160_500 = np.load('notebooks/viscosity/final_1/mobs_160/mobs_500.npy')
mobs_160_700 = np.load('notebooks/viscosity/final_1/mobs_160/mobs_700.npy')
mobs_160_1000 = np.load('notebooks/viscosity/final_1/mobs_160/mobs_1000.npy')
mobs_160_1500 = np.load('notebooks/viscosity/final_1/mobs_160/mobs_1500.npy')


# %%
def gauss(xx, a, b, c):
    return a + b * np.exp(-xx**2 / c**2)


def gauss_zero(xx, a, b):
    return a * np.exp(-xx**2 / b**2)


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


def get_fit_params_sci_arr(xx, sci_arr):
    p0 = [0, 5, 500]
    popt = curve_fit(gauss, xx, sci_arr, p0=p0)[0]
    # return gauss(xx, *popt)
    return popt


def move_sci_to_mobs(sci_arr, T_C, zip_len):

    if T_C == 120:
        if zip_len == 500:
            return mcf.lin_log_interp(sci_500, mobs_120_500)(sci_arr)
        elif zip_len == 700:
            return mcf.lin_log_interp(sci_700, mobs_120_700)(sci_arr)
        elif zip_len == 1000:
            return mcf.lin_log_interp(sci_1000, mobs_120_1000)(sci_arr)
        if zip_len == 1500:
            return mcf.lin_log_interp(sci_1500, mobs_120_1500)(sci_arr)
        else:
            print('specify correct zip length!')
            return mcf.lin_log_interp(sci_1500, mobs_120_1500)(sci_arr)

    elif T_C == 140:
        if zip_len == 500:
            return mcf.lin_log_interp(sci_500, mobs_140_500)(sci_arr)
        elif zip_len == 700:
            return mcf.lin_log_interp(sci_700, mobs_140_700)(sci_arr)
        elif zip_len == 1000:
            return mcf.lin_log_interp(sci_1000, mobs_140_1000)(sci_arr)
        if zip_len == 1500:
            return mcf.lin_log_interp(sci_1500, mobs_140_1500)(sci_arr)
        else:
            print('specify correct zip length!')
            return mcf.lin_log_interp(sci_1500, mobs_140_1500)(sci_arr)

    elif T_C == 140:
        if zip_len == 500:
            return mcf.lin_log_interp(sci_500, mobs_140_500)(sci_arr)
        elif zip_len == 700:
            return mcf.lin_log_interp(sci_700, mobs_140_700)(sci_arr)
        elif zip_len == 1000:
            return mcf.lin_log_interp(sci_1000, mobs_140_1000)(sci_arr)
        if zip_len == 1500:
            return mcf.lin_log_interp(sci_1500, mobs_140_1500)(sci_arr)
        else:
            print('specify correct zip length!')
            return mcf.lin_log_interp(sci_1500, mobs_140_1500)(sci_arr)

    else:
        print('specify correct temperature!')
        if zip_len == 500:
            return mcf.lin_log_interp(sci_500, mobs_140_500)(sci_arr)
        elif zip_len == 700:
            return mcf.lin_log_interp(sci_700, mobs_140_700)(sci_arr)
        elif zip_len == 1000:
            return mcf.lin_log_interp(sci_1000, mobs_140_1000)(sci_arr)
        if zip_len == 1500:
            return mcf.lin_log_interp(sci_1500, mobs_140_1500)(sci_arr)
        else:
            print('specify correct zip length!')
            return mcf.lin_log_interp(sci_1500, mobs_140_1500)(sci_arr)


# %%
# sci_arr = np.load('scission_array_100nm.npy')
# mobs = move_sci_to_mobs(sci_arr, 120, 1500)
#
# plt.figure(dpi=300)
# plt.semilogy(mapping.x_centers_100nm, mobs)
# plt.semilogy(mapping.x_centers_100nm, get_fitted_mobs(mapping.x_centers_100nm, sci_arr, 120, 1000))
# plt.show()
