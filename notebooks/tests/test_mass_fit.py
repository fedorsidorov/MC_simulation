import importlib
import numpy as np
from mapping import mapping_3p3um_80nm as mapping
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functions import diffusion_functions as df
from functions import reflow_functions as rf

mapping = importlib.reload(mapping)
df = importlib.reload(df)
rf = importlib.reload(rf)


# %%
def exp_gauss(xx, A, B, s):
    return np.exp(A - B*np.exp(-xx**2 / s**2))


def gauss(xx, A, B, s):
    return A - B*np.exp(-xx**2 / s**2)


xx = mapping.x_centers_100nm

for i in range(12):
    Mw = np.load('matrix_Mw_1d_100nm_' + str(i) + '.npy')

    # beg_ind = 10
    beg_ind = 1
    popt, _ = curve_fit(exp_gauss, xx[beg_ind:-beg_ind], Mw[beg_ind:-beg_ind], p0=[13.8, 0.6, 200])

    xx_fit = mapping.x_centers_25nm
    Mw_fit = exp_gauss(xx_fit, *popt)

    plt.figure(dpi=300)
    plt.semilogy(xx[beg_ind:-beg_ind], Mw[beg_ind:-beg_ind], 'o-')
    # plt.semilogy(xx_fit, exp_gauss(xx_fit, 13.8, 0.6, 200))
    plt.semilogy(xx_fit, exp_gauss(xx_fit, *popt))

    # plt.semilogy(xx[beg_ind:-beg_ind], Mw[beg_ind:-beg_ind], 'o-')
    # plt.semilogy(xx_fit, exp_gauss(xx_fit, 13.8, 0.6, 200))

    # plt.semilogy(xx_fit, Mw_fit)
    plt.show()

    etas_50nm_fit = rf.get_viscosity_W(125, exp_gauss(mapping.x_centers_50nm, *popt))
    mobs_50nm_fit = rf.get_SE_mobility(etas_50nm_fit)

    plt.figure(dpi=300)
    plt.semilogy(mapping.x_centers_50nm, etas_50nm_fit)
    plt.show()
