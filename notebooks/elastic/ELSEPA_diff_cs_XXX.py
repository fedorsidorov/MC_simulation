import numpy as np
import matplotlib.pyplot as plt
import importlib
import constants as const

import grid as grid
grid = importlib.reload(grid)


# %%
def get_Ruth_diff_cs(Z, E):

    alpha = const.m * const.e**4 * np.pi**2 * Z**(2/3) / (const.h**2 * E * const.eV)
    diff_cs = Z**2 * const.e**4 / (4 * (E * const.eV)**2 * (1 - np.cos(grid.THETA_rad) + alpha)**2)

    return diff_cs


dcs_ruth_Hg = get_Ruth_diff_cs(80, 300)

hootcamp_cs = np.loadtxt('notebooks/elastic/curves/Hootcamp_Hg_300eV/Hootcamp_Hg_300eV.txt')

i = 563


# %% X10
arr_110 = np.load('notebooks/elastic/final_arrays/root_Hg/X10/root_110_diff_cs.npy')
arr_210 = np.load('notebooks/elastic/final_arrays/root_Hg/X10/root_210_diff_cs.npy')
arr_310 = np.load('notebooks/elastic/final_arrays/root_Hg/X10/root_310_diff_cs.npy')
arr_410 = np.load('notebooks/elastic/final_arrays/root_Hg/X10/root_410_diff_cs.npy')


with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.semilogy(grid.THETA_deg, arr_110[i, :] * 1e+16, label=r'ТФМ')
    ax.semilogy(grid.THETA_deg, arr_210[i, :] * 1e+16, label=r'ТФД')
    ax.semilogy(grid.THETA_deg, arr_310[i, :] * 1e+16, label=r'ДХФС')
    ax.semilogy(grid.THETA_deg, arr_410[i, :] * 1e+16, label=r'ДФ')

    ax.semilogy(grid.THETA_deg, dcs_ruth_Hg * 1e+16, label=r'Резерфорд')
    ax.semilogy(hootcamp_cs[:, 0], hootcamp_cs[:, 1], 'k+', markersize=4, label=r'эксперимент')

    ax.legend(loc=1, fontsize=6)
    ax.set(xlabel=r'$\theta$, град')
    ax.set(ylabel=r'$\frac{d \sigma}{d \Omega}$, Å$^2$/ср')
    ax.autoscale(tight=True)
    ax.text(10, 1e+2 * 0.4, r'a)')

    plt.xlim(0, 180)
    plt.ylim(1e-3, 1e+2)

    plt.show()
    fig.savefig('review_figures/dcs_models_X10.jpg', dpi=600)


# %% 4X0
# arr_400 = np.load('notebooks/elastic/final_arrays/root_Hg/4X0/root_400_diff_cs.npy')
arr_410 = np.load('notebooks/elastic/final_arrays/root_Hg/4X0/root_410_diff_cs.npy')
arr_420 = np.load('notebooks/elastic/final_arrays/root_Hg/4X0/root_420_diff_cs.npy')
arr_430 = np.load('notebooks/elastic/final_arrays/root_Hg/4X0/root_430_diff_cs.npy')

plt.figure(dpi=300)


with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    # ax.semilogy(grid.THETA_deg, arr_400[i, :] * 1e+16, label=r'ТФМ')
    ax.semilogy(grid.THETA_deg, arr_410[i, :] * 1e+16, label=r'ФМ')
    ax.semilogy(grid.THETA_deg, arr_420[i, :] * 1e+16, label=r'ТФ')
    ax.semilogy(grid.THETA_deg, arr_430[i, :] * 1e+16, label=r'РТ')

    ax.semilogy(grid.THETA_deg, dcs_ruth_Hg * 1e+16, 'C4', label=r'Резерфорд')
    ax.semilogy(hootcamp_cs[:, 0], hootcamp_cs[:, 1], 'k+', markersize=4, label=r'эксперимент')

    ax.legend(loc=1, fontsize=6)
    ax.set(xlabel=r'$\theta$, град')
    ax.set(ylabel=r'$\frac{d \sigma}{d \Omega}$, Å$^2$/ср')
    ax.autoscale(tight=True)
    ax.text(10, 1e+2 * 0.4, r'a)')

    plt.xlim(0, 180)
    plt.ylim(1e-3, 1e+2)

    plt.show()
    # fig.savefig('review_figures/dcs_models_4X0.jpg', dpi=600)


# %% 41X
arr_410 = np.load('notebooks/elastic/final_arrays/root_Hg/41X/root_410_diff_cs.npy')
arr_411 = np.load('notebooks/elastic/final_arrays/root_Hg/41X/root_411_diff_cs.npy')
arr_412 = np.load('notebooks/elastic/final_arrays/root_Hg/41X/root_412_diff_cs.npy')

plt.figure(dpi=300)


with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.semilogy(grid.THETA_deg, arr_410[i, :] * 1e+16, label=r'нет модели')
    ax.semilogy(grid.THETA_deg, arr_411[i, :] * 1e+16, label=r'Б')
    ax.semilogy(grid.THETA_deg, arr_412[i, :] * 1e+16, label=r'ПЛП')

    ax.semilogy(grid.THETA_deg, dcs_ruth_Hg * 1e+16, 'C4', label=r'Резерфорд')
    ax.semilogy(hootcamp_cs[:, 0], hootcamp_cs[:, 1], 'k+', markersize=4, label=r'эксперимент')

    ax.legend(loc=1, fontsize=6)
    ax.set(xlabel=r'$\theta$, град')
    ax.set(ylabel=r'$\frac{d \sigma}{d \Omega}$, Å$^2$/ср')
    ax.autoscale(tight=True)
    ax.text(10, 1e+2 * 0.4, r'a)')

    plt.xlim(0, 180)
    plt.ylim(1e-3, 1e+2)

    plt.show()
    fig.savefig('review_figures/dcs_models_41X.jpg', dpi=600)


