import numpy as np
import matplotlib.pyplot as plt
import importlib
import constants as const

import grid as grid
grid = importlib.reload(grid)


# %%
def get_Ruth_cs(Z):
    alpha = const.m * const.e**4 * np.pi**2 * Z**(2/3) / (const.h**2 * grid.EE * const.eV)
    cs = np.pi * Z**2 * const.e**4 / ((grid.EE * const.eV)**2 * alpha * (alpha + 2))
    return cs


cs_ruth_Hg = get_Ruth_cs(80)


# %% X10
arr_110 = np.load('notebooks/elastic/final_arrays/root_Hg/X10/root_110_cs.npy')
arr_210 = np.load('notebooks/elastic/final_arrays/root_Hg/X10/root_210_cs.npy')
arr_310 = np.load('notebooks/elastic/final_arrays/root_Hg/X10/root_310_cs.npy')
arr_410 = np.load('notebooks/elastic/final_arrays/root_Hg/X10/root_410_cs.npy')

plt.show()

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.loglog(grid.EE, arr_110 * 1e+16, label=r'ТФМ')
    ax.loglog(grid.EE, arr_210 * 1e+16, label=r'ТФД')
    ax.loglog(grid.EE, arr_310 * 1e+16, label=r'ДХФС', color='C3')
    ax.loglog(grid.EE, arr_410 * 1e+16, label=r'ДФ', color='C2')

    ax.loglog(grid.EE, cs_ruth_Hg * 1e+16, label=r'Резерфорд', color='C4')

    ax.legend(loc=1, fontsize=6)
    ax.set(xlabel=r'$E$, эВ')
    ax.set(ylabel=r'$\sigma$, Å$^2$')
    ax.autoscale(tight=True)
    # ax.text(13, 4e+2, r'б)')
    ax.text(260, 470, r'(б)')

    plt.xlim(10, 1e+4)
    plt.ylim(1e-1, 1e+3)

    fig.savefig('review_figures/cs_models_X10_FINAL.jpg', dpi=600)
    plt.show()


# %% 4X0
arr_410 = np.load('notebooks/elastic/final_arrays/root_Hg/4X0/root_410_cs.npy')
arr_420 = np.load('notebooks/elastic/final_arrays/root_Hg/4X0/root_420_cs.npy')
arr_430 = np.load('notebooks/elastic/final_arrays/root_Hg/4X0/root_430_cs.npy')

plt.figure(dpi=300)


with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.loglog(grid.EE, arr_410 * 1e+16, label=r'ФМ')
    ax.loglog(grid.EE, arr_420 * 1e+16, label=r'ТФ')
    ax.loglog(grid.EE, arr_430 * 1e+16, label=r'РТ', color='C3')

    ax.loglog(grid.EE, cs_ruth_Hg * 1e+16, 'C4', label=r'Резерфорд')

    ax.legend(loc=1, fontsize=6)
    ax.set(xlabel=r'$E$, эВ')
    ax.set(ylabel=r'$\sigma$, Å$^2$')
    ax.autoscale(tight=True)
    # ax.text(13, 4e+2, r'б)')
    ax.text(260, 470, r'(б)')

    plt.xlim(10, 1e+4)
    plt.ylim(1e-1, 1e+3)

    fig.savefig('review_figures/cs_models_4X0_FINAL.jpg', dpi=600)
    plt.show()


# %% 41X
arr_410 = np.load('notebooks/elastic/final_arrays/root_Hg/41X/root_410_cs.npy')
arr_411 = np.load('notebooks/elastic/final_arrays/root_Hg/41X/root_411_cs.npy')
arr_412 = np.load('notebooks/elastic/final_arrays/root_Hg/41X/root_412_cs.npy')

plt.figure(dpi=300)


with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.loglog(grid.EE, arr_410 * 1e+16, label=r'нет модели')
    ax.loglog(grid.EE, arr_411 * 1e+16, label=r'Б')
    ax.loglog(grid.EE, arr_412 * 1e+16, label=r'ЛДА', color='C3')

    ax.loglog(grid.EE, cs_ruth_Hg * 1e+16, 'C4', label=r'Резерфорд')

    ax.legend(loc=1, fontsize=6)
    ax.set(xlabel=r'$E$, эВ')
    ax.set(ylabel=r'$\sigma$, Å$^2$')
    ax.autoscale(tight=True)
    # ax.text(13, 4e+2, r'б)')
    ax.text(260, 470, r'(б)')

    plt.xlim(10, 1e+4)
    plt.ylim(1e-1, 1e+3)

    plt.show()
    fig.savefig('review_figures/cs_models_41X_FINAL.jpg', dpi=600)
