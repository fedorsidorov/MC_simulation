import importlib
import matplotlib.pyplot as plt
import numpy as np
import grid

grid = importlib.reload(grid)


# %% u - Å
u_easiest = np.load('notebooks/Dapor_easiest/u_easiest.npy') * 1e-7 * 1e-1
u_4osc = np.load('notebooks/Dapor_4osc/u_4osc.py.npy') * 1e-1
u_mermin = np.load('notebooks/Dapor_PMMA_Mermin/u_Mermin_nm.npy') * 1e-1

inds = np.where(grid.EE >= 10)

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    # ax.loglog(grid.EE[inds], 1 / u_easiest[inds], label=r'L(x)')
    # ax.loglog(grid.EE[inds], 1 / u_4osc[inds], label=r'4 осциллятора')
    # ax.loglog(grid.EE[inds], 1 / u_mermin[inds], label=r'Мермин')

    ax.loglog(grid.EE[inds], u_easiest[inds], label=r'L(x)')
    ax.loglog(grid.EE[inds], u_4osc[inds], label=r'4 осциллятора')
    ax.loglog(grid.EE[inds], u_mermin[inds], color='C3', label=r'Мермин')

    ax.legend(loc=1, fontsize=6)
    ax.set(xlabel=r'$E$, эВ')
    ax.set(ylabel=r'$\lambda^{-1}$, Å$^{-1}$')
    ax.autoscale(tight=True)
    ax.text(13, 5e-1, r'a)')

    plt.xlim(1e+1, 1e+4)
    plt.ylim(1e-3, 1e+0)

    plt.show()
    # fig.savefig('review_figures/u-1_PMMA_new.jpg', dpi=600)


# %% S - Å
S_easiest = np.load('notebooks/Dapor_easiest/S_easiest.npy') * 1e-7 * 1e-1
S_4osc = np.load('notebooks/Dapor_4osc/S_4osc.py.npy') * 1e-1
S_mermin = np.load('notebooks/Dapor_PMMA_Mermin/S_Mermin_nm.npy') * 1e-1

inds = np.where(grid.EE >= 10)

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.semilogx(grid.EE[inds], S_easiest[inds], label=r'S(x)')
    ax.semilogx(grid.EE[inds], S_4osc[inds], label=r'4 осциллятора')
    ax.semilogx(grid.EE[inds], S_mermin[inds], color='C3', label=r'Мермин')

    ax.legend(loc=1, fontsize=6)
    ax.set(xlabel=r'$E$, эВ')
    ax.set(ylabel=r'$\frac{dE}{ds}$, эВ/Å')
    ax.autoscale(tight=True)
    ax.text(13, 4.5, r'б)')

    plt.xlim(1e+1, 1e+4)
    plt.ylim(0, 5)

    plt.show()
    fig.savefig('review_figures/S_PMMA_new.jpg', dpi=600)

