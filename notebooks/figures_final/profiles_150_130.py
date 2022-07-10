import importlib
import numpy as np
import matplotlib.pyplot as plt
from mapping import mapping_3um_500nm as mm

mm = importlib.reload(mm)


# %% 150 C
xx_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/xx_366_zero.npy')
zz_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/zz_366_zero.npy')

pr_150 = np.loadtxt('/Volumes/Transcend/SIM_DEBER/150.txt')

xx = pr_150[:, 0]
zz = pr_150[:, 1]


with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.plot(xx, zz, 'tab:blue', label=r'SE surface')
    ax.plot(xx, zz, 'r--', linewidth=2, label=r'simulation')
    ax.plot(xx_366, zz_366 + 75, color='tab:purple', label=r'experiment')
    ax.plot(xx_366, zz_366 + 100, color='tab:purple')
    ax.plot(xx_366, np.ones(len(zz_366)) * 500, 'k', label=r'initial surface')

    ax.set(title=r'150$^\circ$C, I=1.2 nA, $\sigma$=250 nm, t=100 s')
    ax.set(xlabel=r'x, nm')
    ax.set(ylabel=r'y, nm')
    ax.legend(fontsize=7, loc='upper right')

    plt.xlim(-1500, 1500)
    plt.ylim(0, 600)

    plt.savefig('figures_final/profile_150.jpg', dpi=600)
    plt.show()


# %% 130 C
xx_360 = np.load('notebooks/DEBER_simulation/exp_profiles/360/xx_360.npy')
zz_360 = np.load('notebooks/DEBER_simulation/exp_profiles/360/zz_360.npy')

pr_130 = np.loadtxt('/Volumes/Transcend/SIM_DEBER/130_surface.txt')
pr_130_inner = np.loadtxt('/Volumes/Transcend/SIM_DEBER/130_inner.txt')

xx_s = pr_130[:, 0]
zz_s = pr_130[:, 1]

xx_i = pr_130_inner[:, 0]
zz_i = pr_130_inner[:, 1]


with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.plot(xx_i, zz_i, 'tab:blue', label=r'SE surface')
    ax.plot(xx_s, zz_s, 'r', linewidth=2, label=r'simulation')
    ax.plot(xx_360, zz_360, '--', color='tab:purple', linewidth=2, label=r'experiment')
    ax.plot(xx_360, np.ones(len(zz_360)) * 500, 'k', label=r'initial surface')

    ax.set(title=r'130$^\circ$C, I=1.25 nA, $\sigma$=400 nm, t=100 s')
    ax.set(xlabel=r'x, nm')
    ax.set(ylabel=r'y, nm')
    ax.legend(fontsize=7, loc='upper right')

    plt.xlim(-1500, 1500)
    plt.ylim(0, 800)

    plt.savefig('figures_final/profile_130.jpg', dpi=600)
    plt.show()












