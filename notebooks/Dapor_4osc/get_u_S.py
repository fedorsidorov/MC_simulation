import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import importlib
from scipy import integrate
import constants as const
import grid as grid
from tqdm import tqdm

const = importlib.reload(const)
grid = importlib.reload(grid)

# OLF_PMMA = np.loadtxt('notebooks/_outdated/OLF_PMMA/PMMA_OLF.txt')
OLF_PMMA = np.load('notebooks/_outdated/OLF_PMMA/Ritsko_Henke_Dapor_Im.npy')

EE_Si = np.load('notebooks/Akkerman_Si_5osc/OLF_Palik+Fano/EE_Palik+Fano.npy')[1179:4241]
OLF_Si = np.load('notebooks/Akkerman_Si_5osc/OLF_Palik+Fano/OLF_Palik+Fano.npy')[1179:4241]

# %%
with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    # ax.loglog(OLF_PMMA[:, 0], OLF_PMMA[:, 1], label=r'ПММА')
    ax.loglog(grid.EE, OLF_PMMA, label=r'ПММА')
    # ax.loglog(EE_Si, OLF_Si, 'C3', label=r'Si')
    ax.loglog(EE_Si, OLF_Si, label=r'Si')

    ax.legend(loc=1, fontsize=6)
    ax.set(xlabel=r'$E$, эВ')
    ax.set(ylabel=r'Im $\left [ \frac{-1}{\varepsilon (0, \omega)} \right ]$')
    ax.autoscale(tight=True)
    ax.text(30, 2, r'a)')

    plt.xlim(1e+1, 1e+4)
    plt.ylim(1e-7, 1e+1)

    plt.show()
    # fig.savefig('review_figures/OLF_a_new.jpg', dpi=600)


# %%
En_arr = [19.46, 25.84, 300.0, 550.0]
Gn_arr = [8.770, 14.75, 140.0, 300.0]
An_arr = [100.0, 286.5, 80.0, 55.0]


def get_E_q(E, q):
    return E + const.hbar**2 * q**2 / (2 * const.m) / const.eV


def get_OSC(hw, En, Gn, An):
    # if hw < 4:
    if hw < 3:
        return 0
    return An * Gn * hw / ((En**2 - hw**2)**2 + (Gn * hw)**2)


def get_ELF(hw, q):
    # ELF = get_OSC(hw, get_E_q(19.46, q), 8.770, 100.0)
    # ELF += get_OSC(hw, get_E_q(25.84, q), 14.75, 286.5)
    # ELF += get_OSC(hw, get_E_q(300.0, q), 140.0, 80.0)
    # ELF += get_OSC(hw, get_E_q(550.0, q), 300.0, 55.0)
    ELF = get_OSC(hw, get_E_q(En_arr[0], q), Gn_arr[0], An_arr[0])
    ELF += get_OSC(hw, get_E_q(En_arr[1], q), Gn_arr[1], An_arr[1])
    ELF += get_OSC(hw, get_E_q(En_arr[2], q), Gn_arr[2], An_arr[2])
    ELF += get_OSC(hw, get_E_q(En_arr[3], q), Gn_arr[3], An_arr[3])
    return ELF


# %%
osc_0 = np.zeros(len(grid.EE))
osc_1 = np.zeros(len(grid.EE))
osc_2 = np.zeros(len(grid.EE))
osc_3 = np.zeros(len(grid.EE))

OLF = np.zeros(len(grid.EE))


for i, e in enumerate(grid.EE):
    osc_0[i] = get_OSC(e, En_arr[0], Gn_arr[0], An_arr[0])
    osc_1[i] = get_OSC(e, En_arr[1], Gn_arr[1], An_arr[1])
    osc_2[i] = get_OSC(e, En_arr[2], Gn_arr[2], An_arr[2])
    osc_3[i] = get_OSC(e, En_arr[3], Gn_arr[3], An_arr[3])

    OLF[i] = get_ELF(e, 0)


# %%
with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.loglog(grid.EE, OLF_PMMA, 'C0', label=r'OLF ПММА')
    ax.loglog(grid.EE, OLF, 'C3', label=r'сумма осцилляторов')
    ax.loglog(grid.EE, osc_0, '--', linewidth=0.7, color='C1', label=r'осц. 1')
    ax.loglog(grid.EE, osc_1, '--', linewidth=0.7, color='C2', label=r'осц. 2')
    ax.loglog(grid.EE, osc_2, '--', linewidth=0.7, color='C3', label=r'осц. 3')
    ax.loglog(grid.EE, osc_3, '--', linewidth=0.7, color='C4', label=r'осц. 4')

    ax.legend(loc=1, fontsize=6)
    ax.set(xlabel=r'$E$, эВ')
    ax.set(ylabel=r'Im $\left [ \frac{-1}{\varepsilon (0, \omega)} \right ]$')
    ax.autoscale(tight=True)
    ax.text(30, 2, r'б)')

    plt.xlim(1e+1, 1e+4)
    plt.ylim(1e-7, 1e+1)

    plt.show()
    fig.savefig('review_figures/OLF_b_new.jpg', dpi=600)


# %% plot ELF
n_bins = 50
xx = np.linspace(0, 100, n_bins)
yy = np.linspace(0, 5, n_bins)
XX, YY = np.meshgrid(xx, yy)
ZZ = np.zeros((len(xx), len(yy)))

for i in range(len(xx)):
    for j in range(len(yy)):
        ZZ[i, j] = get_ELF(xx[i], yy[j] * const.P_au / const.hbar)

with plt.style.context(['science', 'russian-font']):
    factor = 1.2
    fig = plt.figure(figsize=[6.4*factor, 4.8*factor * 1.1], dpi=600)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.viridis)
    ax.view_init(30, 40)
    # ax.view_init(0, 0)

    fontsize = 15.8
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 1.4)

    ax.set_xlabel(r'$E$, эВ', fontsize=fontsize)
    ax.set_ylabel(r'$q$, а.е.', fontsize=fontsize)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'Im $\left [ \frac{-1}{\varepsilon (q, \omega)} \right ]$', fontsize=fontsize, rotation=90)

    plt.show()
    fig.savefig('review_figures/ELF_Drude_15p8.jpg', bbox_inches='tight', pad_inches=0)


# %%
def get_km_kp(E, hw):
    km = np.sqrt(2 * const.m) / const.hbar * (np.sqrt(E) - np.sqrt(E - hw))
    kp = np.sqrt(2 * const.m) / const.hbar * (np.sqrt(E) + np.sqrt(E - hw))
    return km, kp


def get_DIIMFP(E_eV, hw_eV):
    if hw_eV > E_eV:
        return 0

    E = E_eV * const.eV
    hw = hw_eV * const.eV

    def get_Y(q):
        return get_ELF(hw_eV, q) / q

    km, kp = get_km_kp(E, hw)
    integral = integrate.quad(get_Y, km, kp)[0]

    du_dE = 1 / (np.pi * const.a0 * E_eV) * integral  # cm^-1 * eV^-1
    return du_dE * 1e-7  # nm^-1 * eV^-1


def get_u(E_eV):

    E_b = 0

    def get_Y(hw_eV):
        return get_DIIMFP(E_eV, hw_eV)

    return integrate.quad(get_Y, E_b, E_eV / 2)


def get_S(E_eV):

    E_b = 0

    def get_Y(hw_eV):
        return get_DIIMFP(E_eV, hw_eV) * hw_eV

    return integrate.quad(get_Y, E_b, E_eV / 2)


# %%
# EE_sparse = grid.EE[::5]
EE_sparse = grid.EE

u = np.zeros(len(EE_sparse))
S = np.zeros(len(EE_sparse))

progress_bar = tqdm(total=len(EE_sparse), position=0)

for i, e in enumerate(EE_sparse):
    u[i] = get_u(e)[0]
    S[i] = get_S(e)[0]
    progress_bar.update()

# %%
dapor_u = np.loadtxt('notebooks/_outdated/Dapor_4osc/curves/u_2015.txt')

plt.figure(dpi=300)
plt.semilogx(EE_sparse, 1/(u * 1e-1))
plt.semilogx(dapor_u[:, 0], dapor_u[:, 1], '--')

plt.ylim(0, 200)

plt.grid()
plt.show()

# %%
dapor_S = np.loadtxt('notebooks/_outdated/Dapor_4osc/curves/S_2015.txt')

plt.figure(dpi=300)
plt.semilogx(EE_sparse, S / 1e+1)
plt.semilogx(dapor_S[:, 0], dapor_S[:, 1], '--')

plt.ylim(0, 4)

plt.show()







