import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
from scipy import integrate
import constants as const
import grid as grid
from tqdm import tqdm

const = importlib.reload(const)
grid = importlib.reload(grid)


# %%
def get_E_q(E, q):
    return E + const.hbar**2 * q**2 / (2 * const.m) / const.eV


def get_OSC(hw, En, Gn, An):
    if hw < 3:
    # if hw < 4:
        return 0
    return An * Gn * hw / ((En**2 - hw**2)**2 + (Gn * hw)**2)


def get_ELF(hw, q):
    ELF = get_OSC(hw, get_E_q(19.46, q), 8.770, 100.0)
    ELF += get_OSC(hw, get_E_q(25.84, q), 14.75, 286.5)
    ELF += get_OSC(hw, get_E_q(300.0, q), 140.0, 80.0)
    ELF += get_OSC(hw, get_E_q(550.0, q), 300.0, 55.0)
    return ELF


# %%
OLF = np.zeros(len(grid.EE))

for i, e in enumerate(grid.EE):
    OLF[i] = get_ELF(e, 0)


# %%
# dapor_OLF = np.loadtxt('notebooks/Dapor_4osc/curves/OLF_fit_2015.txt')
#
# plt.figure(dpi=300)
# plt.loglog(grid.EE, OLF)
# plt.loglog(dapor_OLF[:, 0], dapor_OLF[:, 1], '--')
# plt.show()


# %%
def get_qm_qp(E, hw):
    qm = np.sqrt(2 * const.m) / const.hbar * (np.sqrt(E) - np.sqrt(E - hw))
    qp = np.sqrt(2 * const.m) / const.hbar * (np.sqrt(E) + np.sqrt(E - hw))
    return qm, qp


def get_DIIMFP(E_eV, hw_eV):
    if hw_eV > E_eV:
        return 0

    E = E_eV * const.eV
    hw = hw_eV * const.eV

    def get_Y(q):
        return get_ELF(hw_eV, q) / q

    qm, qp = get_qm_qp(E, hw)
    integral = integrate.quad(get_Y, qm, qp)[0]

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







