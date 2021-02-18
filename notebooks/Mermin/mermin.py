import importlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from tqdm import tqdm

import grid as grid
import constants as const
from functions import MC_functions as mcf

mcf = importlib.reload(mcf)
const = importlib.reload(const)
grid = importlib.reload(grid)

# %% parameters
PMMA_params = [  # [hwi, hgi, Ai] from devera2011.pdf
    [19.13, 9.03, 2.59e-1],
    [25.36, 14.34, 4.46e-1],
    [70.75, 48.98, 4.44e-3]
]

PMMA_hw_th = 2.99
inv_A = 1e+8  # cm^-1


# %% Lindhard and Mermin functions
def get_eps_L(q, hw_eV_complex, Epl_eV):  # gamma is energy!
    n = (Epl_eV * const.eV / const.hbar) ** 2 * const.m / (4 * np.pi * const.e ** 2)
    kF = (3 * np.pi ** 2 * n) ** (1 / 3)
    vF = const.hbar * kF / const.m
    qF = const.hbar * kF
    EF = const.m * vF ** 2 / 2
    z = q / (2 * qF)
    x = hw_eV_complex * const.eV / EF
    chi_2 = const.e ** 2 / (np.pi * const.hbar * vF)

    def f(xx, zz):
        res = 1 / 2 + 1 / (8 * zz) * (1 - (zz - xx / (4 * zz)) ** 2) * \
              np.log((zz - xx / (4 * zz) + 1) / (zz - xx / (4 * zz) - 1)) + \
              1 / (8 * zz) * (1 - (zz + xx / (4 * zz)) ** 2) * \
              np.log((zz + xx / (4 * zz) + 1) / (zz + xx / (4 * zz) - 1))
        return res

    return 1 + chi_2 / z ** 2 * f(x, z)


def get_eps_M(q, hw_eV_complex, E_pl_eV):
    hw_r = np.real(hw_eV_complex)
    gamma = np.imag(hw_eV_complex)

    num = (1 + 1j * gamma / hw_r) * \
          (get_eps_L(q, hw_eV_complex, E_pl_eV) - 1)

    den = 1 + (1j * gamma / hw_r) * (get_eps_L(q, hw_eV_complex, E_pl_eV) - 1) / (
            get_eps_L(q, 1e-100 + 1e-100j, E_pl_eV) - 1)

    return 1 + num / den


# %% 1 test ELF - OK
HW = np.linspace(1, 80, 100)

eps_M_0p1 = np.zeros(len(HW), dtype=complex)
eps_M_1p0 = np.zeros(len(HW), dtype=complex)
eps_M_10p = np.zeros(len(HW), dtype=complex)

for i, hw in enumerate(HW):
    eps_M_0p1[i] = get_eps_M(1.0 * const.P_au, hw + 0.1j, 20)
    eps_M_1p0[i] = get_eps_M(1.0 * const.P_au, hw + 1j, 20)
    eps_M_10p[i] = get_eps_M(1.0 * const.P_au, hw + 10j, 20)

plt.figure(dpi=300)
plt.plot(HW, np.imag(-1/eps_M_0p1))
plt.plot(HW, np.imag(-1/eps_M_1p0))
plt.plot(HW, np.imag(-1/eps_M_10p))

book = np.loadtxt('data/Dapor/mermin_book.txt')
plt.plot(book[:, 0], book[:, 1], '.')

plt.xlim(0, 80)
plt.ylim(0, 1.2)

plt.grid()
plt.show()

# %% 2 test ELF - OK
HW = np.linspace(1, 80, 100)

eps_L_0p5 = np.zeros(len(HW), dtype=complex)
eps_L_1p0 = np.zeros(len(HW), dtype=complex)
eps_L_1p5 = np.zeros(len(HW), dtype=complex)
eps_M_0p5 = np.zeros(len(HW), dtype=complex)
eps_M_1p0 = np.zeros(len(HW), dtype=complex)
eps_M_1p5 = np.zeros(len(HW), dtype=complex)

for i, hw in enumerate(HW):
    eps_L_0p5[i] = get_eps_L(0.5 * const.P_au, hw + 1e-100j, 20)
    eps_L_1p0[i] = get_eps_L(1.0 * const.P_au, hw + 1e-100j, 20)
    eps_L_1p5[i] = get_eps_L(1.5 * const.P_au, hw + 1e-100j, 20)
    eps_M_0p5[i] = get_eps_M(0.5 * const.P_au, hw + 5j, 20)
    eps_M_1p0[i] = get_eps_M(1.0 * const.P_au, hw + 5j, 20)
    eps_M_1p5[i] = get_eps_M(1.5 * const.P_au, hw + 5j, 20)

plt.figure(dpi=300)
plt.plot(HW, np.imag(-1 / eps_L_0p5))
plt.plot(HW, np.imag(-1 / eps_L_1p0))
plt.plot(HW, np.imag(-1 / eps_L_1p5))
plt.plot(HW, np.imag(-1 / eps_M_0p5))
plt.plot(HW, np.imag(-1 / eps_M_1p0))
plt.plot(HW, np.imag(-1 / eps_M_1p5))

book_L = np.loadtxt('data/Dapor/book_L.txt')
book_M = np.loadtxt('data/Dapor/book_M.txt')

plt.plot(book_L[:, 0], book_L[:, 1], '.')
plt.plot(book_M[:, 0], book_M[:, 1], '.')

plt.xlim(0, 80)
plt.ylim(0, 3)

plt.grid()
plt.show()


# %% PMMA Lindhard, Mermin and Drude ELF
def get_PMMA_ELF_L(q, hw_eV, params_hw_hg_A):
    PMMA_ELF_L = 0

    for line in params_hw_hg_A:
        E_pl_eV, _, A = line
        now_eps_L = get_eps_M(q, hw_eV + 1e-100j, E_pl_eV)

        PMMA_ELF_L += A * np.imag(-1 / now_eps_L)

    return PMMA_ELF_L


def get_PMMA_ELF(q, hw_eV, params_hw_hg_A, kind):
    PMMA_ELF = 0

    for line in params_hw_hg_A:

        E_pl_eV, gamma_eV, A = line

        if kind == 'L':
            gamma_eV = 1e-100

        elif kind == 'M':
            gamma_eV = gamma_eV

        else:
            print('Specify ELF kind!')
            return 0 + 0j

        now_eps = get_eps_M(q, hw_eV + 1j * gamma_eV, E_pl_eV)
        PMMA_ELF += A * np.imag(-1 / now_eps)

    return PMMA_ELF


def get_PMMA_OLF_D(hw_eV, params_hw_hg_A):
    PMMA_OLF_D = 0

    for line in params_hw_hg_A:
        E_pl_eV, hg_eV, A = line
        PMMA_OLF_D += A * E_pl_eV ** 2 * hg_eV * hw_eV / ((E_pl_eV ** 2 - hw_eV ** 2) ** 2 + (hg_eV * hw_eV) ** 2)

    return PMMA_OLF_D


# %% test PMMA OLF - OK
# EE = np.linspace(1, 100, 100)
EE = grid.EE
OLF_M = np.zeros(len(EE))
OLF_D = np.zeros(len(EE))

for i, E in enumerate(EE):
    OLF_M[i] = get_PMMA_ELF(5e-2 * inv_A * const.hbar, E, PMMA_params, kind='M')
    OLF_D[i] = get_PMMA_OLF_D(E, PMMA_params)

plt.figure(dpi=300)
plt.loglog(EE, OLF_M, '*', label='Mermin')
plt.loglog(EE, OLF_D, label='Drude')

ritsko = np.loadtxt('data/Dapor/Ritsko_dashed.txt')
plt.loglog(ritsko[:, 0], ritsko[:, 1], '.', label='Ritsko')

plt.xlim(1, 1e+4)
plt.ylim(1e-5, 1e+1)
plt.xlabel('E, eV')
plt.ylabel('PMMA OLF')
# plt.title('k = 5*10$^{-3}$ $\AA^{-1}$')
plt.title(r'k = 5*10$^{-2}$ $\AA^{-1}$')
plt.grid()
plt.legend()
plt.show()
# plt.savefig('PMMA_OLF_low_E_5e-2.png', dpi=300)

# %% test PMMA ELF - OK
EE = np.linspace(1, 80, 100)
ELF_2_M = np.zeros(len(EE))
ELF_4_M = np.zeros(len(EE))

for i, E in enumerate(EE):
    ELF_2_M[i] = get_PMMA_ELF(2 * inv_A * const.hbar, E, PMMA_params, 'M')
    ELF_4_M[i] = get_PMMA_ELF(4 * inv_A * const.hbar, E, PMMA_params, 'M')

plt.figure(dpi=300)
plt.plot(EE, ELF_2_M, label='my Mermin 2 A-1')
plt.plot(EE, ELF_4_M, label='my Mermin 4 A-1')

DM2 = np.loadtxt('data/Dapor/Dapor_M_2A-1.txt')
DM4 = np.loadtxt('data/Dapor/Dapor_M_4A-1.txt')
plt.plot(DM2[:, 0], DM2[:, 1], 'o')
plt.plot(DM4[:, 0], DM4[:, 1], 'o')

plt.xlim(0, 100)
plt.ylim(0, 0.5)

plt.legend()
plt.grid()

plt.show()


# %% calculate DIIMFP_prec
def get_PMMA_DIIMFP(E_eV, hw_eV, exchange=False):
    if hw_eV > E_eV:
        return 0

    E = E_eV * const.eV
    hw = hw_eV * const.eV

    def get_Y(k):

        if exchange:
            v = np.sqrt(E_eV * const.eV / const.m)
            frac = (const.hbar * k) / (const.m * v)
            return (1 + frac ** 4 - frac ** 2) * get_PMMA_ELF(k * const.hbar, hw_eV, PMMA_params, 'M') / k

        return get_PMMA_ELF(k * const.hbar, hw_eV, PMMA_params, 'M') / k

    km, kp = mcf.get_km_kp(E, hw)
    integral = integrate.quad(get_Y, km, kp)[0]

    return 1 / (np.pi * const.a0 * E_eV) * integral  # cm^-1 * eV^-1


# %% test PMMA DIIMFP_prec - OK
EE_D = [50, 100, 200, 300, 400, 500, 1000]
EE = np.linspace(1, 80, 100)

DIIMFP = np.zeros((len(EE_D), len(EE)))

for i, E_D in enumerate(EE_D):
    for j, E in enumerate(EE):
        # u.progress_bar(i, len(EE_D))
        DIIMFP[i, j] = get_PMMA_DIIMFP(E_D, E)

# %%
plt.figure(dpi=300)

for i in range(len(EE_D)):
    plt.plot(EE, DIIMFP[i, :] * 1e-8)

Dapor_DIIMFP = np.loadtxt('data/Dapor/Dapor_DIIMFP.txt')
plt.plot(Dapor_DIIMFP[:, 0], Dapor_DIIMFP[:, 1], '.')
plt.xlim(0, 100)
plt.ylim(0, 0.008)
plt.grid()
plt.show()

# %%
EE_50 = grid.EE[::50]

# DIIMFP_xchg = np.zeros((len(grid.EE), len(grid.EE)))
DIIMFP_50 = np.zeros((len(EE_50), len(grid.EE)))

progress_bar = tqdm(total=len(EE_50), position=0)

for i, E in enumerate(EE_50):
    for j, hw in enumerate(grid.EE):

        # DIIMFP_xchg[i, j] = get_PMMA_DIIMFP(E, hw, exchange=True)
        DIIMFP_50[i, j] = get_PMMA_DIIMFP(E, hw, exchange=False)

    progress_bar.update()



