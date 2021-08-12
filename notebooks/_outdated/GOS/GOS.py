import importlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from tqdm.auto import trange
import grid
import constants as const
from functions import MC_functions as mcf
from mpl_toolkits.mplot3d import Axes3D

mcf = importlib.reload(mcf)
const = importlib.reload(const)
grid = importlib.reload(grid)


# %% GOS for individual atoms
def get_df_dW_K(k, hw_eV, el):  # BOOK
    if el == 'C':
        Zs = const.Zs_C
        edge = const.K_Ebind_C
    elif el == 'O':
        Zs = const.Zs_O
        edge = const.K_Ebind_O
    else:
        print('Specify atom type - C or O')
        return -1

    if hw_eV < edge:
        return 0

    E = hw_eV * const.eV
    Qp = (k * const.a0 / Zs) ** 2
    kH2 = hw_eV * const.eV / (Zs ** 2 * const.Ry) - 1

    if kH2 > 0:
        kH = np.sqrt(kH2)
        tan_beta_p = 2 * kH / (Qp - kH2 + 1)
        beta_p = np.arctan(tan_beta_p)

        if beta_p < 0:  # seems to be important
            beta_p += np.pi

        num = 256 * E * (Qp + kH2 ** 2 / 3 + 1 / 3) * np.exp(-2 * beta_p / kH)
        den = Zs ** 4 * const.Ry ** 2 * ((Qp - kH2 + 1) ** 2 + 4 * kH2) ** 3 * (1 - np.exp(-2 * np.pi / kH))

    else:
        y = -(-kH2) ** (-1 / 2) * np.log(
            (Qp + 1 - kH2 + 2 * (-kH2) ** (1 / 2)) / (Qp + 1 - kH2 - 2 * (-kH2) ** (1 / 2)))
        num = 256 * E * (Qp + kH2 / 3 + 1 / 3) * np.exp(y)
        den = Zs ** 4 * const.Ry ** 2 * ((Qp - kH2 + 1) ** 2 + 4 * kH2) ** 3

    return num / den


# %% test GOS for C K-shell
EE = np.linspace(300, 1200, 50)  # eV
kk = np.linspace(0.01, 100, 500)  # inv A
FF = np.zeros((len(EE), len(kk)))

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

for i, Ei in enumerate(EE):
    for j, ki in enumerate(kk):
        FF[i, j] = get_df_dW_K(ki * 1e+8, Ei, 'C') * const.eV * 1e+3

    ax.plot(np.ones(len(kk)) * EE[i], np.log((kk * 1e+8 * const.a0) ** 2), FF[i, :])

ax.view_init(30, 30)
plt.grid()
ax.set_xlabel('E, eV')
ax.set_ylabel('ln(ka$_0$)')
ax.set_zlabel('df/dW * 10$^3$, eV$^{-1}$')
plt.show()
# plt.savefig('GOS.png')


# %% PMMA ELF
def get_GOS_ELF(k, hw_eV, el):
    factor = 2 * np.pi ** 2 * const.e ** 2 * const.hbar ** 2 * const.n_MMA / (const.m * hw_eV * const.eV)
    C_GOS = factor * get_df_dW_K(k, hw_eV, 'C') * const.N_C_MMA
    O_GOS = factor * get_df_dW_K(k, hw_eV, 'O') * const.N_O_MMA

    if el == 'C':
        return C_GOS
    if el == 'O':
        return O_GOS
    if el == 'PMMA':
        return C_GOS + O_GOS
    else:
        print('Specify atom type - C, O, PMMA')
        return -1


# %% test PMMA OLF
EE = grid.EE
OLF_PMMA = np.zeros(len(EE))

for i, Ei in enumerate(EE):
    OLF_PMMA[i] = get_GOS_ELF(1e-100, Ei, 'PMMA')

plt.figure(dpi=300)
plt.loglog(EE, OLF_PMMA)
plt.show()


# %%
def get_PMMA_DIIMFP_GOS(T_eV, hw_eV, el='PMMA'):
    if hw_eV > T_eV:
        return 0

    T = T_eV * const.eV
    hw = hw_eV * const.eV

    def get_Y(k):
        return get_GOS_ELF(k, hw_eV, el) / k

    km, kp = mcf.get_km_kp(T, hw)
    integral = integrate.quad(get_Y, km, kp)[0]

    return 1 / (np.pi * const.a0 * T_eV) * integral  # cm^-1 * eV^-1


def get_GOS_IIMFP(T_eV, el):

    E_bind = 0

    if el == 'C':
        E_bind = const.K_Ebind_C
    elif el == 'O':
        E_bind = const.K_Ebind_O

    def get_Y(hw_eV):
        return get_PMMA_DIIMFP_GOS(T_eV, hw_eV, el)
    return integrate.quad(get_Y, E_bind / 2, (E_bind + T_eV) / 2)[0]


# %%
EE = grid.EE

DIIMFP_C = np.zeros((len(EE), len(EE)))
DIIMFP_O = np.zeros((len(EE), len(EE)))
IIMFP_C = np.zeros(len(EE))
IIMFP_O = np.zeros(len(EE))

for i in trange(len(EE), position=0):
    E = EE[i]
    IIMFP_C[i] = get_GOS_IIMFP(E, 'C')
    IIMFP_O[i] = get_GOS_IIMFP(E, 'O')

    for j, hw in enumerate(EE):
        DIIMFP_C[i, j] = get_PMMA_DIIMFP_GOS(E, hw, 'C')
        DIIMFP_O[i, j] = get_PMMA_DIIMFP_GOS(E, hw, 'O')

# %%
np.save('Resources/GOS/C_DIIMFP_E_bind.npy', DIIMFP_C)
np.save('Resources/GOS/O_DIIMFP_E_bind.npy', DIIMFP_O)

np.save('Resources/GOS/C_IIMFP_E_bind.npy', IIMFP_C)
np.save('Resources/GOS/O_IIMFP_E_bind.npy', IIMFP_O)


# %%
def get_cumulated_DIIMFP(DIIMFP, el):

    E_bind = 0

    if el == 'C':
        E_bind = const.K_Ebind_C
    elif el == 'O':
        E_bind = const.K_Ebind_O

    DIIMFP_cumulated = np.ones(np.shape(DIIMFP)) * 2

    for i, E in enumerate(grid.EE):

        inds = np.where(np.logical_and(
            grid.EE > E_bind,
            grid.EE < (E_bind + E) / 2)
        )[0]

        now_integral = np.trapz(DIIMFP[i, inds], x=grid.EE[inds])

        if now_integral == 0:
            continue

        now_cumulated_array = np.ones(len(grid.EE)) * 2

        for j in inds:
            now_cumulated_array[j] = np.trapz(DIIMFP[i, inds[0]:j + 1], x=grid.EE[inds[0]:j + 1]) / now_integral

        DIIMFP_cumulated[i, :] = now_cumulated_array

    return DIIMFP_cumulated


# %%
DIIMFP_C_cumulated = get_cumulated_DIIMFP(DIIMFP_C, 'C')
DIIMFP_O_cumulated = get_cumulated_DIIMFP(DIIMFP_O, 'O')

# %%
np.save('Resources/GOS/PMMA_C_DIIMFP_cumulated_E_bind.npy', DIIMFP_C_cumulated)
np.save('Resources/GOS/PMMA_O_DIIMFP_cumulated_E_bind.npy', DIIMFP_O_cumulated)

# %%
arr_C = np.load('Resources/GOS/C_DIIMFP_cumulated.npy')
arr_O = np.load('Resources/GOS/O_DIIMFP_cumulated.npy')

plt.figure(dpi=300)

for i in range(0, 1000, 50):
    plt.semilogx(grid.EE, arr_O[i, :])

plt.show()
