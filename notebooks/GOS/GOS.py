import importlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from tqdm.auto import trange
import grid as g
import constants_physics as c
from functions import MC_functions as u

u = importlib.reload(u)
c = importlib.reload(c)
g = importlib.reload(g)


# %% GOS for individual atoms
def get_df_dW_K(k, hw_eV, el):  # BOOK
    if el == 'C':
        Zs = c.Zs_C
        edge = c.K_Ebind_C
    elif el == 'O':
        Zs = c.Zs_O
        edge = c.K_Ebind_O
    else:
        print('Specify atom type - C or O')
        return -1

    if hw_eV < edge:
        return 0

    E = hw_eV * c.eV
    Qp = (k * c.a0 / Zs) ** 2
    kH2 = hw_eV * c.eV / (Zs ** 2 * c.Ry) - 1

    if kH2 > 0:
        kH = np.sqrt(kH2)
        tan_beta_p = 2 * kH / (Qp - kH2 + 1)
        beta_p = np.arctan(tan_beta_p)

        if beta_p < 0:  # seems to be important
            beta_p += np.pi

        num = 256 * E * (Qp + kH2 ** 2 / 3 + 1 / 3) * np.exp(-2 * beta_p / kH)
        den = Zs ** 4 * c.Ry ** 2 * ((Qp - kH2 + 1) ** 2 + 4 * kH2) ** 3 * (1 - np.exp(-2 * np.pi / kH))

    else:
        y = -(-kH2) ** (-1 / 2) * np.log(
            (Qp + 1 - kH2 + 2 * (-kH2) ** (1 / 2)) / (Qp + 1 - kH2 - 2 * (-kH2) ** (1 / 2)))
        num = 256 * E * (Qp + kH2 / 3 + 1 / 3) * np.exp(y)
        den = Zs ** 4 * c.Ry ** 2 * ((Qp - kH2 + 1) ** 2 + 4 * kH2) ** 3

    return num / den


# %% test GOS for C K-shell
EE = np.linspace(300, 1200, 50)  # eV
kk = np.linspace(0.01, 100, 500)  # inv A
FF = np.zeros((len(EE), len(kk)))

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

for i, Ei in enumerate(EE):
    for j, ki in enumerate(kk):
        FF[i, j] = get_df_dW_K(ki * 1e+8, Ei, 'C') * c.eV * 1e+3

    ax.plot(np.ones(len(kk)) * EE[i], np.log((kk * 1e+8 * c.a0) ** 2), FF[i, :])

ax.view_init(30, 30)
plt.grid()
ax.set_xlabel('E, eV')
ax.set_ylabel('ln(ka$_0$)')
ax.set_zlabel('df/dW * 10$^3$, eV$^{-1}$')
plt.show()


# plt.savefig('GOS.png', dpi=300)


# %% PMMA ELF
def get_GOS_ELF(k, hw_eV, el):
    factor = 2 * np.pi ** 2 * c.e ** 2 * c.hbar ** 2 * c.n_MMA / (c.m * hw_eV * c.eV)
    C_GOS = factor * get_df_dW_K(k, hw_eV, 'C') * c.N_C_MMA
    O_GOS = factor * get_df_dW_K(k, hw_eV, 'O') * c.N_O_MMA

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
EE = g.EE
OLF_PMMA = np.zeros(len(EE))

for i, Ei in enumerate(EE):
    OLF_PMMA[i] = get_GOS_ELF(1e-100, Ei, 'PMMA')

plt.figure(dpi=300)
plt.loglog(EE, OLF_PMMA)
plt.show()

# np.save('Resources/GOS/PMMA_GOS_OLF_k=1e-100.npy', OLF)


# %%
def get_PMMA_DIIMFP_GOS(T_eV, hw_eV, el='PMMA'):
    if hw_eV > T_eV:
        return 0

    T = T_eV * c.eV
    hw = hw_eV * c.eV

    def get_Y(k):
        return get_GOS_ELF(k, hw_eV, el) / k

    km, kp = u.get_km_kp(T, hw)
    integral = integrate.quad(get_Y, km, kp)[0]

    return 1 / (np.pi * c.a0 * T_eV) * integral  # cm^-1 * eV^-1


def get_GOS_IIMFP(T_eV, el):
    def get_Y(hw_eV):
        return get_PMMA_DIIMFP_GOS(T_eV, hw_eV, el)
    return integrate.quad(get_Y, 0, T_eV / 2)[0]


# %%
EE = g.EE

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

IIMFP_PMMA = IIMFP_C + IIMFP_O
DIIMFP_PMMA = DIIMFP_C + DIIMFP_O

# %%
np.save('Resources/GOS/DIIMFP_GOS_C.npy', DIIMFP_C)
np.save('Resources/GOS/DIIMFP_GOS_O.npy', DIIMFP_O)
np.save('Resources/GOS/DIIMFP_GOS_PMMA.npy', DIIMFP_PMMA)

np.save('Resources/GOS/C_GOS_IIMFP.npy', IIMFP_C)
np.save('Resources/GOS/O_GOS_IIMFP.npy', IIMFP_O)
np.save('Resources/GOS/IIMFP_GOS_PMMA.npy', IIMFP_PMMA)


# %%
DIIMFP_C_norm = np.zeros((len(EE), len(EE)))
DIIMFP_O_norm = np.zeros((len(EE), len(EE)))
DIIMFP_PMMA_norm = np.zeros((len(EE), len(EE)))

for i in range(len(EE)):
    if not np.sum(DIIMFP_C[i, :]) == 0:
        DIIMFP_C_norm[i, :] = DIIMFP_C[i, :] / np.sum(DIIMFP_C[i, :])

    if not np.sum(DIIMFP_O[i, :]) == 0:
        DIIMFP_O_norm[i, :] = DIIMFP_O[i, :] / np.sum(DIIMFP_O[i, :])

    if not np.sum(DIIMFP_C[i, :]) == 0:
        DIIMFP_PMMA_norm[i, :] = DIIMFP_PMMA[i, :] / np.sum(DIIMFP_PMMA[i, :])

# %%
np.save('Resources/GOS/C_GOS_DIIMFP_norm.npy', DIIMFP_C_norm)
np.save('Resources/GOS/O_GOS_DIIMFP_norm.npy', DIIMFP_O_norm)
np.save('Resources/GOS/DIIMFP_GOS_PMMA_norm.npy', DIIMFP_PMMA_norm)
