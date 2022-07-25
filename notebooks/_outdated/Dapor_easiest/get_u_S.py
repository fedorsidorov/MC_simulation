import numpy as np
import matplotlib.pyplot as plt
import importlib
from scipy import integrate
from functions import MC_functions as mcf
import constants as const
import grid
from tqdm import tqdm

grid = importlib.reload(grid)
const = importlib.reload(const)
mcf = importlib.reload(mcf)

OLF = np.loadtxt('notebooks/_outdated/OLF_PMMA/PMMA_OLF.txt')
OLF_RH = np.load('notebooks/_outdated/OLF_PMMA/Ritsko_Henke_Im.npy')
OLF_RHD = np.load('notebooks/_outdated/OLF_PMMA/Ritsko_Henke_Dapor_Im.npy')
# OLF_C = np.loadtxt('notebooks/_outdated/Dapor_easiest/curves/OLF_Ciappa.txt')

# %%
plt.figure(dpi=300)
plt.loglog(OLF[:, 0], OLF[:, 1], '--', label='PMMA OLF')
plt.loglog(grid.EE, OLF_RH, '--', label='Ritsko Henke')
plt.loglog(grid.EE, OLF_RHD, '--', label='Ritsko Henke Dapor')
# plt.loglog(OLF_C[:, 0], OLF_C[:, 1], '--', label='PMMA Ciappa')

plt.legend()
plt.ylim(1e-7, 1e+1)

plt.grid()
plt.show()

# %%
# plt.figure(dpi=300)
# plt.loglog(OLF_ext[:, 0], OLF_ext[:, 1], '.')
# plt.show()


# %%
def get_OLF(E):
    if E < min(grid.EE):
        # return OLF_RH[0]
        return 0
    return mcf.log_log_interp(grid.EE, OLF_RH)(E)


def get_S(x):
    return (1 - x) * np.log(4 / x) - 7/4 * x + x**(3/2) - 33/32 * x**2


def get_G(x):
    return np.log(1.166/x) - 3/4*x - x/4*np.log(4/x) + 1/2*x**(3/2) - x**2/16*np.log(4/x) - 31/48*x**2


def get_u_diff(E, hw):  # WO factor 2 !!!
    return const.m * const.e**2 / (np.pi * const.hbar**2 * E) * get_OLF(hw)


def get_u(E):

    if E/2 < grid.EE[0]:
        return 0

    def get_Y(hw):
        return get_u_diff(E, hw) * get_S(hw / E)

    return integrate.quad(get_Y, 0, E/2, limit=1000)[0]


def get_SP(E):

    if E/2 < grid.EE[0]:
        return 0

    def get_Y(hw):
        return get_u_diff(E, hw) * get_G(hw / E) * hw

    return integrate.quad(get_Y, 0, E/2)[0]


# %%
# u = np.zeros(len(grid.EE))
S = np.zeros(len(grid.EE))

progress_bar = tqdm(total=len(grid.EE), position=0)

for i, e in enumerate(grid.EE):
    # u[i] = get_u(e)
    S[i] = get_SP(e)
    progress_bar.update()


# %%
u_ciappa = np.loadtxt('notebooks/_outdated/Dapor_easiest/curves/u_ciappa.txt')

plt.figure(dpi=300)

# plt.loglog(grid.EE, 1 / u * 1e+8)
plt.loglog(u_ciappa[:, 0], u_ciappa[:, 1])

# plt.ylim(1e+1, 1e+4)
plt.grid()

plt.show()

# %%
S_ciappa = np.loadtxt('notebooks/_outdated/Dapor_easiest/curves/S_ciappa.txt')

plt.figure(dpi=300)

plt.semilogx(grid.EE, S / 1e+8)
plt.semilogx(S_ciappa[:, 0], S_ciappa[:, 1])

# plt.ylim(1e+1, 1e+4)
plt.grid()

plt.show()

# %%
E_ind = 454
now_E = grid.EE[E_ind]

now_u, err = get_u(now_E)[:2]

