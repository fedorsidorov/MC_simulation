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

# %%
plt.figure(dpi=300)
plt.loglog(OLF[:, 0], OLF[:, 1], '--', label='PMMA OLF')
plt.loglog(grid.EE, OLF_RH, '--', label='Ritsko Henke')
plt.loglog(grid.EE, OLF_RHD, '--', label='Ritsko Henke Dapor')

plt.legend()
# plt.ylim(1e-7, 1e+1)

plt.grid()
plt.show()

# %%
# plt.figure(dpi=300)
# plt.loglog(OLF_ext[:, 0], OLF_ext[:, 1], '.')
# plt.show()


# %%
def get_OLF(E):
    if E < min(grid.EE):
        return OLF_RH[0]
    return mcf.log_log_interp(grid.EE, OLF_RH)(E)


# OLF_EE = get_OLF(grid.EE)

# plt.figure(dpi=300)
# plt.loglog(grid.EE_prec, OLF_EE, '.')
# plt.show()


def get_S(x):
    return (1 - x) * np.log(4 / x) - 7/4 * x + x**(3/2) - 33/32 * x**2


def get_u_diff(E, hw):
    return const.m * const.e**2 / (2 * np.pi * const.hbar**2 * E) * get_OLF(hw) * get_S(hw / E)


def get_u(E):

    if E/2 < grid.EE[0]:
        return 0

    def get_Y(hw):
        return get_u_diff(E, hw)

    return integrate.quad(get_Y, grid.EE[0], E/2)[0]


def get_u_trapz(E):

    inds = np.where(grid.EE < E / 2)[0]

    if len(inds) == 0:
        return 0

    return np.trapz(get_u_diff(E, grid.EE[inds]), x=grid.EE[inds])


# %%
u_ee = np.zeros(len(grid.EE))

progress_bar = tqdm(total=len(grid.EE), position=0)

for i, e in enumerate(grid.EE):
    u_ee[i] = get_u(e)
    progress_bar.update()


# %%
u_ciappa = np.loadtxt('notebooks/OLF_PMMA/u_ciappa.txt')

plt.figure(dpi=300)

plt.loglog(grid.EE, 1 / u_ee * 1e+8)
plt.loglog(u_ciappa[:, 0], u_ciappa[:, 1])

# plt.ylim(1e+1, 1e+4)
plt.grid()

plt.show()






