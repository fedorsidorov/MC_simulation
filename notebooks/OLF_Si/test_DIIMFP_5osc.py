import numpy as np
import matplotlib.pyplot as plt
import importlib
from scipy import integrate
from tqdm import tqdm

from functions import MC_functions as mcf
import constants as const
import grid

const = importlib.reload(const)
grid = importlib.reload(grid)
mcf = importlib.reload(mcf)

# %%
E1, _, _, E2, _, _, E3, _, _, E4, _, _, E5, _, _ =\
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/OLF_Si/fit_5osc/params_E_A_w_x5.npy')

EE_bind = [E1, E2, E3, E4, E5]


# %%
def get_u(n_shell):

    DIIMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/OLF_Si/DIIMFP_5osc/DIIMFP_' +
                     str(n_shell) + '.npy')

    u = np.zeros(len(grid.EE))

    for i, E in enumerate(grid.EE):

        if n_shell == 1:
            inds = np.where(
                np.logical_and(
                    # grid.EE > EE_bind[n_shell - 1],
                    # grid.EE > 2.5,
                    grid.EE > 0,
                    grid.EE < (E + EE_bind[n_shell - 1]) / 2
                )
            )

        else:
            inds = np.where(
                np.logical_and(
                    grid.EE > EE_bind[n_shell - 1],
                    # grid.EE > 2.5,
                    grid.EE < (E + EE_bind[n_shell - 1]) / 2
                )
            )

        u[i] = np.trapz(DIIMFP[i, inds], x=grid.EE[inds])

    return u


def get_u_prec(n_shell):

    DIIMFP_prec = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/OLF_Si/DIIMFP_5osc/DIIMFP_prec_' +
                     str(n_shell) + '.npy')

    u = np.zeros(len(grid.EE_prec))

    for i, E in enumerate(grid.EE_prec):

        if n_shell == 1:
            inds = np.where(
                np.logical_and(
                    # grid.EE_prec > EE_bind[n_shell - 1],
                    # grid.EE_prec > 2.5,
                    grid.EE_prec > 0,
                    grid.EE_prec < (E + EE_bind[n_shell - 1]) / 2
                )
            )

        else:
            inds = np.where(
                np.logical_and(
                    grid.EE_prec > EE_bind[n_shell - 1],
                    # grid.EE_prec > 2.5,
                    grid.EE_prec < (E + EE_bind[n_shell - 1]) / 2
                )
            )

        u[i] = np.trapz(DIIMFP_prec[i, inds], x=grid.EE_prec[inds])

    return u


# %%
paper_KLM = np.loadtxt('notebooks/OLF_Si/curves/Akkerman_KLM.txt')

u_1 = get_u(1)
u_2 = get_u(2)
u_3 = get_u(3)
u_4 = get_u(4)
u_5 = get_u(5)

u_K = u_5
u_L = u_3 + u_4
u_M = u_1 + u_2

plt.figure(dpi=300)
plt.loglog(paper_KLM[:, 0], paper_KLM[:, 1] * 1e-18 * const.n_Si, 'ro')

# plt.loglog(grid.EE, get_u(5))
# plt.loglog(grid.EE, get_u(3) + get_u(4))
# plt.loglog(grid.EE, get_u(1) + get_u(2))

plt.loglog(grid.EE_prec, get_u_prec(5))
plt.loglog(grid.EE_prec, get_u_prec(3) + get_u_prec(4))
plt.loglog(grid.EE_prec, get_u_prec(1) + get_u_prec(2))

plt.xlim(1e+1, 1e+4)
plt.ylim(1e+1, 1e+8)

plt.show()


