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
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/fit_5osc/params_E_A_w_x5.npy')

EE_bind = [E1, E2, E3, E4, E5]


# %%
def get_u_prec(n_shell):

    DIIMFP_prec = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/u_diff/DIIMFP_prec_' +
                          str(n_shell) + '.npy')

    u = np.zeros(len(grid.EE_prec))

    for i, E in enumerate(grid.EE_prec):

        if n_shell == 1:
            inds = np.where(
                np.logical_and(
                    grid.EE_prec > 0,
                    grid.EE_prec < (E + EE_bind[n_shell - 1]) / 2
                )
            )

        else:
            inds = np.where(
                np.logical_and(
                    grid.EE_prec > EE_bind[n_shell - 1],
                    grid.EE_prec < (E + EE_bind[n_shell - 1]) / 2
                )
            )

        u[i] = np.trapz(DIIMFP_prec[i, inds], x=grid.EE_prec[inds])

    return u


# %%
paper_KLM_u = np.loadtxt('notebooks/Akkerman_Si_5osc/curves/Akkerman_u_KLM.txt')

plt.figure(dpi=300)
plt.loglog(paper_KLM_u[:, 0], paper_KLM_u[:, 1] * 1e-18 * const.n_Si, 'ro', label='paper K, L, M')

# plt.loglog(grid.EE, get_u(5))
# plt.loglog(grid.EE, get_u(3) + get_u(4))
# plt.loglog(grid.EE, get_u(1) + get_u(2))

plt.loglog(grid.EE_prec, get_u_prec(1) + get_u_prec(2), label='my M')
plt.loglog(grid.EE_prec, get_u_prec(3) + get_u_prec(4), label='my L')
plt.loglog(grid.EE_prec, get_u_prec(5), label='my K')

plt.loglog(grid.EE_prec, get_u_prec(1), '--', label='1', linewidth=1)
plt.loglog(grid.EE_prec, get_u_prec(2), '--', label='2', linewidth=1)
plt.loglog(grid.EE_prec, get_u_prec(3), '--', label='3', linewidth=1)
plt.loglog(grid.EE_prec, get_u_prec(4), '--', label='4', linewidth=1)
plt.loglog(grid.EE_prec, get_u_prec(5), '--', label='5', linewidth=1)

plt.xlabel('E, eV')
plt.ylabel(r'$\mu$, cm$^{-1}$')
plt.xlim(1e+1, 2e+4)
plt.ylim(1e+1, 1e+8)
plt.legend()
plt.grid()
plt.show()
# plt.savefig('u_K_L_M.jpg')

# %%
# u_0 = mcf.log_log_interp(grid.EE_prec, get_u_prec(1))(grid.EE)
# u_1 = mcf.log_log_interp(grid.EE_prec, get_u_prec(2))(grid.EE)
# u_2 = mcf.log_log_interp(grid.EE_prec, get_u_prec(3))(grid.EE)
# u_3 = mcf.log_log_interp(grid.EE_prec, get_u_prec(4))(grid.EE)
# u_4 = mcf.log_log_interp(grid.EE_prec, get_u_prec(5))(grid.EE)

u_0 = np.load('notebooks/Akkerman_Si_5osc/u/u_0.npy')
u_1 = np.load('notebooks/Akkerman_Si_5osc/u/u_1.npy')
u_2 = np.load('notebooks/Akkerman_Si_5osc/u/u_2.npy')
u_3 = np.load('notebooks/Akkerman_Si_5osc/u/u_3.npy')
u_4 = np.load('notebooks/Akkerman_Si_5osc/u/u_4.npy')

plt.figure(dpi=300)
plt.loglog(paper_KLM_u[:, 0], paper_KLM_u[:, 1] * 1e-18 * const.n_Si, 'ro')

plt.loglog(grid.EE, u_4)
plt.loglog(grid.EE, u_2 + u_3)
plt.loglog(grid.EE, u_0 + u_1)

plt.xlim(1e+1, 2e+4)
plt.ylim(1e+1, 1e+8)
plt.grid()
plt.show()

#%%
# np.save('notebooks/Akkerman_Si_5osc/u/u_0.npy', u_0)
# np.save('notebooks/Akkerman_Si_5osc/u/u_1.npy', u_1)
# np.save('notebooks/Akkerman_Si_5osc/u/u_2.npy', u_2)
# np.save('notebooks/Akkerman_Si_5osc/u/u_3.npy', u_3)
# np.save('notebooks/Akkerman_Si_5osc/u/u_4.npy', u_4)


# %%
def get_S_prec(n_shell):

    DIIMFP_prec = np.load(
        '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/u_diff/DIIMFP_prec_' +
        str(n_shell) + '.npy')

    S = np.zeros(len(grid.EE_prec))

    for i, E in enumerate(grid.EE_prec):

        inds = np.where(
            np.logical_and(
                grid.EE_prec > 0,
                grid.EE_prec < E / 2
            )
        )

        S[i] = np.trapz(DIIMFP_prec[i, inds] * grid.EE_prec[inds], x=grid.EE_prec[inds])

    return S


# %%
paper_KLM_S = np.loadtxt('notebooks/Akkerman_Si_5osc/curves/Akkerman_S_KLM.txt')

plt.figure(dpi=300)
plt.loglog(paper_KLM_S[:, 0], paper_KLM_S[:, 1] * 1e+7, 'ro', label='paper')

plt.loglog(grid.EE_prec, get_S_prec(1) + get_S_prec(2), label='my M')
plt.loglog(grid.EE_prec, get_S_prec(3) + get_S_prec(4), label='my L')
plt.loglog(grid.EE_prec, get_S_prec(5), label='my K')

plt.xlabel('E, eV')
plt.ylabel('dE/ds, eV/cm')
plt.xlim(1e+1, 2e+4)
plt.ylim(1e+5, 1e+9)
plt.legend()
plt.grid()
plt.show()
# plt.savefig('SP_K_L_M.jpg')
