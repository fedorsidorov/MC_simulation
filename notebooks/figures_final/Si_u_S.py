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
E0, _, _, E1, _, _, E2, _, _, E3, _, _, E4, _, _ =\
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/fit_5osc/params_E_A_w_x5.npy')

EE_bind = [E0, E1, E2, E3, E4]


# %%
def get_u_prec(n_osc):

    DIIMFP_prec = np.load(
        'notebooks/Akkerman_Si_5osc/u_diff/u_diff_' + str(n_osc) + '_prec.npy')

    u = np.zeros(len(grid.EE_prec))

    for i, E in enumerate(grid.EE_prec):

        if n_osc == 0:
            inds = np.where(
                np.logical_and(
                    grid.EE_prec > 0,
                    grid.EE_prec < (E + EE_bind[n_osc]) / 2
                )
            )

        else:
            inds = np.where(
                np.logical_and(
                    grid.EE_prec > EE_bind[n_osc],
                    grid.EE_prec < (E + EE_bind[n_osc]) / 2
                )
            )

        u[i] = np.trapz(DIIMFP_prec[i, inds], x=grid.EE_prec[inds])

    return u


# %%
paper_KLM_u = np.loadtxt('notebooks/Akkerman_Si_5osc/curves/Akkerman_u_KLM.txt')

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.loglog(paper_KLM_u[:, 0], paper_KLM_u[:, 1] * 1e-18 * const.n_Si, 'r.', label='paper K, L, M')

    ax.loglog(grid.EE_prec, get_u_prec(0) + get_u_prec(1), label='my M')
    ax.loglog(grid.EE_prec, get_u_prec(2) + get_u_prec(3), label='my L')
    ax.loglog(grid.EE_prec, get_u_prec(4), label='my K')

    # plt.loglog(grid.EE_prec, get_u_prec(0), '--', label='1', linewidth=1)
    # plt.loglog(grid.EE_prec, get_u_prec(1), '--', label='2', linewidth=1)
    # plt.loglog(grid.EE_prec, get_u_prec(2), '--', label='3', linewidth=1)
    # plt.loglog(grid.EE_prec, get_u_prec(3), '--', label='4', linewidth=1)
    # plt.loglog(grid.EE_prec, get_u_prec(4), '--', label='5', linewidth=1)

    ax.legend(fontsize=7)
    ax.set(xlabel=r'энергия электрона, эВ')
    ax.set(ylabel=r'$\mu$, cm$^{-1}$')
    ax.autoscale(tight=True)
    plt.xlim(1e+1, 2e+4)
    plt.ylim(1e+1, 1e+8)

    # plt.show()
    fig.savefig('figures_final/Si_KLM_u.jpg', dpi=600)


# %%
u_0 = mcf.log_log_interp(grid.EE_prec, get_u_prec(0))(grid.EE)
u_1 = mcf.log_log_interp(grid.EE_prec, get_u_prec(1))(grid.EE)
u_2 = mcf.log_log_interp(grid.EE_prec, get_u_prec(2))(grid.EE)
u_3 = mcf.log_log_interp(grid.EE_prec, get_u_prec(3))(grid.EE)
u_4 = mcf.log_log_interp(grid.EE_prec, get_u_prec(4))(grid.EE)


plt.loglog(grid.EE, u_4)
plt.loglog(grid.EE, u_2 + u_3)
plt.loglog(grid.EE, u_0 + u_1)

plt.xlim(1e+1, 2e+4)
plt.ylim(1e+1, 1e+8)
plt.grid()
plt.show()

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    plt.loglog(paper_KLM_u[:, 0], paper_KLM_u[:, 1] * 1e-18 * const.n_Si, 'ro')

    plt.loglog(grid.EE_prec, get_u_prec(0) + get_u_prec(1), label='my M')
    plt.loglog(grid.EE_prec, get_u_prec(2) + get_u_prec(3), label='my L')
    plt.loglog(grid.EE_prec, get_u_prec(4), label='my K')

    # plt.loglog(grid.EE_prec, get_u_prec(0), '--', label='1', linewidth=1)
    # plt.loglog(grid.EE_prec, get_u_prec(1), '--', label='2', linewidth=1)
    # plt.loglog(grid.EE_prec, get_u_prec(2), '--', label='3', linewidth=1)
    # plt.loglog(grid.EE_prec, get_u_prec(3), '--', label='4', linewidth=1)
    # plt.loglog(grid.EE_prec, get_u_prec(4), '--', label='5', linewidth=1)

    ax.legend(fontsize=7)
    ax.set(xlabel=r'энергия электрона, эВ')
    ax.set(ylabel=r'$\mu$, cm$^{-1}$')
    ax.autoscale(tight=True)
    plt.xlim(1e+1, 2e+4)
    plt.ylim(1e+1, 1e+8)

    plt.show()
    # fig.savefig('figures_final/Si_KLM_u.jpg', dpi=600)


# %%
def get_S_prec(n_osc):

    u_diff_prec = np.load(
        'notebooks/Akkerman_Si_5osc/u_diff/u_diff_' + str(n_osc) + '_prec.npy')

    S = np.zeros(len(grid.EE_prec))

    for i, E in enumerate(grid.EE_prec):

        inds = np.where(
            np.logical_and(
                grid.EE_prec > 0,
                # grid.EE_prec > EE_bind[n_osc],
                grid.EE_prec < E / 2
            )
        )

        S[i] = np.trapz(u_diff_prec[i, inds] * grid.EE_prec[inds], x=grid.EE_prec[inds])

    return S


# %%
paper_KLM_S = np.loadtxt('notebooks/Akkerman_Si_5osc/curves/Akkerman_S_KLM.txt')

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    plt.loglog(paper_KLM_S[:, 0], paper_KLM_S[:, 1] * 1e+7, 'r.', label='paper')

    plt.loglog(grid.EE_prec, get_S_prec(0) + get_S_prec(1), label='my M')
    plt.loglog(grid.EE_prec, get_S_prec(2) + get_S_prec(3), label='my L')
    plt.loglog(grid.EE_prec, get_S_prec(4), label='my K')

    ax.legend(fontsize=7)
    ax.set(xlabel=r'энергия электрона, эВ')
    ax.set(ylabel=r'dE/ds, эВ/см')
    ax.autoscale(tight=True)
    plt.xlim(1e+1, 2e+4)
    plt.ylim(1e+5, 1e+10)

    # plt.show()
    fig.savefig('figures_final/Si_KLM_S.jpg', dpi=600)