import numpy as np
import matplotlib.pyplot as plt
import importlib

import grid as grid
from _outdated import arrays_nm as arr
import constants as const

arr = importlib.reload(arr)
grid = importlib.reload(grid)
const = importlib.reload(const)

# %%
EE = grid.EE

K_cs = np.loadtxt('notebooks/MuElec/MuElec_K_cs.txt')
L_cs = np.loadtxt('notebooks/MuElec/MuElec_L_cs.txt')
M_cs = np.loadtxt('notebooks/MuElec/MuElec_M_cs.txt')

u = np.load('Resources/MuElec/Si_MuElec_IIMFP.npy')
u_diff = np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/MuElec/diff_sigma_6.npy')\
         * 1e-18 * const.n_Si

# %%
K_cs_my = u[:, 5] / const.n_Si
L_cs_my = (u[:, 3] + u[:, 4]) / const.n_Si
M_cs_my = (u[:, 0] + u[:, 1] + u[:, 2]) / const.n_Si

plt.figure(dpi=300)

plt.loglog(EE, K_cs_my * 1e+18)
plt.loglog(K_cs[:, 0], K_cs[:, 1])

plt.loglog(EE, L_cs_my * 1e+18)
plt.loglog(L_cs[:, 0], L_cs[:, 1])

plt.loglog(EE, M_cs_my * 1e+18)
plt.loglog(M_cs[:, 0], M_cs[:, 1])

plt.show()

# %% test IMFP
u_test = np.zeros((1000, 6))

for n in range(6):
    for i, E in enumerate(grid.EE):
        inds = np.where(grid.EE < E)
        # inds = np.where(grid.EE < E / 2)
        u_test[i, n] = np.trapz(u_diff[n, i, inds], x=grid.EE[inds])

# %%
plt.figure(dpi=300)

for n in range(6):
    # plt.loglog(grid.EE, sigma[:, n], 'o')
    plt.loglog(grid.EE, u_test[:, n] / const.n_Si * 1e+18)

# plt.ylim(1e-3, 1e+3)

plt.show()








