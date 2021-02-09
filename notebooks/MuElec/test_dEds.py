import numpy as np
import matplotlib.pyplot as plt
import importlib

import arrays_nm as arr
import grid
import constants as const

arr = importlib.reload(arr)
grid = importlib.reload(grid)
const = importlib.reload(const)

# %%
K_S = np.loadtxt('notebooks/MuElec/MuElec_K_S.txt')
L_S = np.loadtxt('notebooks/MuElec/MuElec_L_S.txt')
M_S = np.loadtxt('notebooks/MuElec/MuElec_M_S.txt')

u_diff = np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/MuElec/diff_sigma_6.npy')\
         * 1e-18 * const.n_Si

# %% test S
S_test = np.zeros((1000, 6))

for n in range(6):
    for i, E in enumerate(grid.EE):
        inds = np.where(grid.EE < E)
        # inds = np.where(grid.EE < E / 2)
        S_test[i, n] = np.trapz(u_diff[n, i, inds] * EE[inds], x=grid.EE[inds])

# %%
plt.figure(dpi=300)

for n in range(6):
    # plt.loglog(grid.EE, sigma[:, n], 'o')
    plt.loglog(grid.EE, S_test[:, n] * 1e-7)

# plt.ylim(1e-3, 1e+3)

plt.show()

# %%
plt.figure(dpi=300)

plt.loglog(grid.EE, (S_test[:, 0] + S_test[:, 1] + S_test[:, 2]) * 1e-7)
plt.loglog(grid.EE, (S_test[:, 3] + S_test[:, 4]) * 1e-7)
plt.loglog(grid.EE, S_test[:, 5] * 1e-7)

plt.loglog(K_S[:, 0], K_S[:, 1])
plt.loglog(L_S[:, 0], L_S[:, 1])
plt.loglog(M_S[:, 0], M_S[:, 1])

plt.ylim(1e-3, 1e+2)

plt.show()


