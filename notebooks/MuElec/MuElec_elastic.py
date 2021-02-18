import numpy as np
import matplotlib.pyplot as plt
import importlib

import constants as const
import grid
from functions import MC_functions as util

const = importlib.reload(const)
grid = importlib.reload(grid)
util = importlib.reload(util)

# %% sigma
arr = np.loadtxt('notebooks/MuElec/microelec/sigma_elastic_e_Si.dat')
EE_raw = arr[:, 0]
sigma_raw = arr[:, 1]

# %%
sigma = np.zeros((len(grid.EE)))

inds = np.where(np.logical_and(grid.EE > EE_raw.min(), grid.EE < EE_raw.max()))[0]
sigma[inds] = util.log_log_interp(EE_raw, sigma_raw)(grid.EE[inds])

plt.figure(dpi=300)
plt.loglog(grid.EE, sigma * 1e-18 * const.n_Si)
plt.show()

# np.save('/Users/fedor/PycharmProjects/MC_simulation/Resources/MuElec/elastic_u.npy', sigma * 1e-18 * const.n_Si)

# %% sigmadiff
diff_arr = np.loadtxt('notebooks/MuElec/microelec/sigmadiff_elastic_e_Si.dat')
diff_arr = diff_arr[np.where(diff_arr[:, 0] <= 30e+3)]  # cut higher energies
EE_unique = np.unique(diff_arr[:, 0])

# %%
diff_sigma_pre = np.zeros((len(EE_unique), len(grid.THETA_deg)))

plt.figure(dpi=300)

for i, E in enumerate(EE_unique):
    inds = np.where(diff_arr[:, 0] == E)[0]
    theta = diff_arr[inds, 1]
    now_diff_arr = diff_arr[inds, 2]

    if 500 < E < 600:
        plt.plot( theta, now_diff_arr * np.sin(np.deg2rad(np.linspace(0, 180, 181))) )

    # inds = np.where(np.logical_and(grid.EE > hw.min(), grid.EE < hw.max()))[0]
    diff_sigma_pre[i, :] = util.lin_lin_interp(theta, now_diff_arr)(grid.THETA_deg)

plt.show()

diff_sigma_pre[np.where(np.isnan(diff_sigma_pre))] = 0

# %%
diff_sigma = np.zeros((len(grid.EE), len(grid.THETA_deg)))

for j, _ in enumerate(grid.EE):
    inds = np.where(grid.EE > 10)[0]
    diff_sigma[inds, j] = \
        util.log_log_interp(EE_unique, diff_sigma_pre[:, j])(grid.EE[inds])

diff_sigma[np.where(np.isnan(diff_sigma))] = 0

# %%
diff_sigma_norm = np.zeros(np.shape(diff_sigma))

for i in range(len(grid.EE)):
    if np.sum(diff_sigma[i, :]) != 0:
        diff_sigma_norm[i, :] = diff_sigma[i, :] * np.sin(grid.THETA_rad) /\
                                np.sum(diff_sigma[i, :] * np.sin(grid.THETA_rad))

# %%
# plt.figure(dpi=300)
# plt.plot(grid.THETA_deg, diff_sigma[500, :] / 1000)
# plt.plot(grid.THETA_deg, diff_sigma_norm[500, :])
# plt.show()

# %%
np.save('/Users/fedor/PycharmProjects/MC_simulation/Resources/MuElec/elastic_diff_sigma_sin_norm.npy', diff_sigma_norm)
# np.save('/Users/fedor/PycharmProjects/MC_simulation/Resources/MuElec/diff_sigma_6_norm.npy', diff_sigma_6_norm)

# %% test DIIMFP_prec
plt.imshow(diff_sigma)
plt.show()

# %%
# plt.figure(dpi=300)
# plt.plot(grid.THETA_deg, diff_sigma[500, :])
# plt.show()

# %%
sigma_test = np.zeros(np.shape(sigma))

for i, E in enumerate(grid.EE):
    # sigma_test[i] = np.trapz(diff_sigma[i, :] * 2 * np.pi, x=grid.THETA_deg)
    sigma_test[i] = np.trapz(diff_sigma[i, :] * 2 * np.pi * np.sin(grid.THETA_rad), x=grid.THETA_deg)

# %%
plt.figure(dpi=300)
plt.loglog(grid.EE, sigma)
plt.loglog(grid.EE, sigma_test)
plt.show()


