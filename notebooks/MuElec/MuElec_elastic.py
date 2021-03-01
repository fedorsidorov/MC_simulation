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

plt.figure(dpi=300)
plt.loglog(EE_raw, sigma_raw * 1e-18 * const.n_Si, '.-')
plt.show()

# %%
sigma = np.zeros((len(grid.EE)))

inds = np.where(np.logical_and(grid.EE > EE_raw.min(), grid.EE < EE_raw.max()))[0]
sigma[inds] = util.log_log_interp(EE_raw, sigma_raw)(grid.EE[inds])

plt.figure(dpi=300)
plt.loglog(grid.EE, sigma * 1e-18 * const.n_Si, '.-')
plt.show()

# np.save('/Users/fedor/PycharmProjects/MC_simulation/Resources/MuElec/elastic_u.npy', sigma * 1e-18 * const.n_Si)

# %% sigmadiff
diff_arr = np.loadtxt('notebooks/MuElec/microelec/sigmadiff_elastic_e_Si.dat')
diff_arr = diff_arr[np.where(diff_arr[:, 0] <= 30e+3)]  # cut higher energies
EE_unique = np.unique(diff_arr[:, 0])

# %%
sigma_5eV_test = np.trapz(diff_arr[:181, 2] * 2 * np.pi * np.sin(np.deg2rad(diff_arr[:181, 1])), x=diff_arr[:181, 1])

plt.figure(dpi=300)
plt.plot(diff_arr[:181, 1], diff_arr[:181, 2], '.-')
plt.plot(diff_arr[181:362, 1], diff_arr[181:362, 2], '.-')
plt.show()

# %%
diff_sigma_pre = np.zeros((len(EE_unique), len(grid.THETA_deg)))

# plt.figure(dpi=300)

for i, E in enumerate(EE_unique):
    inds = np.where(diff_arr[:, 0] == E)[0]
    theta = diff_arr[inds, 1]
    now_diff_arr = diff_arr[inds, 2]

    # if 500 < E < 600:
    #     plt.plot(theta, now_diff_arr * np.sin(np.deg2rad(np.linspace(0, 180, 181))) )

    # inds = np.where(np.logical_and(grid.EE > hw.min(), grid.EE < hw.max()))[0]
    diff_sigma_pre[i, :] = util.lin_lin_interp(theta, now_diff_arr)(grid.THETA_deg)

# plt.show()

diff_sigma_pre[np.where(np.isnan(diff_sigma_pre))] = 0

# %%
diff_sigma = np.zeros((len(grid.EE), len(grid.THETA_deg)))

for j, _ in enumerate(grid.EE):
    inds = np.where(grid.EE > 10)[0]
    diff_sigma[inds, j] = \
        util.log_log_interp(EE_unique, diff_sigma_pre[:, j])(grid.EE[inds])

diff_sigma[np.where(np.isnan(diff_sigma))] = 0

# %%
plt.figure(dpi=300)

for i in range(270, 290):
    plt.plot(grid.THETA_deg, diff_sigma[i, :])

plt.show()

# %%
diff_sigma_cumulated = np.zeros(np.shape(diff_sigma))

for i, E in enumerate(grid.EE):

    now_diff_sigma = diff_sigma[i, :] * 2 * np.pi * np.sin(grid.THETA_rad)

    now_integral = np.trapz(diff_sigma[i, :], x=grid.THETA_rad)

    if now_integral == 0:
        continue

    now_cumulated_array = np.ones(len(grid.EE))

    for j in range(len(grid.THETA_rad)):
        now_cumulated_array[j] = np.trapz(diff_sigma[i, :j + 1], x=grid.THETA_rad[:j + 1]) / now_integral

    diff_sigma_cumulated[i, :] = now_cumulated_array

# %%
diff_sigma_LSP = np.load('Resources/ELSEPA/Si/Si_diff_cs_cumulated_muffin.npy')

plt.figure(dpi=300)
plt.plot(grid.THETA_deg, diff_sigma_cumulated[500, :])
plt.plot(grid.THETA_deg, diff_sigma_LSP[500, :])
plt.show()

# %%
np.save('Resources/MuElec/elastic_diff_sigma_cumulated.npy', diff_sigma_cumulated)
# np.save('Resources/MuElec/diff_sigma_6_norm.npy', diff_sigma_6_norm)

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
    sigma_test[i] = np.trapz(diff_sigma[i, :] * 2 * np.pi * np.sin(grid.THETA_rad), x=grid.THETA_rad)

# %%
plt.figure(dpi=300)
plt.loglog(grid.EE, sigma)
plt.loglog(grid.EE, sigma_test)
plt.show()


