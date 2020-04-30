import importlib

import matplotlib.pyplot as plt
import numpy as np
import grid as grid
import utilities as util

grid = importlib.reload(grid)
util = importlib.reload(util)

# %% sigma
arr = np.loadtxt('data/MuElec/microelec/sigma_inelastic_e_Si.dat')
EE_raw = arr[:, 0]
sigma_raw = arr[:, 1:]

sigma = np.zeros((len(grid.EE), np.shape(sigma_raw)[1]))

for n in range(6):
    inds = np.where(np.logical_and(grid.EE > EE_raw.min(), grid.EE < EE_raw.max()))[0]
    sigma[inds, n] = util.log_log_interp(EE_raw, sigma_raw[:, n])(grid.EE[inds])

# %% sigmadiff
diff_arr = np.loadtxt('data/MuElec/microelec/sigmadiff_inelastic_e_Si.dat')
diff_arr = diff_arr[np.where(diff_arr[:, 0] <= 30e+3)]  # cut higher energies
EE_unique = np.unique(diff_arr[:, 0])

diff_sigma_6_pre = np.zeros((6, len(EE_unique), len(grid.EE)))

for i, E in enumerate(EE_unique):
    inds = np.where(diff_arr[:, 0] == E)[0]
    hw = diff_arr[inds, 1]
    now_diff_arr = diff_arr[inds, 2:]

    for n in range(6):
        inds = np.where(np.logical_and(grid.EE > hw.min(), grid.EE < hw.max()))[0]
        diff_sigma_6_pre[n, i, inds] = util.log_log_interp(hw, now_diff_arr[:, n])(grid.EE[inds])

diff_sigma_6 = np.zeros((6, len(grid.EE), len(grid.EE)))

for n in range(6):
    for j, _ in enumerate(grid.EE):
        inds = np.where(grid.EE > 10)[0]
        diff_sigma_6[n, inds, j] = \
            util.log_log_interp(EE_unique, diff_sigma_6_pre[n, :, j])(grid.EE[inds])

diff_sigma_6_norm = np.zeros(np.shape(diff_sigma_6))

for n in range(6):
    for i in range(len(grid.EE)):
        if np.sum(diff_sigma_6[n, i, :]) != 0:
            diff_sigma_6_norm[n, i, :] = diff_sigma_6[n, i, :] / np.sum(diff_sigma_6[n, i, :])

# %% test DIIMFP
plt.imshow(diff_sigma_6[5])
plt.show()

# %%
sigma_test = np.zeros(np.shape(sigma))

for n in range(6):
    for i, E in enumerate(grid.EE):
        inds = np.where(grid.EE < E)
        sigma_test[i, n] = np.trapz(diff_sigma_6[n, i, inds], x=grid.EE[inds])

# %% test IMFP
plt.figure(dpi=300)

for n in range(6):
    plt.loglog(grid.EE, sigma[:, n], 'o')
    plt.loglog(grid.EE, sigma_test[:, n])

plt.show()

#%% test diff_cs sums
sigma_corr = np.zeros(np.shape(sigma))
inds = []

for n in range(6):
    for i in range(len(grid.EE)):
        if np.sum(diff_sigma_6_norm[n, i, :]) == 0 and sigma[i, n] > 0:
            sigma_corr[i, n] = 0
            inds.append([n, i])
        else:
            sigma_corr[i, n] = sigma[i, n]

#%%
np.save('Resources/MuElec/Si_MuElec_IIMFP.npy', sigma_corr)
np.save('Resources/MuElec/Si_MuElec_DIIMFP_norm.npy', diff_sigma_6_norm)
