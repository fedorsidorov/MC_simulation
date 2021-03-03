import importlib

import matplotlib.pyplot as plt
import numpy as np
import grid as grid
import constants as const
from functions import MC_functions as mcf

grid = importlib.reload(grid)
const = importlib.reload(const)
mcf = importlib.reload(mcf)

# %% sigma
arr = np.loadtxt('data/MuElec/microelec/sigma_inelastic_e_Si.dat')
EE_raw = arr[:, 0]
sigma_raw = arr[:, 1:]

plt.figure(dpi=300)
plt.loglog(EE_raw, sigma_raw, 'o')
plt.show()

# %%
sigma = np.zeros((len(grid.EE), np.shape(sigma_raw)[1]))

for n in range(6):
    inds = np.where(np.logical_and(grid.EE > EE_raw.min(), grid.EE < EE_raw.max()))[0]
    sigma[inds, n] = mcf.log_log_interp(EE_raw, sigma_raw[:, n])(grid.EE[inds])

# np.save('notebooks/MuElec/MuElec_inelastic_arrays/u_ee_6.npy', sigma * 1e-18 * const.n_Si)

# %%
paper = np.loadtxt('notebooks/OLF_Si/curves/Akkerman_u_KLM.txt')

plt.figure(dpi=300)

plt.loglog(grid.EE, sigma[:, 0] + sigma[:, 1] + sigma[:, 2])
plt.loglog(grid.EE, sigma[:, 3] + sigma[:, 4])
plt.loglog(grid.EE, sigma[:, 5])
plt.loglog(paper[:, 0], paper[:, 1], 'o')
plt.show()

# %% sigmadiff
diff_arr = np.loadtxt('data/MuElec/microelec/sigmadiff_inelastic_e_Si.dat')
diff_arr = diff_arr[np.where(diff_arr[:, 0] <= 30e+3)]  # cut higher energies
EE_unique = np.unique(diff_arr[:, 0])

diff_sigma_6_pre = np.zeros((6, len(EE_unique), len(grid.EE)))

# plt.figure(dpi=300)

for i, E in enumerate(EE_unique):
    inds = np.where(diff_arr[:, 0] == E)[0]
    hw = diff_arr[inds, 1]
    now_diff_arr = diff_arr[inds, 2:]

    # plt.loglog(hw, now_diff_arr, 'o')

    for n in range(6):
        inds = np.where(np.logical_and(grid.EE > hw.min(), grid.EE < hw.max()))[0]
        diff_sigma_6_pre[n, i, inds] = mcf.log_log_interp(hw, now_diff_arr[:, n])(grid.EE[inds])

# plt.ylim(1e-9, 1e+3)
# plt.show()

# %%
plt.figure(dpi=300)

for i in range(0, 49, 5):
    plt.loglog(grid.EE, diff_sigma_6_pre[0, i, :])

plt.ylim(1e-5, 1e+3)
plt.show()

# %%
diff_sigma_6 = np.zeros((6, len(grid.EE), len(grid.EE)))

for n in range(6):
    for j, _ in enumerate(grid.EE):
        inds = np.where(grid.EE > 10)[0]
        diff_sigma_6[n, inds, j] = \
            mcf.log_log_interp(EE_unique, diff_sigma_6_pre[n, :, j])(grid.EE[inds])

# %%
diff_sigma_6_cumulated = np.zeros(np.shape(diff_sigma_6))

for n in range(6):

    DIIMFP = diff_sigma_6[n, :, :]
    DIIMFP_cumulated = np.zeros(np.shape(DIIMFP))

    for i, E in enumerate(grid.EE):

        # inds = np.where(grid.EE < E / 2)[0]
        inds = np.where(grid.EE < E)[0]
        now_integral = np.trapz(DIIMFP[i, inds], x=grid.EE[inds])

        if now_integral == 0:
            continue

        now_cumulated_array = np.ones(len(grid.EE))

        for j in inds:
            now_cumulated_array[j] = np.trapz(DIIMFP[i, :j + 1], x=grid.EE[:j + 1]) / now_integral

        DIIMFP_cumulated[i, :] = now_cumulated_array

    diff_sigma_6_cumulated[n] = DIIMFP_cumulated

# %%
np.save('Resources/MuElec/diff_sigma_6_E.npy', diff_sigma_6)
np.save('Resources/MuElec/diff_sigma_6_E_cumulated.npy', diff_sigma_6_cumulated)

# %% test DIIMFP_prec
plt.imshow(diff_sigma_6_cumulated[5])
plt.show()

# %%
sigma_test = np.zeros(np.shape(sigma))


for n in range(6):
    for i, E in enumerate(grid.EE):
        # inds = np.where(grid.EE < E)
        inds = np.where(grid.EE < E/2)
        sigma_test[i, n] = np.trapz(diff_sigma_6[n, i, inds], x=grid.EE[inds])

# %% test IMFP
plt.figure(dpi=300)

for n in range(6):
    plt.loglog(grid.EE, sigma[:, n], 'o')
    plt.loglog(grid.EE, sigma_test[:, n])

plt.ylim(1e-3, 1e+3)

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
# np.save('Resources/MuElec/Si_MuElec_IIMFP.npy', sigma_corr)
# np.save('Resources/MuElec/Si_MuElec_DIIMFP_norm.npy', diff_sigma_6_norm)
