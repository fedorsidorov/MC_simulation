import numpy as np
import matplotlib.pyplot as plt
import importlib

import constants as const
import indexes
import grid
from functions import MC_functions as mcf

const = importlib.reload(const)
grid = importlib.reload(grid)
mcf = importlib.reload(mcf)
indexes = importlib.reload(indexes)

# %%
EE = grid.EE

diff_arr = np.loadtxt(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/MuElec/microelec/sigmadiff_cumulated_inelastic_e_Si.dat')

diff_arr = diff_arr[np.where(diff_arr[:, 0] <= 30e+3)]  # cut higher energies

diff_arr[np.where(diff_arr == 2)] = 1

EE_unique = np.unique(diff_arr[:, 0])

diff_sigma_pre = np.zeros((6, len(EE_unique), len(grid.EE)))

# %%
hw_list = []
diff_arr_list = []

for n in range(6):

    for i, E in enumerate(EE_unique):
        inds = np.where(diff_arr[:, 0] == E)[0]
        hw = diff_arr[inds, 1]
        now_diff_arr = diff_arr[inds, n + 2]

        hw_list.append(hw)
        diff_arr_list.append(now_diff_arr)

        now_arr = np.zeros(len(grid.EE))

        ind_beg = np.where(grid.EE > hw[0])[0][0]
        now_arr[:ind_beg] = 0

        if len(np.where(grid.EE > hw[-1])[0]) > 0:
            ind_end = np.where(grid.EE > hw[-1])[0][0]
            now_arr[ind_end:] = 1
            now_arr[ind_beg:ind_end] = mcf.log_lin_interp(hw, now_diff_arr)(grid.EE[ind_beg:ind_end])

        else:
            now_arr[ind_beg:] = mcf.log_lin_interp(hw, now_diff_arr)(grid.EE[ind_beg:])

        diff_sigma_pre[n, i, :] = now_arr


for n in range(6):
    diff_sigma_pre[n, 0, :] = diff_sigma_pre[n, 1, :]

# %%
diff_sigma_cumulated = np.zeros((6, len(grid.EE), len(grid.EE)))

for n in range(6):
    for j in range(len(grid.EE)):
        diff_sigma_cumulated[n, indexes.Si_E_cut_ind:, j] =\
            mcf.log_lin_interp(EE_unique, diff_sigma_pre[n, :, j])(grid.EE[indexes.Si_E_cut_ind:])

# %%
# inds_pre = list(range(3, 48, 7))
#
# inds = np.zeros(len(inds_pre), dtype=int)
#
# for i in range(len(inds)):
#     inds[i] = np.argmin(np.abs(grid.EE - EE_unique[inds_pre[i]]))
#
# print(EE_unique[inds_pre])
# print(grid.EE[inds])
#
# plt.figure(dpi=300)
#
# for i in range(len(inds)):
#     plt.semilogx(hw_list[inds_pre[i]], diff_arr_list[inds_pre[i]], 'o')
#     plt.semilogx(grid.EE, diff_sigma_cumulated[0, inds[i], :])
#
# plt.show()

# %% set extra elements to -2
for n in range(6):
    for i in range(len(grid.EE)):
        for j in range(len(grid.EE) - 1):

            if diff_sigma_cumulated[n, i, j + 1] == 0 and diff_sigma_cumulated[n, i, j] == 0:
                diff_sigma_cumulated[n, i, j] = 2

            if diff_sigma_cumulated[n, i, len(grid.EE) - j - 1] == 1 and\
                    diff_sigma_cumulated[n, i, len(grid.EE) - j - 1 - 1] == 1:
                diff_sigma_cumulated[n, i, len(grid.EE) - j - 1] = 2

# %%
np.save('notebooks/MuElec/MuElec_inelastic_arrays/sigma_diff_cumulated_6.npy', diff_sigma_cumulated)



