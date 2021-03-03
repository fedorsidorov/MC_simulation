import importlib

import matplotlib.pyplot as plt
import numpy as np
import grid as grid
from functions import MC_functions as mcf

grid = importlib.reload(grid)
mcf = importlib.reload(mcf)

# %%
cum_arr_raw = np.loadtxt('notebooks/MuElec/microelec/sigmadiff_cumulated_elastic_e_Si.dat')

cum_arr_raw = cum_arr_raw[np.where(cum_arr_raw[:, 0] <= 30e+3)]  # cut higher energies
EE_unique = np.unique(cum_arr_raw[:, 0])

cum_arr_pre = np.zeros((len(EE_unique), len(grid.THETA_deg)))

plt.figure(dpi=300)

for i, E in enumerate(EE_unique):
    inds = np.where(cum_arr_raw[:, 0] == E)[0]
    theta = cum_arr_raw[inds, 2]
    cum_prob = cum_arr_raw[inds, 1]

    plt.plot(theta, cum_prob)

    cum_arr_pre[i, :] = mcf.lin_lin_interp(theta, cum_prob)(grid.THETA_deg)

    plt.plot(grid.THETA_deg, cum_arr_pre[i, :], '--')

plt.grid()
plt.show()

# %%
cum_arr = np.zeros((len(grid.EE), len(grid.THETA_deg)))

for j, _ in enumerate(grid.THETA_deg):
    cum_arr[159:, j] = mcf.lin_lin_interp(EE_unique, cum_arr_pre[:, j])(grid.EE[159:])

cum_arr[:159, :] = cum_arr[159, :]

plt.figure(dpi=300)
plt.plot(grid.THETA_deg, cum_arr[0, :])
plt.show()

# %%
plt.figure(dpi=300)
plt.plot(grid.THETA_deg, cum_arr_pre[27, :])
plt.plot(grid.THETA_deg, cum_arr[659, :], '--')
plt.show()

# %%
np.save('notebooks/MuElec/MuElec_elastic_arrays/u_diff_cumulated.npy', cum_arr)



