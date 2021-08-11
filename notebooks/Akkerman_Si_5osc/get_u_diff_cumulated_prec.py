import importlib

import matplotlib.pyplot as plt
import numpy as np
import grid as grid
from functions import MC_functions as mcf

grid = importlib.reload(grid)
mcf = importlib.reload(mcf)

# %%
Eb0, _, _, Eb1, _, _, Eb2, _, _, Eb3, _, _, Eb4, _, _ =\
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/fit_5osc/params_E_A_w_x5.npy')

EE_bind = [Eb0, Eb1, Eb2, Eb3, Eb4]

# for n_shell in [0, 1, 2, 3, 4]:
for n_osc in [0]:

    print(n_osc)

    u_diff = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/u_diff/u_diff_' +
                     str(n_osc) + '_prec.npy')

    u_diff_cumulated_prec = np.zeros(np.shape(u_diff))

    for i, E in enumerate(grid.EE_prec):

        if n_osc == 0:
            inds = np.where(
                np.logical_and(
                    grid.EE_prec > 0,
                    grid.EE_prec < (E + EE_bind[n_osc]) / 2
                )
            )[0]

        else:
            inds = np.where(
                np.logical_and(
                    grid.EE_prec > EE_bind[n_osc],
                    grid.EE_prec < (E + EE_bind[n_osc]) / 2
                )
            )[0]

        now_integral = np.trapz(u_diff[i, inds], x=grid.EE_prec[inds])

        if now_integral == 0:
            u_diff_cumulated_prec[i, :] = 1
            continue

        now_arr = np.zeros(len(grid.EE_prec))

        for j in inds:
            now_arr[j] = np.trapz(u_diff[i, inds[0]:j + 1], x=grid.EE_prec[inds[0]:j + 1]) / now_integral

        now_arr[inds[-1] + 1:] = 1
        u_diff_cumulated_prec[i, :] = now_arr

    u_diff_cumulated_pre = \
        mcf.log_lin_interp(grid.EE_prec, u_diff_cumulated_prec, axis=1)(grid.EE)

    u_diff_cumulated = \
        mcf.log_lin_interp(grid.EE_prec, u_diff_cumulated_pre, axis=0)(grid.EE)

    np.save('notebooks/Akkerman_Si_5osc/u_diff_cumulated_precised/u_diff_' + str(n_osc) + '_cumulated_prec.npy',
            u_diff_cumulated)

# %%
print(np.where(np.isnan(u_diff_cumulated)))

# %%
osc = 4
inds = [100, 200, 300, 400, 500, 600, 700, 800, 900]
# inds = [800]

arr = np.load('notebooks/Akkerman_Si_5osc/u_diff_cumulated/u_diff_' + str(osc) + '_cumulated.npy')
arr_prec = np.load('notebooks/Akkerman_Si_5osc/u_diff_cumulated_precised/u_diff_' + str(osc) + '_cumulated_prec.npy')

plt.figure(dpi=300)

for i in inds:
    plt.semilogx(grid.EE, arr[i, :])
    plt.semilogx(grid.EE, arr_prec[i, :], '--')

# plt.xlim(0, 400)
plt.grid()
plt.show()







