import importlib

import matplotlib.pyplot as plt
import numpy as np
import grid as grid
from functions import MC_functions as mcf

grid = importlib.reload(grid)
mcf = importlib.reload(mcf)

# %%
E1, _, _, E2, _, _, E3, _, _, E4, _, _, E5, _, _ =\
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/OLF_Si/fit_5osc/params_E_A_w_x5.npy')

EE_bind = [E1, E2, E3, E4, E5]

# for n_shell in [4]:
for n_shell in [1, 2, 3, 4, 5]:

    print(n_shell)

    DIIMFP = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/OLF_Si/DIIMFP_5osc/DIIMFP_' +
                     str(n_shell) + '.npy')

    DIIMFP_cumulated = np.zeros(np.shape(DIIMFP))

    for i, E in enumerate(grid.EE):

        if n_shell == 1:
            inds = np.where(
                np.logical_and(
                    grid.EE > 0,
                    grid.EE < (E + EE_bind[n_shell - 1]) / 2
                )
            )[0]

        else:
            inds = np.where(
                np.logical_and(
                    grid.EE > EE_bind[n_shell - 1],
                    grid.EE < (E + EE_bind[n_shell - 1]) / 2
                )
            )[0]

        now_integral = np.trapz(DIIMFP[i, inds], x=grid.EE[inds])

        if now_integral == 0:
            continue

        now_arr = np.zeros(len(grid.EE))

        for j in inds:
            now_arr[j] = np.trapz(DIIMFP[i, inds[0]:j + 1], x=grid.EE[inds[0]:j + 1]) / now_integral

        now_arr[inds[-1] + 1:] = 1

        DIIMFP_cumulated[i, :] = now_arr

    np.save('notebooks/OLF_Si/DIIMFP_5osc_cumulated/DIIMFP_' + str(n_shell - 1) + '_cumulated.npy', DIIMFP_cumulated)

# %%
plt.figure(dpi=300)
# plt.loglog(grid.EE, DIIMFP[100, :])
plt.semilogx(grid.EE, DIIMFP_cumulated[800, :])
plt.show()

# %%
arr = np.load('notebooks/OLF_Si/DIIMFP_5osc_cumulated/DIIMFP_4_cumulated.npy')




