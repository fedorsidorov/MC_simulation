import importlib
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
from functions import MC_functions as mcf
import grid

grid = importlib.reload(grid)
mcf = importlib.reload(mcf)


# %% OK
el_u_old = np.load('notebooks/Si_old/arrays_Si/Si_el_IMFP_nm.npy')

elastic_model = 'muffin'  # 'easy', 'atomic', 'muffin'
elastic_extrap = ''  # '', 'extrap_'

el_u_now = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/Si/Si_'
    + elastic_model + '_u_' + elastic_extrap + 'nm.npy'
)

plt.figure(dpi=300)
plt.loglog(grid.EE, el_u_old)
plt.loglog(grid.EE, el_u_now)
plt.show()

# %% OK
el_u_diff_old = np.load('notebooks/Si_old/arrays_Si/Si_el_DIMFP_cumulated.npy')

el_u_diff_now = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/Si/Si_diff_cs_cumulated_'
    + elastic_model + '_' + elastic_extrap + '+1.npy'
)

ind = 999

plt.figure(dpi=300)
plt.semilogy(grid.THETA_deg, el_u_diff_old[ind, :])
plt.semilogy(grid.THETA_deg, el_u_diff_now[ind, :], '--')
plt.show()


# %% OK
ee_u_old = np.load('notebooks/Si_old/arrays_Si/5_osc/Si_ee_IMFP_5osc_nm.npy')

ee_u_now = np.zeros((1000, 5))

for j in range(5):
    ee_u_now[:, j] = np.load(
        '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/u/u_' + str(j) + '_nm_precised.npy'
    )

ans = ee_u_now - ee_u_old

# %%
ee_u_diff_old = np.load('notebooks/Si_old/arrays_Si/5_osc/Si_ee_DIMFP_cumulated_5osc.npy')

Si_ee_E_bind = [0, 20.1, 102, 151.1, 1828.9]

ee_u_diff_now = np.zeros((5, 1000, 1000))

for n in range(5):
    ee_u_diff_now[n, :, :] = np.load(
        '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/u_diff_cumulated/u_diff_'
        + str(n) + '_cumulated_precised.npy'
    )

    ee_u_diff_now[np.where(np.abs(ee_u_diff_now - 1) < 1e-10)] = 1

    for i in range(1000):
        for j in range(1000 - 1):

            if ee_u_diff_now[n, i, j] == 0 and ee_u_diff_now[n, i, j + 1] == 0:
                ee_u_diff_now[n, i, j] = -2

            if ee_u_diff_now[n, i, 1000 - j - 1] == 1 and \
                    ee_u_diff_now[n, i, 1000 - j - 2] == 1:
                ee_u_diff_now[n, i, 1000 - j - 1] = -2

    zero_inds = np.where(ee_u_diff_now[n, -1, :] == 0)[0]

    if len(zero_inds) > 0:

        zero_ind = zero_inds[0]

        if grid.EE[zero_ind] < Si_ee_E_bind[n]:
            ee_u_diff_now[n, :, zero_ind] = -2

ee_u_diff_now[0, :4, 5] = -2

ee_u_diff_now[1, :301, :297] = -2
ee_u_diff_now[1, 300, 296] = 0

ee_u_diff_now[2, :461, :457] = -2
ee_u_diff_now[2, 460, 456] = 0

ee_u_diff_now[3, :500, :496] = -2
ee_u_diff_now[3, 499, 495] = 0

ee_u_diff_now[4, :745, :742] = -2
ee_u_diff_now[4, 744, 741] = 0

# %% OK
osc_ind = 4

inds = [300, 400, 500, 600, 700, 800, 900]

plt.figure(dpi=300)

for ind in inds:
    plt.semilogx(grid.EE, ee_u_diff_old[osc_ind, ind, :])
    plt.semilogx(grid.EE, ee_u_diff_now[osc_ind, ind, :], '--')

plt.show()






