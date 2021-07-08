import numpy as np
import matplotlib.pyplot as plt
import importlib

import grid as grid
grid = importlib.reload(grid)

# %%
arr_old = np.load('notebooks/elastic/final_arrays/simple_Si_MC/Si_muffin_diff_cs_plane_norm_cumulated.npy')
arr_new = np.load('notebooks/elastic/final_arrays/cumulated_arrays/Si_el_DIMFP_cumulated.npy')

plt.figure(dpi=300)

for i in range(0, 1000, 50):
    plt.semilogx(grid.EE, arr_old[i, :])
    plt.semilogx(grid.EE, arr_new[i, :], '--')

plt.show()



