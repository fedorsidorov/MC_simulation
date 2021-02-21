import numpy as np
import matplotlib.pyplot as plt
import importlib

import grid as grid
grid = importlib.reload(grid)

# %%
arr_old = np.load('notebooks/elastic/final_arrays/Si/Si_muffin_diff_cs_plane_norm_cumulated.npy')
arr_new = np.load('notebooks/elastic/final_arrays/cumulated_arrays/Si_diff_cs_cumulated_muffin.npy')

plt.figure(dpi=300)

for i in range(0, 1000, 50):
    plt.semilogx(grid.EE, arr_old[i, :])
    plt.semilogx(grid.EE, arr_new[i, :], '--')

plt.show()



