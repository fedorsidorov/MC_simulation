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


# %%
arr_111 = np.load('notebooks/elastic/final_arrays/root_Hg/root_111_diff_cs.npy')
arr_211 = np.load('notebooks/elastic/final_arrays/root_Hg/root_211_diff_cs.npy')
arr_311 = np.load('notebooks/elastic/final_arrays/root_Hg/root_311_diff_cs.npy')
arr_411 = np.load('notebooks/elastic/final_arrays/root_Hg/root_411_diff_cs.npy')

plt.figure(dpi=300)

# for i in range(0, 1000, 50):
i = 500

plt.semilogy(grid.THETA_deg, arr_111[i, :])
plt.semilogy(grid.THETA_deg, arr_211[i, :])
plt.semilogy(grid.THETA_deg, arr_311[i, :])
plt.semilogy(grid.THETA_deg, arr_411[i, :])

plt.show()

# %%
arr_111 = np.load('notebooks/elastic/final_arrays/root_Si/root_111_diff_cs.npy')
arr_211 = np.load('notebooks/elastic/final_arrays/root_Si/root_211_diff_cs.npy')
arr_311 = np.load('notebooks/elastic/final_arrays/root_Si/root_311_diff_cs.npy')
arr_411 = np.load('notebooks/elastic/final_arrays/root_Si/root_411_diff_cs.npy')

plt.figure(dpi=300)

# for i in range(0, 1000, 50):
i = 500

plt.semilogy(grid.THETA_deg, arr_111[i, :])
plt.semilogy(grid.THETA_deg, arr_211[i, :])
plt.semilogy(grid.THETA_deg, arr_311[i, :])
plt.semilogy(grid.THETA_deg, arr_411[i, :])

plt.show()







