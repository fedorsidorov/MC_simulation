import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import grid
import constants as const
from tqdm import tqdm

grid = importlib.reload(grid)

# %%
H_cs = np.load('notebooks/elastic/final_arrays/_outdated/H/easy_diff_cs.npy')
C_cs = np.load('notebooks/elastic/final_arrays/_outdated/C/easy_diff_cs.npy')
O_cs = np.load('notebooks/elastic/final_arrays/_outdated/O/easy_diff_cs.npy')
Si_cs = np.load('notebooks/elastic/final_arrays/_outdated/simple_Si_MC/muffin_diff_cs.npy')

Pei_cs = np.loadtxt('notebooks/elastic/curves/Pei/Si_0.1keV.txt')

ind_50 = 386
ind_100 = 454
ind_500 = 613
ind_1k = 681
ind_2k = 749

# ind = ind_50

# my_PMMA_diff_cs = H_cs*8 + C_cs*5 + O_cs*2
# now_PMMA_diff_cs = my_PMMA_diff_cs[ind]
# final_DCS = now_PMMA_diff_cs * 2 * np.pi * np.sin(grid.THETA_rad)

plt.figure(dpi=300)
plt.semilogy(grid.THETA_deg, Si_cs[ind_100])
plt.semilogy(Pei_cs[:, 0], Pei_cs[:, 1])

plt.show()

ans = grid.EE


# %% total CS
total_cs = np.load('notebooks/elastic/final_arrays/_outdated/H/atomic_cs.npy')

# 7.3211e-17





