import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import constants as const
import grid as grid
from tqdm import tqdm

const = importlib.reload(const)
grid = importlib.reload(grid)

# %%
P_100 = np.loadtxt('notebooks/PMMA_2015/curves/Dapor_P_100eV.txt')
P_1000 = np.loadtxt('notebooks/PMMA_2015/curves/Dapor_P_1keV.txt')

PP = np.load('notebooks/simple_PMMA_MC/PMMA/arrays_PMMA/PMMA_val_DIMFP_cumulated_corr.npy')

E_ind_100 = 454
E_ind_1000 = 681

plt.figure(dpi=300)
plt.semilogx(P_100[:, 0], P_100[:, 1])
plt.semilogx(grid.EE, PP[E_ind_100, :])

plt.semilogx(P_1000[:, 0], P_1000[:, 1])
plt.semilogx(grid.EE, PP[E_ind_1000, :])

plt.show()