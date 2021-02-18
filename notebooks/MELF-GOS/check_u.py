import importlib
import numpy as np

from functions import MC_functions as mcf
import indexes as inds
import grid
import arrays_nm as arr

import matplotlib.pyplot as plt

mcf = importlib.reload(mcf)
arr = importlib.reload(arr)
inds = importlib.reload(inds)
grid = importlib.reload(grid)


# %%
PMMA_ee_u = arr.PMMA_val_IMFP + arr.O_K_ee_IMFP + arr.C_K_ee_IMFP

paper = np.loadtxt('notebooks/MELF-GOS/MELF-GOS.txt')

plt.figure(dpi=300)

plt.semilogx(grid.EE, 1 / PMMA_ee_u)
plt.semilogx(paper[:, 0], paper[:, 1], '.')

plt.xlim(10, 5000)
plt.ylim(0, 10)

plt.grid()

plt.show()








