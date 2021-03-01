import numpy as np
import matplotlib.pyplot as plt
import importlib

import grid

grid = importlib.reload(grid)

# %%
# arr = np.loadtxt('notebooks/OLF_Si/akkerman_Si_shells.txt')
# arr = np.loadtxt('notebooks/OLF_Si/IM.txt')
# arr = np.loadtxt('notebooks/OLF_Si/OLF_Akkerman_fit.txt')
# arr = np.load('notebooks/OLF_Si/OLF_total.npy')

EE = np.load('notebooks/OLF_Si/Palik/E_Palik.npy')
Im = np.load('notebooks/OLF_Si/Palik/Im_Palik.npy')

plt.figure(dpi=300)
# plt.loglog(arr[:, 0], arr[:, 1], 'o')
plt.loglog(EE, Im, 'o')
plt.show()








