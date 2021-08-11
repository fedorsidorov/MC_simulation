import numpy as np
import matplotlib.pyplot as plt
import importlib

import grid

grid = importlib.reload(grid)

# %%
# arr = np.loadtxt('notebooks/Akkerman_Si_5osc/akkerman_Si_shells.txt')
# arr = np.loadtxt('notebooks/Akkerman_Si_5osc/IM.txt')
# arr = np.loadtxt('notebooks/Akkerman_Si_5osc/OLF_Akkerman_fit.txt')
# arr = np.load('notebooks/Akkerman_Si_5osc/OLF_total.npy')

EE = np.load('notebooks/Akkerman_Si_5osc/Palik/E_Palik.npy')
Im = np.load('notebooks/Akkerman_Si_5osc/Palik/Im_Palik.npy')

plt.figure(dpi=300)
# plt.loglog(arr[:, 0], arr[:, 1], 'o')
plt.loglog(EE, Im, 'o')
plt.show()








