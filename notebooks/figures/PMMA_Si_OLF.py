import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import importlib

import grid

grid = importlib.reload(grid)

font = {'size': 14}
matplotlib.rc('font', **font)


# %%
EE_Si = np.load('notebooks/Akkerman_Si_5osc/OLF_Palik+Fano/EE_Palik+Fano.npy')
OLF_Si = np.load('notebooks/Akkerman_Si_5osc/OLF_Palik+Fano/OLF_Palik+Fano.npy')

EE_PMMA = grid.EE
OLF_PMMA = np.load('notebooks/Dapor_easiest/Ritsko_Henke_Im.npy')

plt.figure(dpi=300, figsize=[4, 3])

plt.loglog(EE_PMMA, OLF_PMMA, label=r'ПММА')
plt.loglog(EE_Si, OLF_Si, label=r'Si')

plt.legend(loc='lower left')
# plt.legend(loc='upper right')
plt.xlabel(r'$E$, эВ')
plt.ylabel(r'Im $\left [ \frac{-1}{\varepsilon (0, \omega)} \right ]$')

# plt.xlim(1e+1, 1e+4)
# plt.ylim(1e-9, 1e+2)

plt.xlim(1e+1, 1e+4)
# plt.ylim(1e-9, 1e+3)
plt.ylim(1e-11, 1e+1)

plt.grid()
plt.savefig('PMMA_Si_OLF_shrink.jpg', dpi=300, bbox_inches='tight')
plt.show()
