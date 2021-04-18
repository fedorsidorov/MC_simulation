import numpy as np
import matplotlib.pyplot as plt
import importlib

import grid
import constants as const

grid = importlib.reload(grid)
const = importlib.reload(const)

# %%
arr = np.loadtxt('notebooks/OLF_Si/Fano/Fano.txt')

plt.figure(dpi=300)
plt.loglog(arr[:, 0], arr[:, 1])
plt.grid()
plt.show()

# %%
ph_OLF = const.n_Si * const.c * arr[:, 1] * const.M_Si_atom / (arr[:, 0] * const.eV * 1000 / const.hbar)

EE = np.load('notebooks/OLF_Si/Palik/E_Palik.npy')
Im = np.load('notebooks/OLF_Si/Palik/Im_Palik.npy')

# plt.figure(dpi=300)
# plt.loglog(EE[:4208], Im[:4208], 'o')
# plt.loglog(arr[16:, 0] * 1000, ph_OLF[16:], 'o')

# plt.xlim(1e+2, 1e+3)
# plt.ylim(1e-6, 1e+1)
# plt.grid()

# plt.show()

# %%
EE_total = np.concatenate((EE[:4208], arr[16:, 0] * 1000))
OLF_total = np.concatenate((Im[:4208], ph_OLF[16:]))

plt.figure(dpi=300)
plt.loglog(EE_total, OLF_total, 'o')
plt.show()

# %%
np.save('notebooks/OLF_Si/OLF_Palik+Fano/EE_Palik+Fano.npt', EE_total)
np.save('notebooks/OLF_Si/OLF_Palik+Fano/OLF_Palik+Fano.npt', OLF_total)


