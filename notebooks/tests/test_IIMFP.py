import importlib

import matplotlib.pyplot as plt

import numpy as np

import grid as grid
import constants as c
from _outdated import arrays as a
from functions import MC_functions as u

a = importlib.reload(a)
c = importlib.reload(c)
grid = importlib.reload(grid)
u = importlib.reload(u)

# %% test PMMA arrays_Si
plt.figure(dpi=300)
plt.loglog(grid.EE, a.PMMA_el_IMFP, label='elastic')
plt.loglog(grid.EE, a.PMMA_val_IMFP, label='valence')
plt.loglog(grid.EE, a.C_K_ee_IMFP, label='C 1s')
plt.loglog(grid.EE, a.O_K_ee_IMFP, label='O 1s')
plt.loglog(grid.EE, a.PMMA_ph_IMFP, label='phonon')
plt.loglog(grid.EE, a.PMMA_pol_IMFP, label='polaron')
plt.loglog(grid.EE, a.PMMA_val_IMFP + a.C_K_ee_IMFP + a.O_K_ee_IMFP + a.PMMA_ph_IMFP + a.PMMA_pol_IMFP,
           '--', label='total inelastic')
plt.title('PMMA IIMFP')
plt.xlabel('E, eV')
plt.ylabel('IIMFP, cm$^{-1}$')
plt.xlim(1, 1e+4)
plt.ylim(1e+2, 1e+9)
plt.legend()
plt.grid()
plt.show()

# %% test Si arrays_Si
plt.figure(dpi=300)
plt.loglog(grid.EE, a.Si_el_IMFP, label='elastic')

for n in range(6):
    plt.loglog(grid.EE, a.Si_ee_IMFP_6[:, n], label='Si e-e ' + str(n))

# Si_gryz_IIMFP_5 = np.load('notebooks/Gryzinski/Si_Gryzinski_IMFP_5.npy')
# for n in range(5):
#     plt.loglog(grid.EE, Si_gryz_IIMFP_5[:, n], '.', label='Si Gryzinski ' + str(n))

plt.loglog(grid.EE, np.sum(a.Si_ee_IMFP_6, axis=1), '--', label='total inelastic')

plt.title('Si IIMFP')
plt.xlabel('E, eV')
plt.ylabel('IIMFP, cm$^{-1}$')
plt.xlim(1, 1e+4)
plt.ylim(1e+2, 1e+9)
plt.legend()
plt.grid()
plt.show()
