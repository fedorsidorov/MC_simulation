import importlib

import matplotlib.pyplot as plt

import numpy as np
from numpy import random

import grid as g
from SimClasses import utilities as u, constants as c, arrays as a

a = importlib.reload(a)
c = importlib.reload(c)
g = importlib.reload(g)
u = importlib.reload(u)

# %% test PMMA arrays
plt.figure(dpi=300)
plt.loglog(g.EE, a.PMMA_el_IMFP, label='elastic')
plt.loglog(g.EE, a.PMMA_val_IMFP, label='valence')
plt.loglog(g.EE, a.C_K_ee_IMFP, label='C 1s')
plt.loglog(g.EE, a.O_K_ee_IMFP, label='O 1s')
plt.loglog(g.EE, a.PMMA_ph_IMFP, label='phonon')
plt.loglog(g.EE, a.PMMA_pol_IMFP, label='polaron')
plt.loglog(g.EE, a.PMMA_val_IMFP + a.C_K_ee_IMFP + a.O_K_ee_IMFP + a.PMMA_ph_IMFP + a.PMMA_pol_IMFP,
           '--', label='total inelastic')
plt.title('PMMA IIMFP')
plt.xlabel('E, eV')
plt.ylabel('IIMFP, cm$^{-1}$')
plt.xlim(1, 1e+4)
plt.ylim(1e+2, 1e+9)
plt.legend()
plt.grid()
plt.show()

# %% test Si arrays
plt.figure(dpi=300)
plt.loglog(g.EE, a.Si_el_IMFP, label='elastic')

for n in range(6):
    plt.loglog(g.EE, a.Si_ee_IMFP_6[:, n] * c.n_Si * 1e-18, label='Si e-e ' + str(n))

plt.title('Si IIMFP')
plt.xlabel('E, eV')
plt.ylabel('IIMFP, cm$^{-1}$')
plt.xlim(1, 1e+4)
plt.ylim(1e+2, 1e+9)
plt.legend()
plt.grid()
plt.show()
