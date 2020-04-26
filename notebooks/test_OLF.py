#%%
import numpy as np
import matplotlib.pyplot as plt
import grid as g

OLF_M = np.load('Resources/Mermin/PMMA_Mermin_OLF_k=5e-2.npy')
OLF_G = np.load('Resources/GOS/PMMA_GOS_OLF.npy')

plt.figure(dpi=300)
plt.loglog(g.EE, OLF_M, label='Mermin')
plt.loglog(g.EE, OLF_G, label='GOS')

DI = np.loadtxt('data/Dapor/Dapor_Im.txt')
plt.loglog(DI[:, 0], DI[:, 1], label='Dapor')

plt.xlabel('E, eV')
plt.ylabel('PMMA OLF')
plt.grid()
plt.legend()
plt.show()
