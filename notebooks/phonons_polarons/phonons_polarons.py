import numpy as np
import matplotlib.pyplot as plt
import importlib

import grid

grid = importlib.reload(grid)


#%% PMMA phonons and polarons
def get_PMMA_U_phonon(E, T=300):
    a0 = 5.292e-11 * 1e+2  # cm
    kT = 8.617e-5 * T  # eV
    # kT = mc.k_B * T
    hw = 0.1
    e_0 = 3.9
    e_inf = 2.2
    
    nT = (np.exp(hw/kT) - 1)**(-1)
    KT = 1 / a0 * ((nT + 1)/2) * ((e_0 - e_inf) / (e_0 * e_inf))
    
    U_PH = KT * hw/E * np.log((1 + np.sqrt(1 - hw/E)) / (1 - np.sqrt(1 - hw/E)))  # cm^-1
    
    return U_PH


def get_PMMA_U_polaron(E, C_inv_nm, gamma):
    
    C = C_inv_nm * 1e+7  # nm^-1 -> cm^-1
    U_POL = C * np.exp(-gamma * E)  # cm^-1
    
    return U_POL


#%%
u_ph = get_PMMA_U_phonon(grid.EE)
# u_pol = get_PMMA_U_polaron(grid.EE, 1.5, 0.14)
u_pol = get_PMMA_U_polaron(grid.EE, 0.07, 0.1)

# np.save('Resources/PhPol/PMMA_polaron_IMFP_0p07_0p1.npy', u_pol)
np.save('Resources/PhPol/PMMA_polaron_IMFP_0p07_0p1.npy', u_pol)

# %%
# u_ph_0 = np.load('Resources/PhPol/PMMA_phonon_IMFP.npy')
# u_pol_0 = np.load('Resources/PhPol/PMMA_polaron_IMFP_0p1_0p15.npy')

plt.figure(dpi=300)

# plt.loglog(grid.EE, u_ph)
# plt.loglog(grid.EE, u_pol)

plt.semilogx(grid.EE, u_ph)
plt.semilogx(grid.EE, u_pol)

# plt.loglog(grid.EE, u_ph_0, '--')
# plt.loglog(grid.EE, u_pol_0, '--')

# plt.xlim(1, 1e+4)
# plt.ylim(1e+1, 1e+9)

plt.show()

#%%
D_ph = np.loadtxt('notebooks/phonons_polarons/Dapor_thesis_l_ph.txt')
D_pol = np.loadtxt('notebooks/phonons_polarons/Dapor_thesis_l_pol.txt')

plt.plot(dpi=300)

plt.semilogy(D_ph[:, 0], D_ph[:, 1], 'o', label='Dapor phonon')
plt.semilogy(D_pol[:, 0], D_pol[:, 1], 'o', label='Dapor polaron')

ind = 523
EE = grid.EE[:ind]

u_ph = get_PMMA_U_phonon(EE)
u_pol = get_PMMA_U_polaron(EE, 1.5, 0.14)

plt.semilogy(EE, 1/u_ph * 1e+8, label='my phonon')
plt.semilogy(EE, 1/u_pol * 1e+8, label='my_polaron')

plt.xlim(0, 200)
plt.ylim(1e+0, 1e+3)

plt.legend()
plt.grid()

plt.show()
