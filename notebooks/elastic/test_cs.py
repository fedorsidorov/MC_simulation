import numpy as np
import matplotlib.pyplot as plt
import importlib

import grid
import constants as const
from functions import MC_functions as mcf

const = importlib.reload(const)
grid = importlib.reload(grid)
mcf = importlib.reload(mcf)


#%%
dapor_500eV = np.loadtxt('notebooks/elastic/curves/Dapor_Si_500eV.txt')
dapor_1keV = np.loadtxt('notebooks/elastic/curves/Dapor_Si_1keV.txt')
dapor_2keV = np.loadtxt('notebooks/elastic/curves/Dapor_Si_2keV.txt')

plt.figure(dpi=300)

plt.plot(dapor_500eV[:, 0], dapor_500eV[:, 1], '.', label='Dapor 500 eV')
plt.plot(dapor_1keV[:, 0], dapor_1keV[:, 1], '.', label='Dapor 1 keV')
plt.plot(dapor_2keV[:, 0], dapor_2keV[:, 1], '.', label='Dapor 2 keV')

EE = ['500 eV', '1 keV', '2 keV']

ind_500eV = 613
ind_1keV = 681
ind_2keV = 750

kind = 'muffin'

# diff_cs_pnc = np.load('notebooks/elastic/final_arrays/Si/Si_' + kind + '_diff_cs_plane_norm_cumulated.npy')
# diff_cs_pnc = np.load('data/ELSEPA/Si_muffin_diff_cs_plane_norm_cumulated.npy')

# diff_cs = np.load('Resources/ELSEPA/Si/Si_muffin_diff_cs_plane_norm.npy')
# diff_cs = np.load('Resources/ELSEPA/PMMA/MMA_muffin_diff_cs_plane_norm.npy')
# diff_cs_cumulated = np.zeros(np.shape(diff_cs))

diff_cs_cumulated = np.load('notebooks/elastic/final_arrays/Si/Si_muffin_diff_cs_plane_norm_cumulated.npy')

# for i in range(len(grid.EE)):
#     for j in range(len(grid.THETA_rad)):
#         diff_cs_cumulated[i, j] = np.trapz(diff_cs[i, :j + 1] * 2 * np.pi * np.sin(grid.THETA_rad[:j + 1]),
#                                            x=grid.THETA_rad[:j + 1])

# for i in range(len(diff_cs_pnc[:, 0])):
#     diff_cs_pnc[i, :] = mcf.get_cumulated_array(diff_cs[i, :], x=grid.THETA_rad)


for i, ind in enumerate([ind_500eV, ind_1keV, ind_2keV]):
    now_diff_cs_pnc = diff_cs_cumulated[ind, :]
    plt.plot(grid.THETA_deg, now_diff_cs_pnc, label='my ' + EE[i])


plt.xlabel('theta, degree')
plt.ylabel('probability')

plt.xlim(0, 180)
plt.ylim(0, 1)

plt.grid()
plt.legend()

plt.show()

# plt.savefig('Si_compare.pdf')

# %%
u_P = np.load('Resources/ELSEPA/PMMA/MMA_muffin_u.npy')
u_Si = np.load('Resources/ELSEPA/Si/Si_muffin_u.npy')

plt.figure(dpi=300)

plt.loglog(grid.EE, u_P, label='PMMA')
plt.loglog(grid.EE, u_Si, label='Si')

plt.grid()
plt.legend()

plt.show()

