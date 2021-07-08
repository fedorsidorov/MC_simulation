import importlib
import matplotlib.pyplot as plt
import numpy as np

import grid
import constants as const
from functions import MC_functions as mcf

mcf = importlib.reload(mcf)
const = importlib.reload(const)
grid = importlib.reload(grid)

DIIMFP = np.load('notebooks/Mermin/DIIMFP_prec/DIIMFP_from_parts.npy')

#%% test IMFP - OK
IIMFP = np.zeros(len(grid.EE_prec))

for i, E in enumerate(grid.EE_prec):
    inds = np.where(grid.EE_prec <= E/2)
    IIMFP[i] = np.trapz(DIIMFP[i, inds], grid.EE_prec[inds])


IIMFP_int = mcf.log_log_interp(grid.EE_prec, IIMFP)(grid.EE)
DB = np.loadtxt('notebooks/Mermin/curves/Dapor_BOOK_grey.txt')
paper = np.loadtxt('notebooks/MELF-GOS/MELF-GOS.txt')

plt.figure(dpi=300)

plt.semilogx(grid.EE, 1/IIMFP_int * 1e+8, label='my IMFP')
plt.semilogx(DB[:, 0], DB[:, 1], '.', label='Dapor IMFP')
plt.semilogx(paper[:, 0], paper[:, 1] * 10, '.', label='Dapor IMFP 2015')

plt.xlim(1e+1, 1e+3)
plt.ylim(0, 40)

plt.grid()
plt.legend()
plt.show()

#%% interpolate it all
DIIMFP[np.where(DIIMFP == 0)] = 1e-100

DIIMFP_int = mcf.log_log_interp_2d(grid.EE_prec, grid.EE_prec, DIIMFP)(grid.EE, grid.EE)

DIIMFP[np.where(DIIMFP < 1)] = 0
DIIMFP_int[np.where(DIIMFP_int < 1)] = 0

plt.figure(dpi=300)

# plt.loglog(grid.EE_prec, DIIMFP_prec[555, :])
# plt.loglog(grid.EE, DIIMFP_int[455, :], '--')

# plt.loglog(grid.EE_prec, DIIMFP_prec[740, :])
# plt.loglog(grid.EE, DIIMFP_int[681, :], '--')

plt.semilogx(grid.EE_prec, DIIMFP[241, :])
plt.semilogx(grid.EE, DIIMFP_int[68, :], 'o')

plt.grid()
plt.show()

# %%
DIIMFP_50 = np.load('notebooks/Mermin/DIIMFP_50.npy')

plt.figure(dpi=300)

for i in range(20):
    plt.semilogx(grid.EE, DIIMFP_50[i, :])

for i in range(0, 1000, 50):
    plt.semilogx(grid.EE, DIIMFP_int[i, :], '.')

plt.show()

#%%
IIMFP_test = np.zeros(len(grid.EE))

for i, E in enumerate(grid.EE):
    inds = np.where(grid.EE <= E/2)
    IIMFP_test[i] = np.trapz(DIIMFP_int[i, inds], grid.EE[inds])

plt.figure(dpi=300)
plt.loglog(grid.EE, 1/IIMFP_test * 1e+8, label='my IMFP')
plt.loglog(grid.EE, 1/IIMFP_int * 1e+8, '--', label='my IMFP int')
plt.loglog(DB[:, 0], DB[:, 1], '.', label='Dapor IMFP')
plt.legend()
plt.grid()
plt.show()

#%%
DIIMFP_cumulated = np.zeros((len(grid.EE), len(grid.EE)))

for i, E in enumerate(grid.EE):

    inds = np.where(grid.EE < E/2)[0]
    now_integral = np.trapz(DIIMFP_int[i, inds], x=grid.EE[inds])

    if now_integral == 0:
        continue

    now_cumulated_array = np.ones(len(grid.EE))

    for j in inds:
        now_cumulated_array[j] = np.trapz(DIIMFP_int[i, :j+1], x=grid.EE[:j+1]) / now_integral

    DIIMFP_cumulated[i, :] = now_cumulated_array

# %%
before = np.load('Resources/Mermin/DIIMFP_Mermin_PMMA_norm.npy')

before_cumulated = np.zeros(np.shape(before))

for i in range(len(grid.EE)):
    before_cumulated[i, :] = mcf.get_cumulated_array(before[i, :])

# %%
plt.figure(dpi=300)
plt.semilogx(grid.EE, DIIMFP_cumulated[200, :], label='now')
plt.semilogx(grid.EE, before_cumulated[200, :], label='before')

plt.legend()
plt.grid()
plt.show()

# %%
# np.save('Resources/Mermin/DIIMFP_Mermin_PMMA_cumulated.npy', DIIMFP_cumulated)
np.save('Resources/Mermin/DIIMFP_Mermin_PMMA.npy', DIIMFP_int)




