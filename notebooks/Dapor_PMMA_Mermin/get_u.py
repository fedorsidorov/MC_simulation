import importlib
import matplotlib.pyplot as plt
import numpy as np

import grid
import constants as const
from functions import MC_functions as mcf

mcf = importlib.reload(mcf)
const = importlib.reload(const)
grid = importlib.reload(grid)

DIIMFP = np.load('notebooks/Dapor_PMMA_Mermin/u_diff_prec/u_diff_from_parts.npy')

#%% test IMFP - OK
IIMFP = np.zeros(len(grid.EE_prec))

for i, E in enumerate(grid.EE_prec):
    inds = np.where(grid.EE_prec <= E/2)
    IIMFP[i] = np.trapz(DIIMFP[i, inds], grid.EE_prec[inds])


IIMFP_int = mcf.log_log_interp(grid.EE_prec, IIMFP)(grid.EE)
DB = np.loadtxt('notebooks/Dapor_PMMA_Mermin/curves/Dapor_BOOK_grey.txt')
paper = np.loadtxt('notebooks/_outdated/MELF-GOS/MELF-GOS.txt')

u_nm = IIMFP_int * 1e-7

plt.figure(dpi=300)

plt.semilogx(grid.EE, 1 / (u_nm * 1e-1), label='my IMFP')
plt.semilogx(DB[:, 0], DB[:, 1], '.', label='Dapor IMFP')
plt.semilogx(paper[:, 0], paper[:, 1] * 10, '.', label='Dapor IMFP 2015')

plt.xlim(1e+1, 1e+3)
plt.ylim(0, 40)

plt.grid()
plt.legend()
plt.show()

# %%
# np.save('notebooks/Dapor_PMMA_Mermin/final_arrays/u_nm.npy', u_nm)

