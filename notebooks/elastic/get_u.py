import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import grid
import constants as const
from functions import MC_functions as mcf

const = importlib.reload(const)
grid = importlib.reload(grid)
mcf = importlib.reload(mcf)

#%%
for kind in ['easy', 'atomic', 'muffin']:

    cs_H = np.load('notebooks/elastic/final_arrays/' + kind + '/H/H_' + kind + '_cs.npy')
    cs_C = np.load('notebooks/elastic/final_arrays/' + kind + '/C/C_' + kind + '_cs.npy')
    cs_O = np.load('notebooks/elastic/final_arrays/' + kind + '/O/O_' + kind + '_cs.npy')
    cs_Si = np.load('notebooks/elastic/final_arrays/' + kind + '/Si/Si_' + kind + '_cs.npy')

    cs_MMA = const.N_H_MMA * cs_H + const.N_C_MMA * cs_C + const.N_O_MMA * cs_O
    
    u_PMMA_nm = cs_MMA * const.n_MMA * 1e-7
    u_Si_nm = cs_Si * const.n_Si * 1e-7

    np.save('notebooks/elastic/final_arrays/PMMA/MMA_' + kind + '_cs.npy', cs_MMA)
    np.save('notebooks/elastic/final_arrays/PMMA/PMMA_' + kind + '_u_nm.npy', u_PMMA_nm)
    
    np.save('notebooks/elastic/final_arrays/Si/Si_' + kind + '_cs.npy', cs_Si)
    np.save('notebooks/elastic/final_arrays/Si/Si_' + kind + '_u_nm.npy', u_Si_nm)

# %%
u_PMMA = np.load('notebooks/elastic/final_arrays/PMMA/PMMA_easy_u_extrap_nm.npy')
u_Si = np.load('notebooks/elastic/final_arrays/Si/Si_easy_u_extrap_nm.npy')

plt.figure(dpi=300)
plt.loglog(grid.EE, u_PMMA)
plt.loglog(grid.EE, u_Si)
plt.show()



