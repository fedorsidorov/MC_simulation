import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import indexes
import grid as grid
import constants as const
from functions import MC_functions as mcf
from tqdm import tqdm

indexes = importlib.reload(indexes)
const = importlib.reload(const)
grid = importlib.reload(grid)
mcf = importlib.reload(mcf)


#%%
for model in ['easy', 'atomic', 'muffin']:

    print(model)

    diff_cs_H =\
        np.load(os.path.join('notebooks/elastic/final_arrays', model, 'H', 'H' + '_' + model + '_diff_cs_extrap.npy'))
    diff_cs_C =\
        np.load(os.path.join('notebooks/elastic/final_arrays', model, 'C', 'C' + '_' + model + '_diff_cs_extrap.npy'))
    diff_cs_O =\
        np.load(os.path.join('notebooks/elastic/final_arrays', model, 'O', 'O' + '_' + model + '_diff_cs_extrap.npy'))
    
    diff_cs_MMA = const.N_H_MMA * diff_cs_H + const.N_C_MMA * diff_cs_C + const.N_O_MMA * diff_cs_O

    diff_cs_Si =\
        np.load(os.path.join('notebooks/elastic/final_arrays', model, 'Si', 'Si' + '_' + model + '_diff_cs_extrap.npy'))

    diff_cs_MMA_cumulated = np.zeros(np.shape(diff_cs_MMA))
    diff_cs_Si_cumulated = np.zeros(np.shape(diff_cs_Si))

    progress_bar = tqdm(total=len(grid.EE), position=0)

    for i in range(len(grid.EE)):

        now_diff_cs_MMA_plane = diff_cs_MMA[i, :] * 2 * np.pi * np.sin(grid.THETA_rad)
        now_diff_cs_Si_plane = diff_cs_Si[i, :] * 2 * np.pi * np.sin(grid.THETA_rad)

        for j in range(len(grid.THETA_rad)):

            diff_cs_MMA_cumulated[i, j] =\
                np.trapz(now_diff_cs_MMA_plane[:j+1], x=grid.THETA_rad[:j+1]) /\
                np.trapz(now_diff_cs_MMA_plane, x=grid.THETA_rad)

            diff_cs_Si_cumulated[i, j] =\
                np.trapz(now_diff_cs_Si_plane[:j+1], x=grid.THETA_rad[:j+1]) /\
                np.trapz(now_diff_cs_MMA_plane, x=grid.THETA_rad)

        progress_bar.update()

    np.save('notebooks/elastic/final_arrays/PMMA/PMMA_diff_cs_cumulated_' + model + '_extrap_+1.npy',
            diff_cs_MMA_cumulated)

    np.save('notebooks/elastic/final_arrays/Si/Si_diff_cs_cumulated_' + model + '_extrap_+1.npy',
            diff_cs_Si_cumulated)

# %%
element = 'PMMA'

diff_cs_cumulated =\
    np.load('notebooks/elastic/final_arrays/' + element + '/' + element + '_diff_cs_cumulated_atomic_+1.npy')

plt.figure(dpi=300)

plt.semilogx(grid.THETA_deg, diff_cs_cumulated[indexes.E_100])
plt.semilogx(grid.THETA_deg, diff_cs_cumulated[indexes.E_1000])
plt.semilogx(grid.THETA_deg, diff_cs_cumulated[indexes.E_5000])
plt.semilogx(grid.THETA_deg, diff_cs_cumulated[indexes.E_10000])
plt.semilogx(grid.THETA_deg, diff_cs_cumulated[indexes.E_20000])

plt.grid()
plt.show()

# %%
ans = np.load('notebooks/elastic/final_arrays/atomic/C/C_atomic_diff_cs.npy')





