import numpy as np
import matplotlib.pyplot as plt
import importlib

import grid

grid = importlib.reload(grid)

# %%
arr = np.load('Resources/Mermin/DIIMFP_Mermin_PMMA_norm.npy')

arr_c = np.zeros(np.shape(arr))

for i in range(len(arr)):
    for j in range(len(arr[0])):
        arr_c[i, j] = np.sum(arr[i, :j+1])


# %%
plt.figure(dpi=300)
# plt.semilogx(grid.EE, arr[50, :])
plt.semilogx(grid.EE, arr[613, :])
plt.semilogx(grid.EE, arr[720, :])

# plt.xlim(0, 100)

plt.show()


# %%
np.save('Resources/Mermin/DIIMFP_Mermin_PMMA_norm_cumulated.npy', arr_c)



