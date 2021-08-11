import importlib
import matplotlib.pyplot as plt
import numpy as np

import grid
import constants as const
from functions import MC_functions as mcf

mcf = importlib.reload(mcf)
const = importlib.reload(const)
grid = importlib.reload(grid)

# %%
d_1 = np.load('notebooks/Mermin/u_diff_prec/DIIMFP_000-824.npy')
d_2 = np.load('notebooks/Mermin/u_diff_prec/DIIMFP_824-950.npy')
d_3 = np.load('notebooks/Mermin/u_diff_prec/DIIMFP_950-999.npy')

diff_arr = np.zeros((len(grid.EE), len(grid.EE)))

diff_arr[0:824, :] = d_1[0:824, :]
diff_arr[824:950, :] = d_2[0:126, :]
diff_arr[950:, :] = d_3[:50, :]


# %%
# np.save('notebooks/Mermin/u_diff_prec/u_diff_prec.npy', diff_arr)

# %% check IIMPFP
# u = np.zeros(len(grid.EE))
u = np.zeros(len(grid.EE_prec))

for i, E in enumerate(grid.EE):
    # inds = np.where(grid.EE < E / 2)[0]
    inds = np.where(grid.EE_prec < E / 2)[0]
    # u[i] = np.trapz(diff_arr[i, inds], x=grid.EE[inds])
    u[i] = np.trapz(diff_arr[i, inds], x=grid.EE_prec[inds])

# %%
u_pre = np.load('Resources/Mermin/IIMFP_Mermin_PMMA.npy')

plt.figure(dpi=300)
plt.loglog(grid.EE_prec, u)
plt.loglog(grid.EE, u_pre, '--')
plt.show()




