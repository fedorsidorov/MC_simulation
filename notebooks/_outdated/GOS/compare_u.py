import importlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from tqdm.auto import trange
import grid
import constants as const
from functions import MC_functions as mcf
from mpl_toolkits.mplot3d import Axes3D

mcf = importlib.reload(mcf)
const = importlib.reload(const)
grid = importlib.reload(grid)

# %%
C = np.load('Resources/GOS/_outdated/C_IIMFP.npy')
O = np.load('Resources/GOS/_outdated/O_IIMFP.npy')
C_b = np.load('Resources/GOS/C_IIMFP_E_bind.npy')
O_b = np.load('Resources/GOS/O_IIMFP_E_bind.npy')

plt.figure(dpi=300)
plt.loglog(grid.EE, C, '--')
plt.loglog(grid.EE, C_b)
plt.loglog(grid.EE, O, '--')
plt.loglog(grid.EE, O_b)
plt.show()
