import numpy as np
import matplotlib.pyplot as plt
import importlib

import constants as const
import grid

const = importlib.reload(const)
grid = importlib.reload(grid)

# %% total cs
g4_el_u = np.load('Resources/MuElec/elastic_u.npy')
my_el_u = np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/ELSEPA/Si/Si_muffin_u.npy')

plt.figure(dpi=300)
plt.loglog(grid.EE, g4_el_u)
plt.loglog(grid.EE, my_el_u)
plt.show()

# %% differential cs






