import numpy as np
import matplotlib.pyplot as plt
import importlib

import constants as const
import grid
from functions import MC_functions as util

const = importlib.reload(const)
grid = importlib.reload(grid)
util = importlib.reload(util)

# %% u
u_mu = np.load('notebooks/MuElec/MuElec_inelastic_arrays/u_ee_6.npy')

u_elf_0 = np.load('notebooks/Akkerman_Si_5osc/u/u_0.npy')
u_elf_1 = np.load('notebooks/Akkerman_Si_5osc/u/u_1.npy')
u_elf_2 = np.load('notebooks/Akkerman_Si_5osc/u/u_2.npy')
u_elf_3 = np.load('notebooks/Akkerman_Si_5osc/u/u_3.npy')
u_elf_4 = np.load('notebooks/Akkerman_Si_5osc/u/u_4.npy')

u_elf = u_elf_0 + u_elf_1 + u_elf_2 + u_elf_3 + u_elf_4

plt.figure(dpi=300)
plt.loglog(grid.EE, np.sum(u_mu, axis=1), label='MuElec')
plt.loglog(grid.EE, u_elf, label='ELF 5 osc')
plt.grid()
plt.legend()
plt.show()

# %% sigma diff cumulated
cum_elf_0 = np.load('notebooks/Akkerman_Si_5osc/u_diff_cumulated/u_diff_0_cumulated.npy')
cum_elf_1 = np.load('notebooks/Akkerman_Si_5osc/u_diff_cumulated/u_diff_1_cumulated.npy')
cum_elf_2 = np.load('notebooks/Akkerman_Si_5osc/u_diff_cumulated/u_diff_2_cumulated.npy')
cum_elf_3 = np.load('notebooks/Akkerman_Si_5osc/u_diff_cumulated/u_diff_3_cumulated.npy')
cum_elf_4 = np.load('notebooks/Akkerman_Si_5osc/u_diff_cumulated/u_diff_4_cumulated.npy')

cum_mu = np.load('notebooks/MuElec/MuElec_inelastic_arrays/u_diff_cumulated_6.npy')

plt.figure(dpi=300)

ind = 900

plt.semilogx(grid.EE, cum_elf_4[ind, :])
plt.semilogx(grid.EE, cum_mu[5, ind, :])

plt.show()



