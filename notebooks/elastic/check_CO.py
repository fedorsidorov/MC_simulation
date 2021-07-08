import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import constants as const
import grid as grid
from tqdm import tqdm

const = importlib.reload(const)
grid = importlib.reload(grid)


# %%
def get_Ruth_diff_cs(Z, E):

    alpha = const.m * const.e**4 * np.pi**2 * Z**(2/3) / (const.h**2 * E * const.eV)
    diff_cs = Z**2 * const.e**4 / (4 * (E * const.eV)**2 * (1 - np.cos(grid.THETA_rad) + alpha)**2)

    return diff_cs


# %%
ans = np.loadtxt('notebooks/elastic/curves/CO_Dapor.txt')

DCS_C = np.loadtxt('notebooks/elastic/curves/DCS_C_500_eV.txt')
DCS_O = np.loadtxt('notebooks/elastic/curves/DCS_O_500_eV.txt')
DCS_CO = (DCS_C[:, 1] + DCS_O[:, 1]) * 2.8e-21 * 1e+20

dcs_ruth_C = get_Ruth_diff_cs(6, 500)
dcs_ruth_O = get_Ruth_diff_cs(8, 500)

dcs_ruth_CO = (dcs_ruth_C + dcs_ruth_O) * 1e+16

# %% ELSEPA
# C_dcs = np.load('notebooks/elastic/final_arrays/muffin/C/C_muffin_diff_cs.npy')
# O_dcs = np.load('notebooks/elastic/final_arrays/muffin/O/O_muffin_diff_cs.npy')

C_dcs = np.load('notebooks/elastic/final_arrays/easy/C/C_easy_diff_cs.npy')
O_dcs = np.load('notebooks/elastic/final_arrays/easy/O/O_easy_diff_cs.npy')

CO_dcs = (C_dcs[613, :] + O_dcs[613, :]) * 1e+16

# %%
plt.figure(dpi=300)
plt.semilogy(ans[:, 0], ans[:, 1], label='Dapor Book')
plt.semilogy(DCS_C[:, 0], DCS_CO, label='NIST')
plt.semilogy(grid.THETA_deg, dcs_ruth_CO, '-', label='Screened Rutherford')
plt.semilogy(grid.THETA_deg, CO_dcs, '--', label='ELSEPA')

plt.ylim(0.001, 100)

plt.legend()
plt.grid()
plt.show()

# %%
bns = np.loadtxt('notebooks/elastic/curves/Dapor_Ruth_Au.txt')

dcs_ruth = get_Ruth_diff_cs(79, 1100) * 1e+16

plt.figure(dpi=300)
plt.semilogy(bns[:, 0], bns[:, 1], label='Dapor Book')
plt.semilogy(grid.THETA_deg, dcs_ruth, '-', label='Screened Rutherford')

plt.ylim(0.001, 100)

plt.legend()
plt.grid()
plt.show()
