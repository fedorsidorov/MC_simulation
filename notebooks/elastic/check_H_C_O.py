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


def get_Ruth_cs(Z):
    alpha = const.m * const.e**4 * np.pi**2 * Z**(2/3) / (const.h**2 * grid.EE * const.eV)
    cs = np.pi * Z**2 * const.e**4 / ((grid.EE * const.eV)**2 * alpha * (alpha + 2))
    return cs


kind = 'easy'

H_diff = np.load('notebooks/elastic/final_arrays/' + kind + '/H/H_' + kind + '_diff_cs.npy')
C_diff = np.load('notebooks/elastic/final_arrays/' + kind + '/C/C_' + kind + '_diff_cs.npy')
O_diff = np.load('notebooks/elastic/final_arrays/' + kind + '/O/O_' + kind + '_diff_cs.npy')

E_ind = 700
E = grid.EE[E_ind]

H_diff_R = get_Ruth_diff_cs(1, E)
C_diff_R = get_Ruth_diff_cs(6, E)
O_diff_R = get_Ruth_diff_cs(8, E)

# %
plt.figure(dpi=300)

plt.semilogy(grid.THETA_deg, H_diff[E_ind, :] * 1e+16, label='H elsepa')
plt.semilogy(grid.THETA_deg, H_diff_R * 1e+16, label='H Ruth')

plt.semilogy(grid.THETA_deg, C_diff[E_ind, :] * 1e+16, label='C elsepa')
plt.semilogy(grid.THETA_deg, C_diff_R * 1e+16, label='C Ruth')

plt.semilogy(grid.THETA_deg, O_diff[E_ind, :] * 1e+16, label='O elsepa')
plt.semilogy(grid.THETA_deg, O_diff_R * 1e+16, label='O Ruth')

plt.ylim(1e-5, 10)

plt.legend()
plt.grid()
plt.show()

# %%
kind = 'muffin'

H_cs = np.load('notebooks/elastic/final_arrays/' + kind + '/H/H_' + kind + '_cs.npy')
C_cs = np.load('notebooks/elastic/final_arrays/' + kind + '/C/C_' + kind + '_cs.npy')
O_cs = np.load('notebooks/elastic/final_arrays/' + kind + '/O/O_' + kind + '_cs.npy')

H_cs_R = get_Ruth_cs(1)
C_cs_R = get_Ruth_cs(6)
O_cs_R = get_Ruth_cs(8)

plt.figure(dpi=300)

plt.semilogy(grid.THETA_deg, H_cs * 1e+16, label='H elsepa')
plt.semilogy(grid.THETA_deg, H_cs_R * 1e+16, label='H Ruth')

plt.semilogy(grid.THETA_deg, C_cs * 1e+16, label='C elsepa')
plt.semilogy(grid.THETA_deg, C_cs_R * 1e+16, label='C Ruth')

plt.semilogy(grid.THETA_deg, O_cs * 1e+16, label='O elsepa')
plt.semilogy(grid.THETA_deg, O_cs_R * 1e+16, label='O Ruth')

plt.ylim(1e-3, 1e+3)

plt.legend()
plt.grid()
plt.show()

# %%
E_ind = 400
E = grid.EE[E_ind]

H_diff_R = get_Ruth_diff_cs(1, E)
C_diff_R = get_Ruth_diff_cs(6, E)
O_diff_R = get_Ruth_diff_cs(8, E)

MMA_diff = H_diff*8 + C_diff*5 + O_diff*2
MMA_diff_R = H_diff_R*8 + C_diff_R*5 + O_diff_R*2

plt.figure(dpi=300)

plt.semilogy(grid.THETA_deg, MMA_diff[E_ind, :], label='H elsepa')
plt.semilogy(grid.THETA_deg, MMA_diff_R, label='H Ruth')

plt.legend()
plt.grid()
plt.show()

# %%
MMA_cs = H_cs*8 + C_cs*5 + O_cs*2
MMA_cs_R = H_cs_R*8 + C_cs_R*5 + O_cs_R*2

PMMA_u = const.n_MMA * MMA_cs * 1e-7
PMMA_u_R = const.n_MMA * MMA_cs_R * 1e-7

PMMA_val_u = np.load('notebooks/simple_PMMA_MC/PMMA/arrays_PMMA/PMMA_val_IMFP_nm.npy')
PMMA_el_u = np.load('notebooks/simple_PMMA_MC/PMMA/arrays_PMMA/PMMA_el_IMFP_nm.npy')

plt.figure(dpi=300)

plt.loglog(grid.EE, PMMA_u, label='PMMA elsepa')
plt.loglog(grid.EE, PMMA_u_R, label='PMMA Ruth')
plt.loglog(grid.EE, PMMA_val_u, label='PMMA val')
plt.loglog(grid.EE, PMMA_el_u, label='PMMA el old')

plt.legend()
plt.grid()
plt.show()
