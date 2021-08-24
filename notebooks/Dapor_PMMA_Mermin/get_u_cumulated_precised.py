import importlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import grid
import constants as const
from functions import MC_functions as mcf

mcf = importlib.reload(mcf)
const = importlib.reload(const)
grid = importlib.reload(grid)

u_diff_prec = np.load('notebooks/Dapor_PMMA_Mermin/u_diff_prec/u_diff_from_parts.npy')

# %%
plt.figure(dpi=300)
# plt.plot(grid.EE_prec, u_diff_prec[600, :])
plt.imshow(np.log(u_diff_prec))
# plt.xlim(0, 100)
plt.show()

#%% interpolate it all
# u_diff[np.where(u_diff == 0)] = 1e-100
# DIIMFP_int = mcf.log_log_interp_2d(grid.EE_prec, grid.EE_prec, u_diff)(grid.EE, grid.EE)

# u_diff[np.where(u_diff < 1)] = 0
# DIIMFP_int[np.where(DIIMFP_int < 1)] = 0

# plt.figure(dpi=300)

# plt.loglog(grid.EE_prec, u_diff_prec[555, :])
# plt.loglog(grid.EE, DIIMFP_int[455, :], '--')

# plt.loglog(grid.EE_prec, u_diff_prec[740, :])
# plt.loglog(grid.EE, DIIMFP_int[681, :], '--')

# plt.semilogx(grid.EE_prec, u_diff[241, :])
# plt.semilogx(grid.EE, DIIMFP_int[68, :], 'o')

# plt.grid()
# plt.show()

# %%
# DIIMFP_50 = np.load('notebooks/Mermin/DIIMFP_50.npy')
#
# plt.figure(dpi=300)
#
# for i in range(20):
#     plt.semilogx(grid.EE, DIIMFP_50[i, :])
#
# for i in range(0, 1000, 50):
#     plt.semilogx(grid.EE, DIIMFP_int[i, :], '.')
#
# plt.show()

#%%
# IIMFP_test = np.zeros(len(grid.EE))
#
# for i, E in enumerate(grid.EE):
#     inds = np.where(grid.EE <= E/2)
#     IIMFP_test[i] = np.trapz(DIIMFP_int[i, inds], grid.EE[inds])
#
# plt.figure(dpi=300)
# plt.loglog(grid.EE, 1/IIMFP_test * 1e+8, label='my IMFP')
# plt.loglog(grid.EE, 1/IIMFP_int * 1e+8, '--', label='my IMFP int')
# plt.loglog(DB[:, 0], DB[:, 1], '.', label='Dapor IMFP')
# plt.legend()
# plt.grid()
# plt.show()

#%%
u_diff_cumulated_prec = np.zeros((len(grid.EE_prec), len(grid.EE_prec)))

progress_bar = tqdm(total=len(grid.EE_prec), position=0)

for i, E in enumerate(grid.EE_prec):

    inds = np.where(grid.EE_prec < E/2)[0]
    now_integral = np.trapz(u_diff_prec[i, inds], x=grid.EE_prec[inds])

    if now_integral == 0:
        u_diff_cumulated_prec[i, :] = 1
        continue

    now_cumulated_array = np.ones(len(grid.EE_prec))

    for j in inds:
        now_cumulated_array[j] = np.trapz(u_diff_prec[i, :j + 1], x=grid.EE_prec[:j + 1]) / now_integral

    u_diff_cumulated_prec[i, :] = now_cumulated_array

    progress_bar.update()

# %%
# before = np.load('Resources/Mermin/DIIMFP_Mermin_PMMA_norm.npy')
#
# before_cumulated = np.zeros(np.shape(before))
#
# for i in range(len(grid.EE)):
#     before_cumulated[i, :] = mcf.get_cumulated_array(before[i, :])

# %%
dapor_P_100 = np.loadtxt('notebooks/Dapor_PMMA_Mermin/curves/Dapor_P_100eV.txt')
dapor_P_1000 = np.loadtxt('notebooks/Dapor_PMMA_Mermin/curves/Dapor_P_1keV.txt')

plt.figure(dpi=300)
plt.semilogx(grid.EE_prec, u_diff_cumulated_prec[555, :], label='my 100')
# plt.semilogx(grid.EE_prec, u_diff_cumulated_prec[740, :], label='my 1000')

plt.semilogx(dapor_P_100[:, 0], dapor_P_100[:, 1], '--', label='dapor 100')
# plt.semilogx(dapor_P_1000[:, 0], dapor_P_1000[:, 1], '--', label='dapor 1000')

plt.xlim(1, 100)

plt.legend()

plt.grid()
plt.show()

# %% Interpolate it all
u_diff_cumulated =\
    mcf.log_log_lin_interp_2d(grid.EE_prec, grid.EE_prec, u_diff_cumulated_prec)(grid.EE, grid.EE)

u_diff_cumulated_pre_2 =\
    mcf.log_lin_interp(grid.EE_prec, u_diff_cumulated_prec, axis=1)(grid.EE)

u_diff_cumulated_2 =\
    mcf.log_lin_interp(grid.EE_prec, u_diff_cumulated_pre_2, axis=0)(grid.EE)

# %%
DIIMFP_cumulated_old = np.load('notebooks/simple_PMMA_MC/arrays_PMMA/PMMA_el_DIMFP_cumulated.npy')

# E_ind = 454
E_ind = 681

plt.figure(dpi=300)
plt.semilogx(grid.EE, u_diff_cumulated[E_ind, :])
plt.semilogx(grid.EE, u_diff_cumulated_2[E_ind, :], '--')
plt.semilogx(grid.EE, DIIMFP_cumulated_old[E_ind, :])

# plt.semilogx(dapor_P_100[:, 0], dapor_P_100[:, 1], '--', label='dapor 100')
plt.semilogx(dapor_P_1000[:, 0], dapor_P_1000[:, 1], '--', label='dapor 1000')

plt.xlim(1, 100)
plt.grid()
plt.show()