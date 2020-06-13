import matplotlib.pyplot as plt

import importlib

import numpy as np
import MC_classes_DEBER as mcd
import mapping_exp_80nm_Camscan as mapping
from functions import DEBER_functions as deber
mcd = importlib.reload(mcd)
deber = importlib.reload(deber)
mapping = importlib.reload(mapping)


# %% plot DATA
# def plot_DATA(DATA, d_PMMA, xx, zz_vac, E_cut=5):
#     print('initial size =', len(DATA))
#     DATA_cut = DATA[np.where(DATA[:, 9] > E_cut)]
#     print('cut DATA size =', len(DATA_cut))
#     fig, ax = plt.subplots(dpi=300)
#
#     for tn in range(int(np.max(DATA_cut[:, 0]))):
#         if len(np.where(DATA_cut[:, 0] == tn)[0]) == 0:
#             continue
#         now_DATA_cut = DATA_cut[np.where(DATA_cut[:, 0] == tn)]
#         ax.plot(now_DATA_cut[:, 4], now_DATA_cut[:, 6])
#
#     if d_PMMA != 0:
#         points = np.linspace(-d_PMMA * 2, d_PMMA * 2, 100)
#         ax.plot(points, np.zeros(len(points)), 'k')
#         ax.plot(points, np.ones(len(points)) * d_PMMA, 'k')
#
#     plt.plot(xx * 1e+7, zz_vac * 1e+7)
#
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.xlabel('x, nm')
#     plt.ylabel('z, nm')
#     plt.xlim(-200, 200)
#     plt.ylim(0, 200)
#     plt.gca().invert_yaxis()
#     plt.grid()
#     plt.show()

# plot_DATA(DATA, d_PMMA * 1e+7, xx, zz_vac)

# %%
# xx = np.load('data/diffusion/x_nm_exp_80_nm_Camscan.npy') * 1e-7
# hh = np.load('data/diffusion/h_nm_exp_80_nm_Camscan.npy') * 1e-7
# zz_vac = np.ones(len(xx)) * 80e-7 - hh

xx = mapping.x_centers_2nm * 1e-7
zz_vac = np.zeros(len(xx))

# %%
n_electrons = 10
zip_length = 1000
d_PMMA = 80e-7

e_DATA_PMMA_val = deber.get_e_DATA_PMMA_val(xx, zz_vac, n_electrons)
scission_matrix = deber.get_scission_matrix(e_DATA_PMMA_val)
zz_vac = deber.get_profile_after_diffusion(scission_matrix, zip_length, xx, zz_vac, d_PMMA)

# %%
T_C = 125
# N = 100
N = 100

An_array, tau_n_array = deber.get_A_tau_arrays(
    xx=xx,
    zz_vac=zz_vac,
    T_C=T_C,
    N=N
)

# %%
t = 10
l0 = 3.3e-6  # m

hh_vac = deber.get_h_at_t(xx, An_array, tau_n_array, l0, t) * 1e+2  # m -> cm

# plt.figure(dpi=300)
# plt.plot(xx, d_PMMA - zz_vac)
# plt.plot(xx, hh_vac)
#
# plt.plot(xx, deber.get_h_at_t(xx, An_array, tau_n_array, l0, 300) * 1e+2)
#
# plt.show()

zz_vac_new = d_PMMA - hh_vac

plt.figure(dpi=300)
plt.plot(xx, zz_vac)
plt.plot(xx, zz_vac_new)
plt.show()
