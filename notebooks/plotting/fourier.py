import importlib

import matplotlib.pyplot as plt
import numpy as np

import mapping_exp_80nm_Camscan as mapping
from functions import reflow_functions as rf
from functions import DEBER_functions as deber

deber = importlib.reload(deber)
mapping = importlib.reload(mapping)
rf = importlib.reload(rf)


# %%
xx = np.load('xx_diffusion.npy')
zz_vac_list = np.load('zz_vac_diffused_list.npy')
hh_vac_list = 80e-7 - zz_vac_list

# plt.figure(dpi=300)
# plt.plot(xx, hh_vac_list[0])
# plt.show()

zz_vac = zz_vac_list[0, :]
hh_vac = hh_vac_list[0, :]

d_PMMA = 80e-7
N_fourier = 21
T_C = 125
eta = 1e+5
gamma = rf.get_PMMA_surface_tension(125)
N = 21
h0 = np.trapz(hh_vac, x=xx) / (np.max(xx) - np.min(xx)) * 1e-2  # cm -> m
l0 = 3.3e-6  # m

An_array, Bn_array, tau_n_array = deber.get_An_Bn_tau_arrays(eta, xx, zz_vac, T_C, N_fourier)

# %%
t0 = 0
t1 = 10

hh_vac_0 = deber.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, t0) * 1e+2  # m -> cm
hh_vac_1 = deber.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, t1) * 1e+2  # m -> cm

# %%
_, _ = plt.subplots(dpi=600)
fig = plt.gcf()
fig.set_size_inches(5, 5)

font_size = 13

plt.plot(xx*1e+7, hh_vac_list[0]*1e+7, label='после диффузии')
plt.plot(xx*1e+7, hh_vac_0*1e+7, label='фурье-представление')
plt.plot(xx*1e+7, hh_vac_1*1e+7, 'r', label='после растекания')

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

plt.legend(loc='upper right', fontsize=10)
plt.xlim(-1500, 1500)
plt.ylim(76, 81)
plt.xlabel('$x$, нм', fontsize=font_size)
plt.ylabel('$z$, нм', fontsize=font_size)
plt.grid()

# plt.show()
plt.savefig('fourier.tiff', dpi=600)


# %%
arr = np.array((
     (14, 28),
     (19, 27),
     (23, 30),
     (17, 23),
     (10, 31),
     (4, 18),
     (23, 21),
     (15, 29),
     (23, 21),
     (19, 38),
     (20, 27),
     (11, 40),
     (17, 22)
))

