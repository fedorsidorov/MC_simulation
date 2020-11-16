import importlib

import matplotlib.pyplot as plt
import numpy as np

from mapping import mapping_3p3um_80nm as mapping
from functions import reflow_functions as rf
from functions import DEBER_functions as deber

deber = importlib.reload(deber)
mapping = importlib.reload(mapping)
rf = importlib.reload(rf)

# %%
xx = np.load('xx_diffusion.npy')
zz_vac_list = np.load('zz_vac_list_700.npy')
hh_vac_list = 80e-7 - zz_vac_list

plt.figure(dpi=300)
plt.plot(xx, hh_vac_list[-1])
plt.show()

# %%
zz_vac = zz_vac_list[-1, :]
hh_vac = hh_vac_list[-1, :]

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
t = 200

hh_vac_final = deber.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, t) * 1e+2  # m -> cm

_, _ = plt.subplots(dpi=600)
fig = plt.gcf()
fig.set_size_inches(3.5, 3.5)

font_size = 8

exp_prof = np.loadtxt('data/DEBER_profiles/Camscan_80nm/Camscan_new.txt')

beg, end = 40, 170
exp_xx = exp_prof[beg:end, 0] - 2960
exp_yy = exp_prof[beg:end, 1] + 47

plt.plot(exp_xx, exp_yy, label='эcперимент')
plt.plot(xx*1e+7, hh_vac_final*1e+7, label='моделирование')

plt.legend()

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

plt.legend(fontsize=font_size)
plt.xlim(-2000, 2000)
plt.ylim(40, 80)
plt.xlabel('$x$, нм', fontsize=font_size)
plt.ylabel('$z$, нм', fontsize=font_size)
plt.grid()

# plt.show()
plt.savefig('final.tiff', dpi=600)
