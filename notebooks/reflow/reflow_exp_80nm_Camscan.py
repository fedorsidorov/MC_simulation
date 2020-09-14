import importlib

import matplotlib.pyplot as plt
import numpy as np

from mapping import mapping_3p3um_80nm as mapping
from functions import MC_functions as mf
from functions import reflow_functions as rf

rf = importlib.reload(rf)

# %%
x_arr_nm = np.load('data/diffusion/x_nm_exp_80_nm_Camscan.npy')
z_arr_nm = np.load('data/diffusion/h_nm_exp_80_nm_Camscan.npy')

x_arr_nm[0] = mapping.x_min
x_arr_nm[-1] = mapping.x_max

x_arr = x_arr_nm * 1e-9
z_arr = z_arr_nm * 1e-9

# plt.figure(dpi=300)
# plt.plot(x_arr_nm, z_arr_nm)
# plt.show()

# %%
l0_nm = mapping.l_x
h0_nm = mapping.l_z

l0 = mapping.l_x * 1e-9
h0 = mapping.l_z * 1e-9

T_C = 125  # C
# N = 50
N = 100

eta = rf.get_PMMA_950K_viscosity(T_C)
gamma = rf.get_PMMA_surface_tension(T_C)

xx_nm = np.linspace(mapping.x_min, mapping.x_max, 100)
xx = xx_nm * 1e-9

zz_nm = mf.lin_lin_interp(x_arr_nm, z_arr_nm)(xx_nm)
zz = zz_nm * 1e-9

# plt.figure(dpi=300)
# plt.plot(xx_nm, zz_nm)
# plt.show()

# %%
An_array_nm = rf.get_An_array(xx_nm, zz_nm, l0_nm, N=N)
An_array = An_array_nm * 1e-9

# %%
tau_n_array = rf.get_tau_n_array(eta=eta, gamma=gamma, h0=h0, l0=l0, N=N)

# plt.figure(dpi=300)
# plt.semilogy(tau_n_array)
# plt.semilogy(tau_n_easy_array, '--')
# plt.show()

# %%
result = np.zeros(len(xx))
result += An_array[0]

for n in range(1, len(An_array)):
    result += An_array[n] * np.cos(2 * np.pi * n * xx / l0)

plt.figure(dpi=300)
# plt.plot(xx, result)
# plt.plot(xx_nm, zz_nm)
plt.plot(xx, zz)
plt.plot(xx, result, '--')
plt.show()

# %%
t = 0 * 60  # s
zz_final = rf.get_h_at_t(xx, An_array, tau_n_array, l0, t)

plt.figure(dpi=300)
plt.plot(xx, zz)
plt.plot(xx, zz_final, '--')
plt.show()
