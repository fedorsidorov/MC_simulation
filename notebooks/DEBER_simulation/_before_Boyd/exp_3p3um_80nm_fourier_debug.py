import importlib
import warnings

import matplotlib.pyplot as plt
import numpy as np

from _outdated import MC_classes_nm as mcc
from mapping import mapping_3p3um_80nm as mm
from functions import DEBER_functions as deber
from functions import diffusion_functions as df
from functions import fourier_functions as ff
from functions import MC_functions as mcf

deber = importlib.reload(deber)
mcc = importlib.reload(mcc)
mcf = importlib.reload(mcf)
mm = importlib.reload(mm)
df = importlib.reload(df)
ff = importlib.reload(ff)

warnings.filterwarnings('ignore')

# %%
xx = mm.x_centers_5nm
zz_vac = np.zeros(len(xx))

N_ten_electrons = 32
# N_ten_electrons = 5
n_electrons = int(78 / 2)  # doubling !!!

E0 = 20e+3
r_beam = 100

zip_length = 1000
# zip_length = 2000

T_C = 125
weight = 0.275
# N_fourier = 100
N_fourier = 20

t = 10  # s
l0 = 3.3e-6  # m
eta = 5e+6  # Pa s
gamma = 34e-3  # N / m

# %%
# for i in range(N_ten_electrons):
i = 1

print('electron group', i + 1, 'of', N_ten_electrons)

plt.figure(dpi=300)
plt.plot(xx, zz_vac)
plt.title('suda herachat electrons')
plt.show()

print('simulate e-beam scattering')
e_DATA, e_DATA_PMMA_val = deber.get_e_DATA_PMMA_val(xx, zz_vac, mm.d_PMMA_nm, n_electrons, E0, r_beam)

# %%
scission_matrix, e_matrix_E_dep = deber.get_scission_matrix(e_DATA_PMMA_val, weight=weight)

# %%
print('simulate diffusion')
zz_vac = df.get_profile_after_diffusion(scission_matrix, zip_length, xx, zz_vac, mm.d_PMMA_cm, mult=10)

# %%
plt.figure(dpi=300)
plt.plot(xx, (80e-7 - zz_vac)*1e+7)
plt.show()

# %%
zz_surface = (80e-7 - zz_vac)*1e+7

xx_nm = np.concatenate(([mm.x_min], xx, [mm.x_max]))
zz_surface_nm = np.concatenate(([zz_surface[0]], zz_surface, [zz_surface[-1]]))
l0_nm = l0 * 1e+9

# plt.figure(dpi=300)
# plt.plot(xx_nm, zz_surface_nm)
# plt.show()

# %%
print('simulate reflow')
An_array_nm = ff.get_An_array(xx_nm, zz_surface_nm, l0_nm, N_fourier)
Bn_array_nm = ff.get_Bn_array(xx_nm, zz_surface_nm, l0_nm, N_fourier)

An_array_m = An_array_nm * 1e-9
Bn_array_m = Bn_array_nm * 1e-9

tau_n_array = ff.get_tau_n_easy_array(eta, gamma, h0=An_array_m[0], l0=l0, N=N_fourier)

# %%
now_t = 10

hh_vac = ff.get_h_at_t(xx*1e-9, An_array_m, Bn_array_m, tau_n_array, l0, now_t) * 1e+2  # m -> cm
zz_vac_reflowed = mm.d_PMMA_cm - hh_vac

plt.figure(dpi=300)
plt.plot(xx, (mm.d_PMMA_cm - zz_vac) * 1e+7, label='after diffusion')
plt.plot(xx, (mm.d_PMMA_cm - zz_vac_reflowed) * 1e+7, label='after reflow')
plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.legend()
plt.grid()
plt.show()

# %%
# plt.figure(dpi=300)
# plt.plot(xx, zz_vac_reflowed * 1e+7)
# plt.show()

zz_vac = zz_vac_reflowed * 1e+7





# %%
# plt.figure(dpi=300)
# plt.plot(xx * 1e+7, (d_PMMA - zz_vac) * 1e+7)
# plt.show()

# t_after = 100000
#
# hh_vac_new = deber.get_h_at_t(xx, An_array, tau_n_array, l0, t_after) * 1e+2  # m -> cm
#
# plt.figure(dpi=300)
# plt.plot(xx * 1e+7, hh_vac_new * 1e+7, label='after reflow')
# plt.xlabel('x, nm')
# plt.ylabel('z, nm')
# plt.legend()
# plt.grid()
# plt.show()
