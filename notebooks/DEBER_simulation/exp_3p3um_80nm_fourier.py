import importlib
import warnings

import matplotlib.pyplot as plt
import numpy as np

import MC_classes_nm as mcc
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
xx = mm.x_centers_10nm  # nm
zz_vac = np.zeros(len(xx))  # nm

N_steps = 32
n_electrons_10s = 84

E0 = 20e+3
r_beam = 100

# zip_length = 300
# zip_length = 600
# zip_length = 700
zip_length = 1000

T_C = 125
weight = 0.275
N_fourier = 20

t_step = 10  # s
l0 = 3300  # nm
l0_m = l0 * 1e-9

# eta = 1e+6  # Pa s
eta = 1e+5  # Pa s

gamma = 34e-3  # N / m

# %%
# for i in range(N_steps):
for i in range(5):

    print('step №', i)

    print('simulate e-beam scattering and scissions')
    e_DATA, e_DATA_PMMA_val = deber.get_e_DATA_PMMA_val(xx, zz_vac, mm.d_PMMA_nm, n_electrons_10s, E0, r_beam)
    scission_matrix, e_matrix_E_dep = deber.get_scission_matrix(e_DATA_PMMA_val, weight=weight)

    print('simulate diffusion')
    zz_vac = df.get_profile_after_diffusion(scission_matrix, zip_length, xx, zz_vac, mm.d_PMMA_cm, mult=10)

    zz_surface = 80 - zz_vac

    xx_nm = np.concatenate(([mm.x_min], xx, [mm.x_max]))
    zz_surface_nm = np.concatenate(([zz_surface[0]], zz_surface, [zz_surface[-1]]))

    print('simulate reflow')
    An_array_nm = ff.get_An_array(xx_nm, zz_surface_nm, l0, N_fourier)
    Bn_array_nm = ff.get_Bn_array(xx_nm, zz_surface_nm, l0, N_fourier)

    An_array_m = An_array_nm * 1e-9
    Bn_array_m = Bn_array_nm * 1e-9

    tau_n_array = ff.get_tau_n_easy_array(eta, gamma, h0=An_array_m[0], l0=l0_m, N=N_fourier)

    hh_vac = ff.get_h_at_t(xx * 1e-9, An_array_m, Bn_array_m, tau_n_array, l0_m, t_step) * 1e+2  # m -> cm
    zz_vac_reflowed = mm.d_PMMA_cm - hh_vac

    plt.figure(dpi=300)
    plt.plot(xx, mm.d_PMMA_nm - zz_vac, label='after diffusion')
    plt.plot(xx, (mm.d_PMMA_cm - zz_vac_reflowed) * 1e+7, label='after reflow')
    plt.title('step № ' + str(i + 1) + ' of ' + str(N_steps))
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.legend()
    plt.grid()
    plt.show()

    zz_vac = zz_vac_reflowed * 1e+7

# %% cooling
plt.figure(dpi=300)
plt.plot(xx, mm.d_PMMA_nm - zz_vac)
plt.show()

# %%
t_after = 200

hh_vac_new = ff.get_h_at_t(xx * 1e-9, An_array_m, Bn_array_m, tau_n_array, l0_m, t_after) * 1e+2  # m -> cm

zz_vac_reflowed = mm.d_PMMA_cm - hh_vac_new

plt.figure(dpi=300)
plt.plot(xx, (mm.d_PMMA_cm - zz_vac_reflowed) * 1e+7, label='after reflow')
# plt.plot(xx, hh_vac_new * 1e+7, label='after reflow')
plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.legend()
plt.grid()
plt.show()

# %%
np.save('notebooks/DEBER_simulation/zz_vac_600_1e+5_final.npy', zz_vac_reflowed)
np.save('notebooks/DEBER_simulation/zz_surface_600_1e+5_final.npy', (mm.d_PMMA_cm - zz_vac_reflowed) * 1e+7)


# %%
np.save('notebooks/DEBER_simulation/xx_fourier_5.npy', xx)
np.save('notebooks/DEBER_simulation/zz_vac_fourier_5.npy', zz_vac)
