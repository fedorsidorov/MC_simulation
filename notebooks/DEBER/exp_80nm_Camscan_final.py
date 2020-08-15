import importlib
import warnings

import matplotlib.pyplot as plt
import numpy as np

import MC_classes_DEBER as mcd
import mapping_exp_80nm_Camscan as mapping
from functions import DEBER_functions as deber

mcd = importlib.reload(mcd)
deber = importlib.reload(deber)
mapping = importlib.reload(mapping)

warnings.filterwarnings('ignore')

# %%
xx = mapping.x_centers_2nm * 1e-7
zz_vac = np.zeros(len(xx))
d_PMMA = 80e-7

N_ten_electrons = 32
# N_ten_electrons = 10
# n_electrons = 78
n_electrons = int(78 / 2)  # doubling !!!

T_C = 125
# N_fourier = 100
N_fourier = 20

t = 10  # s
l0 = 3.3e-6  # m
# eta = 5e+6  # Pa s
eta = 1e+5  # Pa s

# zip_length = 1000
# zip_length = 2000  # too much
# zip_length = 500
zip_length = 700

zz_vac_list = []

for i in range(N_ten_electrons):
    print('electron group', i + 1, 'of', N_ten_electrons)

    print('simulate e-beam scattering')
    e_DATA_PMMA_val = deber.get_e_DATA_PMMA_val(xx, zz_vac, n_electrons)
    scission_matrix = deber.get_scission_matrix(e_DATA_PMMA_val)

    print('simulate diffusion')
    zz_vac_diffused = deber.get_profile_after_diffusion(
        scission_matrix=scission_matrix,
        zip_length=zip_length,
        xx=xx,
        zz_vac=zz_vac,
        d_PMMA=d_PMMA,
        double=True
    )

    print('simulate reflow')
    An_array, tau_n_array = deber.get_An_tau_arrays(eta, xx, zz_vac_diffused, T_C, N_fourier)

    hh_vac = deber.get_h_at_t_even(xx, An_array, tau_n_array, l0, t) * 1e+2  # m -> cm
    zz_vac_reflowed = d_PMMA - hh_vac

    zz_vac_list.append(zz_vac_reflowed)

    plt.figure(dpi=300)
    plt.plot(xx * 1e+7, (d_PMMA - zz_vac_diffused) * 1e+7, label='after diffusion')
    plt.plot(xx * 1e+7, (d_PMMA - zz_vac_reflowed) * 1e+7, label='after reflow')
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.legend()
    plt.grid()
    plt.show()

    zz_vac = zz_vac_reflowed

# %%
# plt.figure(dpi=300)
# plt.plot(xx * 1e+7, (d_PMMA - zz_vac) * 1e+7)
# plt.show()

t_after = 500

hh_vac_new = deber.get_h_at_t_even(xx, An_array, tau_n_array, l0, t_after) * 1e+2  # m -> cm

plt.figure(dpi=300)
plt.plot(xx * 1e+7, hh_vac_new * 1e+7, label='after reflow')
plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.legend()
plt.grid()
plt.show()

# plt.savefig('eta=1e+5_zip=500_after=150s.png', dpi=300)
