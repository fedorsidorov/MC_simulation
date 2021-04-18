# %%
import importlib
import constants
import numpy as np
from mapping import mapping_3p3um_80nm as mapping
from functions import MC_functions as mcf
from functions import DEBER_functions as deber
from functions import mapping_functions as mf
from functions import diffusion_functions as df
from functions import reflow_functions as rf
from functions import plot_functions as pf
from functions._outdated import SE_functions as ef

mapping = importlib.reload(mapping)
deber = importlib.reload(deber)
mcf = importlib.reload(mcf)
mf = importlib.reload(mf)
df = importlib.reload(df)
ef = importlib.reload(ef)
rf = importlib.reload(rf)
pf = importlib.reload(pf)

# %%
l_x = mapping.l_x * 1e-7
l_y = mapping.l_y * 1e-7
area = l_x * l_y
d_PMMA = mapping.z_max * 1e-7  # cm
j_exp_s = 1.9e-9  # A / cm
j_exp_l = 1.9e-9 * l_x
dose_s = 0.6e-6  # C / cm^2
dose_l = dose_s * l_x
t = dose_l / j_exp_l  # 316 s
dt = 1  # s
Q = dose_s * area
n_electrons = Q / constants.e_SI  # 2 472
n_electrons_s = int(np.around(n_electrons / t))

r_beam = 150e-7

zip_length = 1000
T_C = 125
Tg = 120
dT = T_C - Tg

time_s = 10

xx = mapping.x_centers_50nm * 1e-7  # cm
zz_vac = np.zeros(len(xx))  # cm

zz_vac_list = [zz_vac]

for i in range(10000):

    print(i)

    e_DATA, e_DATA_Pn = deber.get_e_DATA_Pn(
        xx=xx,
        zz_vac=zz_vac,
        d_PMMA=d_PMMA,
        n_electrons=10,
        E0=20e+3,
        r_beam=r_beam
    )

    np.save('data/e_DATA_Pn_80nm_point/e_DATA_Pn_' + str(i) + '.npy', e_DATA_Pn)

# %%
# e_DATA_test = np.load('data/e_DATA_Pn_80nm_point/e_DATA_Pn_0.npy')

# pf.plot_e_DATA(e_DATA_test, d_PMMA=80, E_cut=5, proj='xz')
