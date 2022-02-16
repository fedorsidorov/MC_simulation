import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import importlib

from mapping import mapping_3um_500nm as mm
from functions import diffusion_functions

df = importlib.reload(diffusion_functions)
mm = importlib.reload(mm)


# %%
scission_matrix = np.load('notebooks/DEBER_simulation/test_scission_matrix.npy')

zip_length = 1000
monomer_matrix = scission_matrix * zip_length

conc_matrix_0 = monomer_matrix / (mm.step_50nm * mm.ly * mm.step_50nm * 1e-21)

plt.figure(dpi=300)
plt.imshow(conc_matrix_0.transpose())
plt.title('monomer concentration, cm$^{-3}$')
plt.colorbar()
plt.show()


# conc_matrix = deepcopy(conc_matrix_0)
D = df.get_D(150, 1)  # 2.4e-9

xx = mm.x_centers_50nm * 1e-7
zz = mm.z_centers_50nm * 1e-7

time_step = 0.1
total_time = 1

conc_matrix = df.make_simple_diffusion_sim(
    conc_matrix=conc_matrix_0,
    D=D/100,
    x_len=len(xx),
    z_len=len(zz),
    time_step=time_step,
    h_nm=mm.step_50nm,
    total_time=total_time
)

plt.figure(dpi=300)
plt.imshow(conc_matrix.transpose())
plt.title('monomer concentration, cm$^{-3}$')
plt.colorbar()
plt.show()
