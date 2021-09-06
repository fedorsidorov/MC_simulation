import importlib
import constants
import matplotlib.pyplot as plt
import numpy as np
import constants as const
from mapping import mapping_5um_900nm as mm
from functions import array_functions as af
from functions import MC_functions as mcf
from functions import e_matrix_functions as emf
from functions import scission_functions as sf
from functions import reflow_functions as rf
from functions import SE_functions as ef
from functions import e_beam_MC as eb_MC

import indexes as ind

const = importlib.reload(const)
eb_MC = importlib.reload(eb_MC)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
mcf = importlib.reload(mcf)
mm = importlib.reload(mm)
af = importlib.reload(af)
ef = importlib.reload(ef)
sf = importlib.reload(sf)
rf = importlib.reload(rf)

# %%
sci_mat = np.load('notebooks/DEBER_simulation/vary_zz_vac/scission_matrix_46.npy')
scission_matrix = np.zeros(np.shape(sci_mat))

for i in range(47):
    now_sci_mat = np.load('notebooks/DEBER_simulation/vary_zz_vac/scission_matrix_' + str(i) + '.npy')
    scission_matrix += now_sci_mat

plt.figure(dpi=300)
plt.imshow(scission_matrix.transpose())

plt.xlabel('x, nm')
plt.ylabel('z, nm')

plt.show()

# %%
scission_array = np.sum(scission_matrix, axis=1)

xx = mm.x_centers_5nm

plt.figure(dpi=300)
plt.plot(xx, scission_array)

plt.xlabel('x, nm')
plt.ylabel('n scissions')
plt.grid()

plt.ylim(0, 8000)
plt.show()

# %%
area_exp = 1310063  # nm^2
volume_exp = area_exp * mm.ly

zip_len_average = volume_exp / const.V_mon_nm3 / np.sum(scission_matrix)

n_mon_out_array = scission_array * zip_len_average
n_mon_0_arr = np.ones(len(mm.x_centers_5nm)) * mm.step_5nm * mm.ly * mm.d_PMMA / const.V_mon_nm3
n_mon_array = n_mon_0_arr - n_mon_out_array

plt.figure(dpi=300)
plt.plot(xx, n_mon_array)

plt.xlabel('x, nm')
plt.ylabel('n monomers out')
plt.grid()

plt.show()

# %%
z_PMMA_array = n_mon_array * const.V_mon_nm3 / mm.ly / mm.step_5nm

plt.figure(dpi=300)
plt.plot(xx, z_PMMA_array)

plt.xlabel('x, nm')
plt.ylabel('z PMMA')
plt.grid()

plt.show()

# %% check
zz_vac_new = n_mon_out_array * const.V_mon_nm3 / mm.step_5nm / mm.ly

area = np.trapz(zz_vac_new, x=xx)

plt.figure(dpi=300)
plt.plot(xx, zz_vac_new)

plt.xlabel('x, nm')
plt.ylabel('zz vac out')
plt.grid()

plt.show()





