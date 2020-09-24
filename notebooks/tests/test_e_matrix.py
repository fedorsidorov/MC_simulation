import importlib

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import indexes as ind
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import scission_functions as sf
from functions import plot_functions as pf
from functions import G_functions as Gf

from mapping import mapping_viscosity as mm

mm = importlib.reload(mm)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
af = importlib.reload(af)
pf = importlib.reload(pf)
sf = importlib.reload(sf)
Gf = importlib.reload(Gf)

# %%
xx = mm.x_centers_5nm
# zz_vac = np.zeros(len(xx))
zz_vac = np.ones(len(xx)) * ((1 + np.cos(xx * np.pi / (mm.l_x / 2))) * mm.d_PMMA) / 5

source = 'data/e_DATA_test/'
scission_matrix_total = np.zeros(mm.hist_5nm_shape)
weight = 0.225
n_files = 100
primary_electrons_in_file = 10

file_cnt = 0

now_e_DATA = np.load(source + 'e_DATA_' + str(file_cnt % n_files) + '.npy')

# if file_cnt > n_files:
#     emf.rotate_DATA(now_e_DATA)

# emf.add_uniform_xy_shift_to_e_DATA(now_e_DATA, [mm.x_min, mm.x_max], [mm.y_min, mm.y_max])

# af.snake_array(
#     array=now_e_DATA,
#     x_ind=ind.DATA_x_ind,
#     y_ind=ind.DATA_y_ind,
#     z_ind=ind.DATA_z_ind,
#     xyz_min=[mm.x_min, mm.y_min, -np.inf],
#     xyz_max=[mm.x_max, mm.y_max, np.inf]
# )

now_e_val_DATA = now_e_DATA[np.where(now_e_DATA[:, ind.DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]

scissions = sf.get_scissions_easy(now_e_val_DATA, weight=weight)

scission_matrix = np.histogramdd(
    sample=now_e_val_DATA[:, ind.DATA_coord_inds],
    bins=mm.bins_5nm,
    weights=scissions
)[0]

scission_matrix_total += scission_matrix

# %%
pf.plot_e_DATA(now_e_DATA, mm.d_PMMA, xx, zz_vac, limits=[[-200, 200], [-100, 200]])

# %%
sci_mat = np.load('data/sci_mat_viscosity/scission_matrix_total_20.npy')
sci_mat_2d = np.sum(sci_mat, axis=1)

plt.figure(dpi=300)
plt.imshow(sci_mat_2d)
plt.show()


