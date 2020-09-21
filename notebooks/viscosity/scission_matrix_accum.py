import importlib

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import indexes as ind
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import plot_functions as pf
from functions import scission_functions as sf
from functions import G_functions as Gf

from mapping import mapping_viscosity as mapping

mapping = importlib.reload(mapping)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
af = importlib.reload(af)
pf = importlib.reload(pf)
sf = importlib.reload(sf)
Gf = importlib.reload(Gf)

# %%
source = 'data/e_DATA_Pn_80nm_point/'
scission_matrix_total = np.zeros(mapping.hist_5nm_shape)
weight = 0.225
# file_cnt = 0
n_files = 3200
primary_electrons_in_file = 10

for file_cnt in range(2500):

    now_e_DATA = np.load(source + 'e_DATA_Pn_' + str(file_cnt % n_files) + '.npy')
    # file_cnt += 1

    if file_cnt > n_files:
        emf.rotate_DATA(now_e_DATA)

    emf.add_uniform_xy_shift_to_e_DATA(now_e_DATA, [mapping.x_min, mapping.x_max], [mapping.y_min, mapping.y_max])

    af.snake_array(
        array=now_e_DATA,
        x_ind=ind.DATA_x_ind,
        y_ind=ind.DATA_y_ind,
        z_ind=ind.DATA_z_ind,
        xyz_min=[mapping.x_min, mapping.y_min, -np.inf],
        xyz_max=[mapping.x_max, mapping.y_max, np.inf]
    )

    now_e_val_DATA = now_e_DATA[np.where(now_e_DATA[:, ind.DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]

    scissions = sf.get_scissions_easy(now_e_val_DATA, weight=weight)

    scission_matrix = np.histogramdd(
        sample=now_e_val_DATA[:, ind.DATA_coord_inds],
        bins=mapping.bins_5nm,
        weights=scissions
    )[0]

    scission_matrix_total += scission_matrix

    if file_cnt % 100 == 0:
        print(file_cnt // 100, 'hundred files')
        print(np.average(scission_matrix_total))
        np.save('data/sci_mat_viscosity/scission_matrix_total_' +
                str(file_cnt // 100) + '.npy', scission_matrix_total)


# %%
sci_mat = np.load('data/sci_mat_viscosity/scission_matrix_total_20.npy')
sci_mat_2d = np.sum(sci_mat, axis=1)

plt.figure(dpi=300)
plt.imshow(sci_mat_2d)
plt.show()


