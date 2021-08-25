import importlib

import numpy as np
from tqdm import tqdm

import indexes as ind
from functions import array_functions as af
from functions import e_matrix_functions as emf
# from functions import plot_functions as pf
# from functions import scission_functions as sf
# from functions import G_functions as Gf

from mapping import mapping_harris as mapping

mapping = importlib.reload(mapping)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
af = importlib.reload(af)
# pf = importlib.reload(pf)
# sf = importlib.reload(sf)
# Gf = importlib.reload(Gf)

# %%
e_matrix_val = np.zeros(mapping.hist_5nm_shape)
e_matrix_E_dep = np.zeros(mapping.hist_5nm_shape)

# dose_uC_cm2 = 50
dose_uC_cm2 = 100
n_electrons_required = emf.get_n_electrons_2D(dose_uC_cm2, mapping.l_x, mapping.l_y)
n_electrons = 0

source = 'data/4Harris/'

file_cnt = 0
progress_bar = tqdm(total=n_electrons_required, position=0)

while n_electrons < n_electrons_required:

    n_files = 100
    primary_electrons_in_file = 100

    now_e_DATA = np.load(source + 'e_DATA_' + str(file_cnt % n_files) + '.npy')

    # check PMMA and inelastic events
    now_e_DATA = now_e_DATA[
        np.where(
            np.logical_and(now_e_DATA[:, 2] == 0, now_e_DATA[:, 3] >= 1)
        )
    ]

    file_cnt += 1

    if file_cnt > n_files:
        emf.rotate_DATA(now_e_DATA)

    for primary_e_id in range(primary_electrons_in_file):

        now_prim_e_DATA = emf.get_e_id_DATA_corr(now_e_DATA, primary_electrons_in_file, primary_e_id)

        emf.add_uniform_xy_shift_to_e_DATA(now_prim_e_DATA,
                                           [mapping.x_min, mapping.x_max], [mapping.y_min, mapping.y_max])

        af.snake_array(
            array=now_prim_e_DATA,
            x_ind=ind.DATA_x_ind,
            y_ind=ind.DATA_y_ind,
            z_ind=ind.DATA_z_ind,
            xyz_min=[mapping.x_min, mapping.y_min, -np.inf],
            xyz_max=[mapping.x_max, mapping.y_max, np.inf]
        )

        e_matrix_val_sci += np.histogramdd(
            sample=now_prim_e_val_DATA[:, ind.DATA_coord_inds],
            bins=mapping.bins_5nm
        )[0]

        e_matrix_E_dep += np.histogramdd(
            sample=now_prim_e_DATA_Pv[:, ind.DATA_coord_inds],
            bins=mapping.bins_5nm,
            weights=now_prim_e_DATA_Pv[:, ind.DATA_E_dep_ind]
        )[0]

        n_electrons += 1
        progress_bar.update()

        if n_electrons >= n_electrons_required:
            break

print(np.sum(e_matrix_val_sci) / np.sum(e_matrix_E_dep) * 100)

np.save('data/scission_mat_weight/3/e_matrix_scissions_' + str(weight) + '.npy', e_matrix_val_sci)
np.save('data/scission_mat_weight/3/e_matrix_dE_' + str(weight) + '.npy', e_matrix_E_dep)

# %%
# print(np.sum(e_matrix_val_sci) / np.sum(e_matrix_E_dep) * 100)

# np.save('data/choi_weight/e_matrix_val_ion_sci_' + str(weight) + '.npy', e_matrix_val_sci)
# np.save('data/choi_weight/e_matrix_dE_' + str(weight) + '.npy', e_matrix_E_dep)
