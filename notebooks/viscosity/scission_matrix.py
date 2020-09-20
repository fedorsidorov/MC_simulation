import importlib

import numpy as np
from tqdm import tqdm

import indexes as ind
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import plot_functions as pf
from functions import scission_functions as sf
from functions import G_functions as Gf

from mapping import mapping_3p3um_80nm as mapping

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
deg_paths = sf.degpaths_all  # !!!

file_cnt = 0

n_files = 3200
primary_electrons_in_file = 10

# now_DATA = np.load(source + 'e_DATA_Pn_' + str(file_cnt % n_files) + '.npy')
# file_cnt += 1


while np.min(scission_matrix_total) < 100:

    now_DATA = np.load(source + 'e_DATA_Pn_' + str(file_cnt % n_files) + '.npy')
    file_cnt += 1

    print(np.min(scission_matrix_total))

    if file_cnt > n_files:
        emf.rotate_DATA(now_DATA)

    for primary_e_id in range(primary_electrons_in_file):

        now_prim_e_DATA = emf.get_e_id_DATA_corr(now_DATA, primary_electrons_in_file, primary_e_id)

        emf.add_uniform_xy_shift_to_track(now_prim_e_DATA,
                                          [mapping.x_min, mapping.x_max], [mapping.y_min, mapping.y_max])

        af.snake_array(
            array=now_prim_e_DATA,
            x_ind=ind.DATA_x_ind,
            y_ind=ind.DATA_y_ind,
            z_ind=ind.DATA_z_ind,
            xyz_min=[mapping.x_min, mapping.y_min, -np.inf],
            xyz_max=[mapping.x_max, mapping.y_max, np.inf]
        )

        now_prim_e_val_DATA = \
            now_prim_e_DATA[np.where(now_prim_e_DATA[:, ind.DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]

        scissions = sf.get_scissions_easy(now_prim_e_val_DATA, weight=weight)

        scission_matrix_total += np.histogramdd(
            sample=now_prim_e_val_DATA[:, ind.DATA_coord_inds],
            bins=mapping.bins_5nm,
            weights=scissions
        )[0]
