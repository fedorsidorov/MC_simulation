import importlib

import numpy as np
from tqdm import tqdm

import indexes as ind
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import plot_functions as pf
from functions import scission_functions as sf
from mapping import mapping_harris as mapping

mapping = importlib.reload(mapping)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
af = importlib.reload(af)
pf = importlib.reload(pf)
sf = importlib.reload(sf)

# %%
e_matrix_val_exc_sci = np.zeros(mapping.hist_2nm_shape)
e_matrix_val_ion_sci = np.zeros(mapping.hist_2nm_shape)
e_matrix_E_dep = np.zeros(mapping.hist_2nm_shape)

n_electrons_required = emf.get_n_electrons_2D(100, mapping.l_x, mapping.l_y)
n_electrons = 0

source = '/Users/fedor/PycharmProjects/MC_simulation/data/e_DATA/Harris/'

deg_paths = {"C-C2": 4}
# deg_paths = {"C-C2": 4, "C-C'": 2}
# deg_paths = {"C-C2": 4, "C-C'": 2, "C-C3": 1}

progress_bar = tqdm(total=n_electrons_required, position=0)

while n_electrons < n_electrons_required:

    n_files = 500
    file_cnt = 0
    primary_electrons_in_file = 100

    now_DATA = np.load(source + '/DATA_Pn_' + str(file_cnt % n_files) + '.npy')

    if file_cnt > n_files:
        emf.rotate_DATA(now_DATA)

    for primary_e_id in range(primary_electrons_in_file):
        now_prim_e_DATA = emf.get_e_id_DATA(now_DATA, primary_e_id)

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

        e_matrix_E_dep += np.histogramdd(
            sample=now_prim_e_DATA[:, ind.DATA_coord_inds],
            bins=mapping.bins_2nm,
            weights=now_prim_e_DATA[:, ind.DATA_E_dep_ind]
        )[0]

        now_prim_e_val_DATA =\
            now_prim_e_DATA[np.where(now_prim_e_DATA[:, ind.DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]
        scissions = sf.get_scissions(now_prim_e_val_DATA, deg_paths)

        inds_exc = np.where(now_prim_e_val_DATA[:, ind.DATA_E2nd_ind] == 0)[0]
        inds_ion = np.where(now_prim_e_val_DATA[:, ind.DATA_E2nd_ind] > 0)[0]

        e_matrix_val_exc_sci += np.histogramdd(
            sample=now_prim_e_val_DATA[inds_exc, :][:, ind.DATA_coord_inds],
            bins=mapping.bins_2nm,
            weights=scissions[inds_exc]
        )[0]

        e_matrix_val_ion_sci += np.histogramdd(
            sample=now_prim_e_val_DATA[inds_ion, :][:, ind.DATA_coord_inds],
            bins=mapping.bins_2nm,
            weights=scissions[inds_ion]
        )[0]

        n_electrons += 1
        progress_bar.update()

        if n_electrons >= n_electrons_required:
            break

# %%
print(np.sum(e_matrix_val_ion_sci) / np.sum(e_matrix_E_dep) * 100)
print(np.sum(e_matrix_val_exc_sci) / np.sum(e_matrix_E_dep) * 100)

# %%
np.save('data/e_matrix/Harris/C-C2:4_C-C\':2_C-C3:1/e_matrix_val_exc_sci.npy', e_matrix_val_exc_sci)
np.save('data/e_matrix/Harris/C-C2:4_C-C\':2_C-C3:1/e_matrix_val_ion_sci.npy', e_matrix_val_ion_sci)
np.save('data/e_matrix/Harris/C-C2:4_C-C\':2_C-C3:1/e_matrix_dE.npy', e_matrix_E_dep)
