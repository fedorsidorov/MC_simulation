import importlib

import numpy as np
from tqdm import tqdm

import indexes as ind
import mapping_exp_80nm as mapping
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import plot_functions as pf
from functions import scission_functions as sf

mapping = importlib.reload(mapping)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
af = importlib.reload(af)
pf = importlib.reload(pf)
sf = importlib.reload(sf)

# %%
r_beam_nm = 5
dose_pC_cm = 6e+3
n_electrons_required = emf.get_n_electrons_1D(dose_pC_cm, mapping.x_max - mapping.x_min)

n_electrons = 0

source = '/Volumes/ELEMENTS/e_DATA/Harris/'

# deg_paths = {"C-C2": 4}
deg_paths = {"C-C2": 4, "C-C'": 2}
# deg_paths = {"C-C2": 4, "C-C'": 2, "C-C3": 1}

n_files = 500
primary_electrons_in_file = 100
file_cnt = 0

n_sweeps = 200
progress_bar = tqdm(total=n_sweeps, position=0)

for i in range(n_sweeps):

    e_matrix_val_exc_sci = np.zeros(mapping.hist_5nm_shape)
    e_matrix_val_ion_sci = np.zeros(mapping.hist_5nm_shape)
    e_matrix_E_dep = np.zeros(mapping.hist_5nm_shape)

    for _ in range(15):

        now_DATA = np.load(source + 'DATA_Pn_' + str(file_cnt % n_files) + '.npy')
        file_cnt += 1

        now_DATA = now_DATA[np.where(now_DATA[:, ind.DATA_z_ind] <= mapping.z_max)]

        if file_cnt > n_files:
            emf.rotate_DATA(now_DATA)

        for primary_e_id in range(primary_electrons_in_file):
            now_prim_e_DATA = emf.get_e_id_DATA(now_DATA, primary_e_id)

            emf.add_uniform_xy_shift_to_track(now_prim_e_DATA,
                                              [mapping.x_min, mapping.x_max], [-r_beam_nm, r_beam_nm])

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
                bins=mapping.bins_5nm,
                weights=now_prim_e_DATA[:, ind.DATA_E_dep_ind]
            )[0]

            now_prim_e_val_DATA = \
                now_prim_e_DATA[np.where(now_prim_e_DATA[:, ind.DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]
            scissions = sf.get_scissions(now_prim_e_val_DATA, deg_paths)

            inds_exc = np.where(now_prim_e_val_DATA[:, ind.DATA_E2nd_ind] == 0)[0]
            inds_ion = np.where(now_prim_e_val_DATA[:, ind.DATA_E2nd_ind] > 0)[0]

            e_matrix_val_exc_sci += np.histogramdd(
                sample=now_prim_e_val_DATA[inds_exc, :][:, ind.DATA_coord_inds],
                bins=mapping.bins_5nm,
                weights=scissions[inds_exc]
            )[0]

            e_matrix_val_ion_sci += np.histogramdd(
                sample=now_prim_e_val_DATA[inds_ion, :][:, ind.DATA_coord_inds],
                bins=mapping.bins_5nm,
                weights=scissions[inds_ion]
            )[0]

    scissions_matrix = e_matrix_val_exc_sci + e_matrix_val_ion_sci
    np.save('/Volumes/ELEMENTS/e_matrix/exp_80nm/scission_matrix_' + str(i) + '.npy', scissions_matrix)
    np.save('/Volumes/ELEMENTS/e_matrix/exp_80nm/E_dep_matrix_' + str(i) + '.npy', e_matrix_E_dep)

    progress_bar.update()
