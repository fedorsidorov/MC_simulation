import importlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import constants as const
import indexes as ind
import mapping_exp_80nm_Camscan as mapping
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
r_beam_nm = 100
delta_y_cm = (mapping.y_max - mapping.y_min) * 1e-7

line_pitch_cm = 3.3e-4
line_area_cm2_per_cm = line_pitch_cm * 1

dose_uC_cm2 = 0.6
dose_pC_cm = dose_uC_cm2 * line_area_cm2_per_cm * 1e+6  # uC -> pC

sweep_time = 20e-3
j_beam_A_cm2 = 1.9e-9
dose_per_sweep_per_cm = j_beam_A_cm2 * line_area_cm2_per_cm * sweep_time
n_electrons_per_sweep_per_cm = dose_per_sweep_per_cm / const.e_SI
n_electrons_per_sweep = n_electrons_per_sweep_per_cm * delta_y_cm

time_for_10_electrons = 10 / n_electrons_per_sweep * sweep_time

# n_electrons_required = emf.get_n_electrons_1D(dose_pC_cm, mapping.y_max - mapping.y_min)
n_electrons_required = int(np.round(line_area_cm2_per_cm * 20e-7 * 0.6e-6 / const.e_SI))
n_electrons = 0

# %%
source = '/Volumes/ELEMENTS/e_DATA/20_keV_80nm/'

# deg_paths = {"C-C2": 4}
# deg_paths = {"C-C2": 4, "C-C'": 2}
# deg_paths = {"C-C2": 4, "C-C'": 2, "C-C3": 1}
deg_paths = sf.degpaths_all_WO_Oval

n_files = 500
primary_electrons_in_file = 100
file_cnt = 0

# e_matrix_val_exc_sci = np.zeros(mapping.hist_2nm_shape)
# e_matrix_val_ion_sci = np.zeros(mapping.hist_2nm_shape)
e_matrix_val_sci = np.zeros(mapping.hist_2nm_shape)
e_matrix_E_dep = np.zeros(mapping.hist_2nm_shape)

weight = 0.35
n_files_required = 25
progress_bar = tqdm(total=n_files_required, position=0)

for _ in range(n_files_required):

    now_DATA = np.load(source + 'DATA_Pn_' + str(file_cnt % n_files) + '.npy')
    file_cnt += 1

    now_DATA = now_DATA[np.where(now_DATA[:, ind.DATA_z_ind] <= mapping.z_max)]

    if file_cnt > n_files:
        emf.rotate_DATA(now_DATA)

    for primary_e_id in range(primary_electrons_in_file):
        now_prim_e_DATA = emf.get_e_id_DATA(now_DATA, primary_e_id)

        # scanning along X axis !!!
        emf.add_gaussian_xy_shift_to_track(now_prim_e_DATA, 0, r_beam_nm, [mapping.y_min, mapping.y_max])

        # af.snake_array(
        #     array=now_prim_e_DATA,
        #     x_ind=ind.DATA_x_ind,
        #     y_ind=ind.DATA_y_ind,
        #     z_ind=ind.DATA_z_ind,
        #     xyz_min=[mapping.x_min, mapping.y_min, -np.inf],
        #     xyz_max=[mapping.x_max, mapping.y_max, np.inf]
        # )

        e_matrix_E_dep += np.histogramdd(
            sample=now_prim_e_DATA[:, ind.DATA_coord_inds],
            bins=mapping.bins_2nm,
            weights=now_prim_e_DATA[:, ind.DATA_E_dep_ind]
        )[0]

        now_prim_e_val_DATA = \
            now_prim_e_DATA[np.where(now_prim_e_DATA[:, ind.DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]

        scissions = sf.get_scissions(now_prim_e_val_DATA, deg_paths, weight=weight)

        e_matrix_val_sci += np.histogramdd(
            sample=now_prim_e_val_DATA[:, ind.DATA_coord_inds],
            bins=mapping.bins_2nm,
            weights=scissions
        )[0]

    progress_bar.update()

# scissions_matrix = e_matrix_val_exc_sci + e_matrix_val_ion_sci

# %%
y_avg_matrix = np.average(e_matrix_val_sci, axis=1)

# plt.figure(dpi=300)
# plt.imshow(y_avg_matrix[700:-700, :].transpose())
# plt.show()

np.save('data/e_matrix/exp_80nm_Camscan/scission_matrix.npy', e_matrix_val_sci)
np.save('data/e_matrix/exp_80nm_Camscan/E_dep_matrix.npy', e_matrix_E_dep)
