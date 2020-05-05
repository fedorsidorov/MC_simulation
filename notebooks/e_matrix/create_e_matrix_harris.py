import importlib

import numpy as np
from tqdm import tqdm

import indexes as ind
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import plot_functions as pf
from functions import scission_functions as sf

emf = importlib.reload(emf)
ind = importlib.reload(ind)
af = importlib.reload(af)
pf = importlib.reload(pf)
sf = importlib.reload(sf)


# %% scissions
def get_scissions(DATA, degpath_dict):
    EE = DATA[:, ind.DATA_E_dep_ind] + DATA[:, ind.DATA_E2nd_ind] + DATA[:, ind.DATA_E_ind]  # energy before collision
    scission_probs = sf.get_scission_probs(degpath_dict, E_array=EE)
    return np.array(np.random.random(len(DATA)) < scission_probs).astype(int)


# %%
l_x, l_y, l_z = 100, 100, 500
x_beg, y_beg, z_beg = -l_x / 2, -l_y / 2, 0
x_end, y_end, z_end = x_beg + l_x, y_beg + l_y, z_beg + l_z

step_2nm = 2

x_bins_2nm = np.arange(x_beg, x_end + 1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end + 1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end + 1, step_2nm)
bins_2nm = x_bins_2nm, y_bins_2nm, z_bins_2nm

# %%
len_x, len_y, len_z = len(x_bins_2nm) - 1, len(y_bins_2nm) - 1, len(z_bins_2nm) - 1

e_matrix_val_exc_sci = np.zeros((len_x, len_y, len_z))
e_matrix_val_ion_sci = np.zeros((len_x, len_y, len_z))
e_matrix_E_dep = np.zeros((len_x, len_y, len_z))

n_electrons_required = emf.get_n_electrons_2D(100, l_x, l_y)
n_electrons = 0

source = '/Users/fedor/PycharmProjects/MC_simulation/data/e_DATA/Harris/'
progress_bar = tqdm(total=n_electrons_required, position=0)

# degpaths = {"C-C2": 4}
degpaths = {"C-C2": 4, "C-C'": 2}


while n_electrons < n_electrons_required:

    n_files = 500
    file_cnt = 0
    primary_electrons_in_file = 100

    now_DATA = np.load(source + '/DATA_Pn_' + str(file_cnt % n_files) + '.npy')

    if file_cnt > n_files:
        emf.rotate_DATA(now_DATA)

    for primary_e_id in range(primary_electrons_in_file):
        now_prim_e_DATA = emf.get_e_id_DATA(now_DATA, primary_e_id)
        emf.add_uniform_xy_shift(now_prim_e_DATA, [x_beg, x_end], [y_beg, y_end])
        af.snake_array(
            array=now_prim_e_DATA,
            x_ind=ind.DATA_x_ind,
            y_ind=ind.DATA_y_ind,
            z_ind=ind.DATA_z_ind,
            xyz_min=[x_beg, y_beg, -np.inf],
            xyz_max=[x_end, y_end, np.inf]
        )

        e_matrix_E_dep += np.histogramdd(
            sample=now_prim_e_DATA[:, ind.DATA_coord_inds],
            bins=bins_2nm,
            weights=now_prim_e_DATA[:, ind.DATA_E_dep_ind]
        )[0]

        now_prim_e_val_DATA =\
            now_prim_e_DATA[np.where(now_prim_e_DATA[:, ind.DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]
        scissions = get_scissions(now_prim_e_val_DATA, degpaths)

        inds_exc = np.where(now_prim_e_val_DATA[:, ind.DATA_E2nd_ind] == 0)[0]
        inds_ion = np.where(now_prim_e_val_DATA[:, ind.DATA_E2nd_ind] > 0)[0]

        e_matrix_val_exc_sci += np.histogramdd(
            sample=now_prim_e_val_DATA[inds_exc, :][:, ind.DATA_coord_inds],
            bins=bins_2nm,
            weights=scissions[inds_exc]
        )[0]

        e_matrix_val_ion_sci += np.histogramdd(
            sample=now_prim_e_val_DATA[inds_ion, :][:, ind.DATA_coord_inds],
            bins=bins_2nm,
            weights=scissions[inds_ion]
        )[0]

        n_electrons += 1
        progress_bar.update(1)

        if n_electrons >= n_electrons_required:
            break

# %%
print(np.sum(e_matrix_val_ion_sci) / np.sum(e_matrix_E_dep) * 100)
print(np.sum(e_matrix_val_exc_sci) / np.sum(e_matrix_E_dep) * 100)

# %%
np.save('data/e_matrix/C-C2:4_C-C\':2/e_matrix_val_exc_sci.npy', e_matrix_val_exc_sci)
np.save('data/e_matrix/C-C2:4_C-C\':2/e_matrix_val_ion_sci.npy', e_matrix_val_ion_sci)
np.save('data/e_matrix/C-C2:4_C-C\':2/e_matrix_dE.npy', e_matrix_E_dep)
