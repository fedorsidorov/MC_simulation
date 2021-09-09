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

# from mapping import mapping_viscosity_80nm as mm
from mapping._outdated import mapping_viscosity_900nm as mm

mm = importlib.reload(mm)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
af = importlib.reload(af)
pf = importlib.reload(pf)
sf = importlib.reload(sf)
Gf = importlib.reload(Gf)

# %%
xx = mm.x_centers_5nm
zz_vac = np.zeros(len(xx))
# zz_vac = np.ones(len(xx)) * ((1 + np.cos(xx * np.pi / (lx / 2))) * d_PMMA) / 5

source = 'data/e_DATA_Pv_900nm/'

event_matrix_total = np.zeros(mm.hist_5nm_shape)
PMMA_val_matrix_total = np.zeros(mm.hist_5nm_shape)
PMMA_val_z_array_total = np.zeros(len(mm.z_centers_5nm))
scission_matrix_total = np.zeros(mm.hist_5nm_shape)

weight = 0.225
n_files = 151
n_primaries_in_file = 50

progress_bar = tqdm(total=n_files, position=0)

for file_cnt in range(n_files):

    now_e_DATA = np.load(source + 'e_DATA_' + str(file_cnt % n_files) + '.npy')

    emf.add_individual_uniform_xy_shifts_to_e_DATA(now_e_DATA, n_primaries_in_file,
                                                   [-mm.lx/2, mm.lx/2], [-mm.ly/2, mm.ly/2])

    af.snake_array(
        array=now_e_DATA,
        x_ind=ind.e_DATA_x_ind,
        y_ind=ind.e_DATA_y_ind,
        z_ind=ind.e_DATA_z_ind,
        xyz_min=[mm.x_min, mm.y_min, -np.inf],
        xyz_max=[mm.x_max, mm.y_max, np.inf]
    )

    now_e_DATA_PMMA = now_e_DATA[np.where(now_e_DATA[:, ind.e_DATA_layer_id_ind] == ind.PMMA_ind)]
    now_e_DATA_PMMA_val =\
        now_e_DATA_PMMA[np.where(now_e_DATA_PMMA[:, ind.e_DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]

    # now_e_DATA_PMMA_val = emf.delete_snaked_vacuum_events(now_e_DATA_PMMA_val, xx, zz_vac)

    event_matrix_total += np.histogramdd(
        sample=now_e_DATA[:, ind.e_DATA_coord_inds],
        bins=mm.bins_5nm
    )[0]

    PMMA_val_matrix = np.histogramdd(
        sample=now_e_DATA_PMMA_val[:, ind.e_DATA_coord_inds],
        bins=mm.bins_5nm
    )[0]

    PMMA_val_z_array = np.histogram(
        a=now_e_DATA_PMMA_val[:, ind.e_DATA_z_ind],
        bins=mm.z_bins_5nm
    )[0]

    scission_matrix = np.histogramdd(
        sample=now_e_DATA_PMMA_val[:, ind.e_DATA_coord_inds],
        bins=mm.bins_5nm,
        weights=sf.get_scissions(now_e_DATA_PMMA_val, weight=weight)
    )[0]

    PMMA_val_matrix_total += PMMA_val_matrix
    PMMA_val_z_array_total += PMMA_val_z_array
    scission_matrix_total += scission_matrix

    progress_bar.update()

# %%
event_matrix_total_2d = np.sum(event_matrix_total, axis=1)
PMMA_val_matrix_2d = np.sum(PMMA_val_matrix_total, axis=1)

plt.figure(dpi=300)
plt.imshow(event_matrix_total_2d.transpose())
# plt.imshow(PMMA_val_matrix_2d.transpose())
# plt.plot(mm.z_centers_5nm, PMMA_val_z_array_total)
plt.show()

# %%
pf.plot_e_DATA(now_e_DATA, mm.d_PMMA, xx, zz_vac, limits=[[-1000, 1000], [-100, 1000]])

# %%
# e_DATA_single = emf.get_e_id_e_DATA(now_e_DATA, n_primary_electrons_in_file, e_id=49)
# pf.plot_e_DATA(e_DATA_single, mm.d_PMMA, xx, zz_vac)

# pf.plot_e_DATA(now_e_DATA, mm.d_PMMA, xx, zz_vac, limits=[[-200, 200], [-100, 200]])
# pf.plot_e_DATA(now_e_DATA, mm.d_PMMA, xx, zz_vac, limits=[[-200, 200], [-100, 200]])

# pf.plot_e_DATA(now_e_DATA_PMMA, mm.d_PMMA, xx, zz_vac, limits=[[-100, 100], [0, 100]])
# pf.plot_e_DATA(now_e_DATA_PMMA_val, mm.d_PMMA, xx, zz_vac, limits=[[-100, 100], [0, 100]])

# %%
sci_mat = np.load('data/sci_mat_viscosity/scission_matrix_total_20.npy')
sci_mat_2d = np.sum(sci_mat, axis=1)

plt.figure(dpi=300)
plt.imshow(sci_mat_2d)
plt.show()
