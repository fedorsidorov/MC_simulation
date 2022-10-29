import importlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# import indexes as ind
from functions import array_functions as af
from functions import e_matrix_functions as emf
# from functions import plot_functions as pf
# from functions import scission_functions as sf
# from functions import G_functions as Gf

from mapping import mapping_harris as mapping

mapping = importlib.reload(mapping)
emf = importlib.reload(emf)
# ind = importlib.reload(ind)
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

# source = '/Users/fedor/PycharmProjects/MC_simulation/data/4Harris/'
# source = '/Volumes/TOSHIBA EXT/e_DATA/harris/'
source = '/Volumes/Transcend/e_DATA_500nm_point/0/'

# primary_electrons_in_file = 100
primary_electrons_in_file = 31
n_files = 600
file_cnt = 0

progress_bar = tqdm(total=n_electrons_required, position=0)

while n_electrons < n_electrons_required:

    # print(file_cnt)
    now_e_DATA = np.load(source + 'e_DATA_Pv_' + str(file_cnt % n_files) + '.npy')
    file_cnt += 1

    # check PMMA and inelastic events
    now_e_DATA = now_e_DATA[
        np.where(
            np.logical_and(now_e_DATA[:, 2] == 0, now_e_DATA[:, 3] == 1)
        )
    ]

    # if file_cnt > n_files:
    #     emf.rotate_DATA(now_e_DATA, x_ind=4, y_ind=5)

    for primary_e_id in range(primary_electrons_in_file):

        now_prim_e_DATA = emf.get_e_id_e_DATA_simple(now_e_DATA, primary_electrons_in_file, primary_e_id)

        # now_prim_e_DATA = now_prim_e_DATA[
        #     np.where(
        #         np.logical_and(now_prim_e_DATA[:, 2] == 0, now_prim_e_DATA[:, 3] == 1)
        #     )
        # ]

        if now_prim_e_DATA is None:
            print('file', file_cnt, 'e_id', primary_e_id, 'data is None')
            continue

        emf.add_uniform_xy_shift_to_e_DATA(now_prim_e_DATA,
                                           [mapping.x_min, mapping.x_max], [mapping.y_min, mapping.y_max])

        af.snake_array(
            array=now_prim_e_DATA,
            x_ind=4,
            y_ind=5,
            z_ind=6,
            xyz_min=[mapping.x_min, mapping.y_min, -np.inf],
            xyz_max=[mapping.x_max, mapping.y_max, np.inf]
        )

        e_matrix_val += np.histogramdd(
            sample=now_prim_e_DATA[:, 4:7],
            bins=mapping.bins_5nm
        )[0]

        e_matrix_E_dep += np.histogramdd(
            sample=now_prim_e_DATA[:, 4:7],
            bins=mapping.bins_5nm,
            weights=now_prim_e_DATA[:, 7]
        )[0]

        n_electrons += 1
        progress_bar.update()

        if n_electrons >= n_electrons_required:
            break

# %%
# print(np.sum(e_matrix_val) / np.sum(e_matrix_E_dep) * 100)

np.save('data/e_matrix_val_TRUE_2.npy', e_matrix_val)
np.save('data/e_matrix_E_dep_2.npy', e_matrix_E_dep)

# %%
plt.figure(dpi=300)
plt.imshow(np.sum(e_matrix_val, axis=1).transpose())
plt.show()

# %%
# ans = np.load('data/4Harris/e_DATA_Pn_19.npy')
#
# bns = ans[
#         np.where(
#             np.logical_and(ans[:, 2] == 0, ans[:, 3] == 1)
#         )
#     ]
#
# cns = emf.get_e_id_e_DATA_simple(ans, primary_electrons_in_file, 36)
