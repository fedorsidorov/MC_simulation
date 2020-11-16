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

# DATA = np.load('data/e_DATA/DATA_test.npy')
# DATA_5 = emf.get_e_id_DATA(DATA, 5)


# %%
l_xyz = np.array((100, 100, 500))
lx, ly, lz = l_xyz

x_beg, y_beg, z_beg = -lx / 2, -ly / 2, 0
xyz_beg = np.array((x_beg, y_beg, z_beg))
xyz_end = xyz_beg + l_xyz
x_end, y_end, z_end = xyz_end

step_2nm = 2

x_bins_2nm = np.arange(x_beg, x_end + 1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end + 1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end + 1, step_2nm)

bins_2nm = x_bins_2nm, y_bins_2nm, z_bins_2nm


# %% scissions
def get_scissions(degpath_dict, DATA):
    EE = DATA[:, ind.DATA_E_dep_ind] + DATA[:, ind.DATA_E2nd_ind] + DATA[:, ind.DATA_E_ind]
    scission_probs = sf.get_scission_probs(degpath_dict, E_array=EE)

    scissions = np.random.random(len(DATA)) < np.ones(len(inds)) * \
                b_map_sc[b] / sf.MMA_bonds[b][1]

    return scissions


# %%
len_x, len_y, len_z = len(x_bins_2nm) - 1, len(y_bins_2nm) - 1, len(z_bins_2nm) - 1
e_matrix_shape = len(x_bins_2nm) - 1, len(y_bins_2nm) - 1, len(z_bins_2nm) - 1

e_matrix_dE = np.zeros(e_matrix_shape)

# val_ion_list = [[[None] * (len(z_bins_2nm) - 1)] * (len(y_bins_2nm) - 1)] * (len(x_bins_2nm) - 1)
#
# valence_excitation_deque = deque(
#     deque(
#         deque(
#             deque() for k in range(len_z)
#         ) for j in range(len_y)
#     ) for i in range(len_x)
# )
#
# valence_ionization_deque = deque(
#     deque(
#         deque(
#             deque() for k in range(len_z)
#         ) for j in range(len_y)
#     ) for i in range(len_x)
# )

valence_Edep_E2nd_E_array = np.zeros((len_x, len_y, len_z, 1000, 3))
valence_inds_array = np.zeros((len_x, len_y, len_z))

# n_electrons_required = emf.get_n_electrons_2D(100, lx, ly)
n_electrons_required = 100
n_electrons = 0

##########
bond_dict_sc = {"C-C2": 4}
##########

source = '/Users/fedor/PycharmProjects/MC_simulation/data/e_DATA/harris/'
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
        emf.add_uniform_xy_shift(now_prim_e_DATA, [x_beg, x_end], [y_beg, y_end])
        af.snake_array(
            array=now_prim_e_DATA,
            x_ind=ind.DATA_x_ind,
            y_ind=ind.DATA_y_ind,
            z_ind=ind.DATA_z_ind,
            xyz_min=[x_beg, y_beg, -np.inf],
            xyz_max=[x_end, y_end, np.inf]
        )

        e_matrix_dE += np.histogramdd(
            now_prim_e_DATA[:, ind.DATA_coord_inds],
            bins=bins_2nm,
            weights=now_prim_e_DATA[:, ind.DATA_E_dep_ind]
        )[0]

        for line in now_prim_e_DATA:
            if line[ind.DATA_process_id_ind] != ind.sim_PMMA_ee_val_ind:
                continue
            hist = np.histogramdd(line[ind.DATA_coord_inds].reshape(1, 3), bins=bins_2nm)[0]
            pos_arr_arr = np.where(hist == 1)
            pos_x, pos_y, pos_z = pos_arr_arr[0][0], pos_arr_arr[1][0], pos_arr_arr[2][0]
            # E_before_collision = line[ind.DATA_E_dep_ind] + line[ind.DATA_E2nd_ind] + line[ind.DATA_E_ind]

            valence_Edep_E2nd_E_array[pos_x, pos_y, pos_z, valence_inds_array[pos_x, pos_y, pos_z], :] = \
                line[ind.DATA_E_dep_ind], line[ind.DATA_E2nd_ind], line[ind.DATA_E_ind]

            # if line[ind.DATA_E2nd_ind] > 0:
            #     valence_ionization_deque[pos_x][pos_y][pos_z].append(E_before_collision)
            # else:
            #     valence_excitation_deque[pos_x][pos_y][pos_z].append(E_before_collision)

        n_electrons += 1
        progress_bar.update(1)

        if n_electrons >= n_electrons_required:
            break

# %%
# np.save('Harris_e_matrix_val_2СС_1.0ester_2020_G.npy', e_matrix_val)
# np.save('Harris_e_matrix_dE_2СС_1.0ester_2020_G.npy', e_matrix_dE)

# np.save('Harris_e_matrix_val_3СС3_1.0ester_2020_G_abs.npy', e_matrix_val)
# np.save('Harris_e_matrix_dE_3СС3_1.0ester_2020_G_abs.npy', e_matrix_dE)


# %%
# e_matrix_val = np.load('Harris_e_matrix_val_2CC.npy')
# e_matrix_dE = np.load('Harris_e_matrix_dE_2CC.npy')

# e_matrix_val = np.load('Harris_e_matrix_val_2СС_1.0ester_2020_G_abs.npy')
# e_matrix_dE = np.load('Harris_e_matrix_dE_2СС_1.0ester_2020_G_abs.npy')

# %%

# print(np.sum(e_matrix_val) / np.sum(e_matrix_dE) * 100)

# %%
# print(get_Gs({'C-C2': 4, 'Cp-Cg': 2}))

# weights = np.linspace(0, 2, 100)
# Gs_array = np.zeros(len(weights))
#
# for i, w in enumerate(weights):
#     print(i)
#
#     Gs_array[i] = get_Gs({'C-C2': 4, 'Cp-Cg': w})
#
#     print(Gs_array[i])

# %%
# sf.get_Gs_charlesby(27)
#
# %%
# for i in np.arange(0, 2, 0.1):
#     print(i, get_Gs({'C-C2': 4, 'Cp-Cg': i}))
#
# %%
# weights = np.zeros(6)
#
# for i, t in enumerate((50, 70, 90, 110, 130, 150)):
#
#     print(t)
#
#     G_req = sf.get_Gs_charlesby(t)
#
#     l, r = 0, 2
#
#     w = 1
#
#     G_now = 0
#
#     print('required:', G_req)
#
#     while np.abs(G_now - G_req) > 0.1:
#
#         G_now = get_Gs({'C-C2': 4, 'Cp-Cg': w})
#
#         print('now:', G_now)
#
#         if G_now < G_req:
#             l = w
#             w = (w + r) / 2
#             r = r
#
#         else:
#             r = w
#             w = (w + l) / 2
#             l = l
#
#     weights[i] = w
#
#     print(w)
#
# %%
# tt = np.array((40, 50, 60, 70, 80, 90, 100, 110, mobs_120, 130, 140, 150))
# ww = np.array((0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.05, 1.2, 1.4, 1.45, 1.65))
#
# plt.plot(tt, ww)
#
# plt.xlabel('T, C')
# plt.ylabel('weight of ester group bond')
#
# plt.xlim(30, 160)
# plt.ylim(0, 2)
#
# plt.grid()
#
