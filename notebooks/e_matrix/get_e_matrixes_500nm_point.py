import importlib
import numpy as np
import matplotlib.pyplot as plt

from mapping import mapping_3um_500nm as mm
import indexes as ind
from functions import array_functions as af
from functions import e_matrix_functions as emf

af = importlib.reload(af)
emf = importlib.reload(emf)
mm = importlib.reload(mm)
ind = importlib.reload(ind)


# %%
x_step, z_step = mm.step_1nm, mm.step_1nm
x_bins, z_bins = mm.x_bins_1nm, mm.z_bins_1nm
x_centers, z_centers = mm.x_centers_1nm, mm.z_centers_1nm

scission_weight = 0.08  # 130 C - 0.082748
# scission_weight = 0.09  # 150 C - 0.088568

n_electrons_required_s = 1870  # I = 1.2 nA
n_electrons_in_file = 31
n_files_s = int(n_electrons_required_s / 2 / n_electrons_in_file)

exposure_time = 100
time_step = 1

# z0 = 100
z0 = 200

beam_sigma = 300

# n_files = 800
n_files = 250
file_cnt = 0

now_time = 0

# while now_time < exposure_time:
while now_time < 1:

    print('Now time =', now_time)

    now_val_matrix = np.zeros((len(x_centers), len(z_centers)))

    # for n in range(n_files_s):
    for n in range(100):

        now_e_DATA_Pv = np.load(
            'notebooks/DEBER_simulation/e_DATA_500nm_point/' + str(z0) + '/e_DATA_Pv_' +
            str(file_cnt % n_files) + '.npy'
        )

        file_cnt += 1
        emf.rotate_DATA(now_e_DATA_Pv)
        emf.add_gaussian_x_shift_to_e_DATA(now_e_DATA_Pv, beam_sigma)

        af.snake_array(
            array=now_e_DATA_Pv,
            x_ind=ind.e_DATA_x_ind,
            y_ind=ind.e_DATA_y_ind,
            z_ind=ind.e_DATA_z_ind,
            xyz_min=[mm.x_min, mm.y_min, -np.inf],
            xyz_max=[mm.x_max, mm.y_max, np.inf]
        )

        now_val_matrix += np.histogramdd(
            sample=now_e_DATA_Pv[:, [ind.e_DATA_x_ind, ind.e_DATA_z_ind]],
            bins=[x_bins, z_bins]
        )[0]

    now_val_matrix += now_val_matrix[::-1, :]

    # now_scission_matrix = np.zeros(np.shape(now_val_matrix), dtype=int)

    # for i, _ in enumerate(x_centers):
    #     for k, _ in enumerate(z_centers):
    #         n_val = int(now_val_matrix[i, k])
    #         scissions = np.where(np.random.random(n_val) < scission_weight)[0]
    #         now_scission_matrix[i, k] = len(scissions)

    # np.save('notebooks/DEBER_simulation/val_matrixes_1nm_sigma_' + str(beam_sigma) +
    #         'nm/val_matrix_' + str(now_time) + '.npy', now_scission_matrix)

    now_time += time_step


# %%
plt.figure(dpi=300)
plt.imshow(now_val_matrix.transpose())
plt.show()


# %%
mat_300 = np.load('notebooks/DEBER_simulation/scission_matrixes_sigma_300nm/scission_matrix_1nm_0.npy')
mat_500 = np.load('notebooks/DEBER_simulation/scission_matrixes_sigma_500nm/scission_matrix_1nm_0.npy')

x_new_inds = ((x_centers - mm.x_min) // 50).astype(int)
z_new_inds = ((z_centers - mm.z_min) // 50).astype(int)

mat_300_50nm = np.zeros((len(mm.x_centers_50nm), len(mm.z_centers_50nm)))
mat_500_50nm = np.zeros((len(mm.x_centers_50nm), len(mm.z_centers_50nm)))

for i in range(len(x_centers)):
    for k in range(len(z_centers)):

        mat_300_50nm[x_new_inds[i], z_new_inds[k]] += mat_300[i, k]
        mat_500_50nm[x_new_inds[i], z_new_inds[k]] += mat_500[i, k]

# %%
plt.figure(dpi=300)
plt.imshow(mat_500_50nm.transpose())
plt.show()

