import importlib
import numpy as np
import matplotlib.pyplot as plt

from mapping import mapping_5um_900nm as mm
import indexes as ind

import constants

mm = importlib.reload(mm)
ind = importlib.reload(ind)
constants = importlib.reload(constants)


# %%
lx = mm.lx * 1e-7
ly = mm.ly * 1e-7
# area = lx * ly
area = 3 * 3.9 * 1e-2

dose_factor = 3.8

exposure_time = 1024
It = 0.85e-9 * exposure_time * area  # C
n_lines = 625

pitch = 5e-4  # cm
ratio = 1.3 / 1
L_line = pitch * n_lines * ratio

It_line = It / n_lines  # C
It_line_l = It_line / L_line

y_depth = mm.ly * 1e-7  # cm

sim_dose = It_line_l * y_depth * dose_factor
n_electrons_required = sim_dose / 1.6e-19
n_electrons_required_s = int(n_electrons_required / exposure_time)  # 1870.77

n_electrons_in_file = 93  # sovpadenie ?


# %%
x_step, z_step = mm.step_1nm, mm.step_1nm
x_bins, z_bins = mm.x_bins_1nm, mm.z_bins_1nm
x_centers, z_centers = mm.x_centers_1nm, mm.z_centers_1nm

scission_weight = 0.09  # 150 C - 0.08856, 160 C - 0.09142

n_files = 470

now_time = 0

while now_time < exposure_time:

    print('Now time =', now_time)

    now_val_matrix = np.zeros((len(x_centers), len(z_centers)))

    for n in range(2):

        now_e_DATA_Pv = np.load(
            'notebooks/DEBER_simulation/e_DATA_Pv_900nm_300nm/e_DATA_Pv_' +
            str((now_time * 2 + n) % n_files) + '.npy'
        )

        now_val_matrix += np.histogramdd(
            sample=now_e_DATA_Pv[:, [ind.e_DATA_x_ind, ind.e_DATA_z_ind]],
            bins=[x_bins, z_bins]
        )[0]

    now_val_matrix += now_val_matrix[::-1, :]

    now_scission_matrix = np.zeros(np.shape(now_val_matrix), dtype=int)

    for i, _ in enumerate(x_centers):
        for k, _ in enumerate(z_centers):
            n_val = int(now_val_matrix[i, k])
            scissions = np.where(np.random.random(n_val) < scission_weight)[0]
            now_scission_matrix[i, k] = len(scissions)

    np.save(
        'notebooks/DEBER_simulation/scission_matrixes_sigma_300nm_900/scission_matrix_1nm_' + str(now_time) + '.npy',
        now_scission_matrix
    )

    now_time += time_step


# %% 73451
plt.figure(dpi=300)
plt.imshow(now_scission_matrix.transpose())
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

