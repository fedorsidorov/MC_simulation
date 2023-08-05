import importlib
import numpy as np
import matplotlib.pyplot as plt

from mapping import mapping_3um_500nm as mm
import indexes as ind

mm = importlib.reload(mm)
ind = importlib.reload(ind)


# %%
x_step, z_step = mm.step_1nm, mm.step_1nm
x_bins, z_bins = mm.x_bins_1nm, mm.z_bins_1nm
x_centers, z_centers = mm.x_centers_1nm, mm.z_centers_1nm

scission_weight = 0.05  # room
# scission_weight = 0.09  # 150 C - 0.088568

exposure_time = 10
time_step = 1

n_files = 40

now_time = 0

while now_time < exposure_time:

    print('Now time =', now_time)

    now_val_matrix = np.zeros((len(x_centers), len(z_centers)))

    for n in range(10):

        now_e_DATA_Pv = np.load(
            '/Volumes/Transcend/e_DATA_Pv_snaked_sigma_300nm/e_DATA_Pv_' +
            # str((now_time * 10 * n) % n_files) + '.npy'
            str((now_time * 10 + n) % n_files) + '.npy'
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
        'notebooks/DEBER_simulation/scission_matrixes_sigma_300nm_corr/scission_matrix_1nm_' + str(now_time) + '.npy',
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

