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

scission_weight = 0.09  # 150 C - 0.088568

# exposure_time = 100
exposure_time = 1
time_step = 1

now_time = 0

while now_time < exposure_time:

    print('Now time =', now_time)

    now_val_matrix = np.zeros((len(x_centers), len(z_centers)))

    print('get val_matrix')

    for n in range(10):

        print(n)

        now_e_DATA_Pv = np.load(
            'notebooks/DEBER_simulation/e_DATA_Pv_snaked/e_DATA_Pv_' + str((now_time * 10 * n) % 628) + '.npy'
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

    now_time += time_step


# %% 73451
plt.figure(dpi=300)
plt.imshow(now_scission_matrix.transpose())
plt.show()

# %%
scission_matrix_5nm = np.zeros(mm.hist_50nm_shape)



scission_matrix_5nm = np.histogramdd(
    sample=now_scission_matrix,
    bins=[mm.x_bins_5nm, mm.z_bins_5nm],
    weights=now_scission_matrix
)[0]

