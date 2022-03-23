import numpy as np
import matplotlib.pyplot as plt
import importlib

from mapping import mapping_3um_500nm as mm
import indexes as ind
from functions import array_functions as af

af = importlib.reload(af)
mm = importlib.reload(mm)
ind = importlib.reload(ind)


# %%
x_step, z_step = 100, 5
xx_bins, zz_bins = mm.x_bins_50nm, mm.z_bins_5nm
xx_centers, zz_centers = mm.x_centers_50nm, mm.z_centers_5nm

zz_vac_bins = np.zeros(len(xx_bins))
zz_vac_centers = np.zeros(len(xx_centers))
zz_vac_center_inds = np.zeros(len(xx_centers)).astype(int)

zz_inner_centers = np.zeros(len(xx_centers))

ratio_arr = np.zeros(len(xx_centers))

hist = np.zeros((len(xx_centers), len(zz_centers)))


for i in range(600):

    now_e_DATA = np.load('/Volumes/Transcend/e_DATA_500nm_point/50/e_DATA_Pv_' + str(i) + '.npy')

    now_e_DATA[:, ind.e_DATA_x_ind] += np.random.normal(loc=0, scale=300)

    af.snake_array(
        array=now_e_DATA,
        x_ind=ind.e_DATA_x_ind,
        y_ind=ind.e_DATA_y_ind,
        z_ind=ind.e_DATA_z_ind,
        xyz_min=[mm.x_min, mm.y_min, -np.inf],
        xyz_max=[mm.x_max, mm.y_max, np.inf]
    )

    hist += np.histogramdd(
        sample=now_e_DATA[:, [ind.e_DATA_x_ind, ind.e_DATA_z_ind]],
        bins=[xx_bins, zz_bins]
    )[0]

# %%
plt.figure(dpi=300)
plt.plot(xx_centers, np.sum(hist, axis=1))
# plt.imshow(hist)
plt.grid()
plt.show()

# %%
mat_old = np.load('notebooks/DEBER_simulation/total_scission_matrix_old.npy')
mat_new = np.load('notebooks/DEBER_simulation/total_scission_matrix_new.npy')

plt.figure(dpi=300)
plt.plot(xx_centers, np.sum(mat_old, axis=1))
plt.plot(xx_centers, np.sum(mat_new, axis=1))
plt.grid()
plt.show()


