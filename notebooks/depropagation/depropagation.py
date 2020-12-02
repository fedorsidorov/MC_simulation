import copy
import importlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import constants as const
from mapping import mapping_3p3um_80nm as mapping
from functions import diffusion_functions as df

df = importlib.reload(df)

# %%
scission_matrix = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/'
                          'e_matrix/exp_80nm_Camscan/scission_matrix.npy')
scission_matrix_sum_y = np.sum(scission_matrix, axis=1)

resist_shape = mapping.hist_2nm_shape

# D = 3.16e-6 * 1e+7 ** 2  # cm^2 / s -> nm^2 / s
# delta_t = 1e-7  # s

zip_length = 1000
n_monomers_groups = zip_length // 10

# %%
x_escape_array = np.zeros(int(np.sum(scission_matrix_sum_y) * n_monomers_groups))
pos = 0

progress_bar = tqdm(total=resist_shape[2], position=0)

for z_ind in range(resist_shape[2]):

    progress_bar.update()

    for x_ind in range(resist_shape[0]):

        n_scissions = int(scission_matrix_sum_y[x_ind, z_ind])

        if n_scissions == 0:
            continue

        xz_0 = mapping.x_bins_2nm[x_ind], mapping.z_bins_2nm[z_ind]
        l_xz = mapping.l_x, mapping.l_z

        for i in range(n_scissions):
            for n in range(n_monomers_groups):
                x_escape_array[pos] = df.track_monomer(xz_0, l_xz)
                pos += 1

# %%
mon_h_nm = const.V_mon * 1e+7 ** 3 / mapping.step_2nm ** 2

x_escape_array_corr = np.zeros(np.shape(x_escape_array))

for i, x_esc in enumerate(x_escape_array):

    while x_esc > mapping.x_max:
        x_esc -= mapping.l_x

    while x_esc < mapping.x_min:
        x_esc += mapping.l_x

    x_escape_array_corr[i] = x_esc

# %%
x_escape_hist = np.histogram(x_escape_array_corr, bins=mapping.bins_2nm[0])[0]

# %%
xx = (mapping.bins_2nm[0][:-1] + mapping.bins_2nm[0][1:]) / 2
h_final = mapping.z_max - x_escape_hist * mon_h_nm

h_final_corr = copy.deepcopy(h_final)
h_final_corr[np.where(h_final_corr < 0)] = 0

h_final_easy = mapping.z_max - np.sum(scission_matrix_sum_y, axis=1) * n_monomers_groups * mon_h_nm

plt.figure(dpi=300)
plt.plot(xx, h_final_easy)
plt.plot(xx, h_final)
plt.show()

# %%

