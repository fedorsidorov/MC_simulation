# %% Import
import numpy as np
import os
import importlib

# %%
# E0_list = [50, 100, 150, 200, 300, 350, 400, 450, 500, 600, 800, 1000]
# E0_list = [100, 200, 300, 400, 1000]
E0_list = [1000]

model = 'Mermin'

# n_files = 1000
# n_tracks = 100

n_files = 1
n_tracks = 1

num = 0

while num < n_files:

    for E0 in E0_list:

        DATA = mcf.get_DATA(E0, n_tracks)

        inds = np.where(np.logical_and(DATA[:, 5] < 0, DATA[:, 7] < 50))
        n_2ndaries = len(inds[0])

        dest_dir = os.path.join(mc.sim_folder, 'e_DATA', '2ndaries', model, str(E0))

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        fname = os.path.join(dest_dir, 'n_2nd_for_100_prim_tracks_' + str(num) + '.npy')
        # np.save(fname, np.array((n_2ndaries)))

        print('file ' + fname + ' is ready')

    num += 1

# %%
DATA = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/e_DATA/DATA_0.npy')

pd.plot_DATA(DATA, 4, 6)




