import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# %% new check dose deposition
folder = '/Volumes/Transcend/NEW_e_DATA_900nm/'
n_files = 100
n_primaries_in_file = 100
x_ind, y_ind, z_ind = 4, 5, 6
E_dep_ind = 7

bins = np.arange(1, 6000, 100)
n_bins = len(bins) - 1

hist_dE = np.zeros(n_bins)
bin_centers = (bins[:-1] + bins[1:])/2

progress_bar = tqdm(total=n_files, position=0)

for i in range(n_files):

    now_data = np.load(folder + '/e_DATA_' + str(i) + '.npy')

    now_z = now_data[1:, z_ind]
    now_dE = now_data[1:, E_dep_ind]
    hist_dE += np.histogram(now_z, bins=bins, weights=now_dE)[0]

    progress_bar.update()

xx, yy = bin_centers, hist_dE / n_files / n_primaries_in_file


# %%
# xx, yy = get_E_dep('/Volumes/Transcend/NEW_e_DATA_900nm/', 100, 100, z_ind=6, E_dep_ind=7)

# %%
plt.figure(dpi=300)

plt.plot(xx, yy)
plt.xlabel('depth, nm')
plt.ylabel('Dose, eV/nm')

plt.grid()
plt.show()
# plt.savefig('E_dep.jpg')

# %%
# bns = np.load('/Volumes/Transcend/MC_Si/10keV/e_DATA_0.npy')
