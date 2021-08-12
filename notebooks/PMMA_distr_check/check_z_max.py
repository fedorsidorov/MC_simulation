import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import constants as const
from tqdm import tqdm


# %% check range
def get_z_max(folder, n_files, n_primaries, max_z, n_bins):

    bins = np.linspace(0, max_z, n_bins + 1)

    hist_z_max = np.zeros(n_bins)
    bin_centrers = (bins[:-1] + bins[1:])/2

    progress_bar = tqdm(total=n_files, position=0)

    for n in range(n_files):

        data = np.load(folder + '/e_DATA_' + str(n) + '.npy')

        for n_pr in range(n_primaries):

            now_data = data[np.where(data[:, 0] == n_pr)]
            now_z_max = np.max(now_data[:, 5])
            # hist_z_max += np.histogram(now_z_max, bins=bins)[0]
            hist_z_max += np.histogram(now_z_max, bins=bins, weights=now_z_max)[0]

        progress_bar.update()

    return bin_centrers, hist_z_max / n_files / n_primaries


# %%
# bin_centers, hist = get_z_max('data/4CASINO/500', 100, 100, max_z=25, n_bins=26)
# bin_centers, hist = get_z_max('data/4CASINO/1000', 100, 100, max_z=80, n_bins=20)
bin_centers, hist = get_z_max('data/4CASINO/10000', 100, 100, max_z=3000, n_bins=20)

hist /= np.sum(hist)

# %%
# casino = np.loadtxt('notebooks/PMMA_distr_check/distributions/z_max_0p5keV.dat')
# casino = np.loadtxt('notebooks/PMMA_distr_check/distributions/z_max_1keV.dat')
casino = np.loadtxt('notebooks/PMMA_distr_check/distributions/z_max_10keV.dat')

plt.figure(dpi=300)

# plt.plot(casino[:, 0], casino[:, 1] * 12, label='0.5 keV')
# plt.plot(casino[:, 0], casino[:, 1] * 15, label='1 keV')
plt.plot(casino[:, 0], casino[:, 1], label='10 keV')

plt.plot(bin_centers, hist, '-o', label='my simulation')

# plt.title('z_max, 1 keV')
plt.xlabel('z, nm')
plt.ylabel('hits')
# plt.xlim(0, 80)
# plt.ylim(0, 0.2)
plt.legend()
plt.grid()

plt.show()
# plt.savefig('z_max_1keV.jpg')

