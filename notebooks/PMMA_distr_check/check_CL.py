import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# %% new check dose deposition
def get_E_dep_z(folder, n_files, n_primaries, max_z, n_bins):

    bins = np.linspace(0, max_z, n_bins + 1)

    hist_dE = np.zeros(n_bins)
    bin_centrers = (bins[:-1] + bins[1:])/2

    progress_bar = tqdm(total=n_files, position=0)

    for i in range(n_files):

        now_data = np.load(folder + '/e_DATA_' + str(i) + '.npy')

        for n_el in range(n_primaries):
            now_e_data = now_data[np.where(now_data[:, 0] == n_el)]

            now_e_z = now_e_data[:, 5]
            now_e_dE = now_e_data[:, 6]

            hist_dE += np.histogram(now_e_z, bins=bins, weights=now_e_dE)[0]

        progress_bar.update()

    return bin_centrers, hist_dE / n_files / n_primaries


# %% Z
# bin_centers, hist = get_E_dep_z('data/4CASINO/500', 100, 100, max_z=40, n_bins=20)
# bin_centers, hist = get_E_dep_z('data/4CASINO/1000', 100, 100, max_z=80, n_bins=20)
bin_centers, hist = get_E_dep_z('data/4CASINO/10000', 100, 100, max_z=3000, n_bins=20)

hist /= np.sum(hist)

# %% Z
# casino = np.loadtxt('notebooks/PMMA_distr_check/distributions/CL_z_0p5keV.dat')
# casino = np.loadtxt('notebooks/PMMA_distr_check/distributions/CL_z_1keV.dat')
casino = np.loadtxt('notebooks/PMMA_distr_check/distributions/CL_z_10keV.dat')

plt.figure(dpi=300)

# plt.plot(casino[:, 0], casino[:, 1] / 2, label='CASINO')
# plt.plot(casino[:, 0], casino[:, 1] / 3, label='CASINO')
plt.plot(casino[:, 0], casino[:, 1] / 4, label='CASINO')

plt.plot(bin_centers, hist, 'o-', label='my_simulation')

# plt.title('CL_z, 500 eV')
plt.xlabel('z, nm')
plt.ylabel('hits')
# plt.xlim(0, 3000)
# plt.ylim(0, 0.12)
plt.legend()
plt.grid()

plt.show()
# plt.savefig('CL_z_10keV.jpg')


# %% r
def get_E_dep_r(folder, n_files, n_primaries, max_r, n_bins):

    bins = np.linspace(0, max_r, n_bins + 1)

    hist_dE = np.zeros(n_bins)
    bin_centrers = (bins[:-1] + bins[1:])/2

    progress_bar = tqdm(total=n_files, position=0)

    for i in range(n_files):

        now_data = np.load(folder + '/e_DATA_' + str(i) + '.npy')

        for n_el in range(n_primaries):
            now_e_data = now_data[np.where(now_data[:, 0] == n_el)]

            now_e_r = np.sqrt(now_e_data[:, 3]**2 + now_e_data[:, 4]**2)
            now_e_dE = now_e_data[:, 6]

            hist_dE += np.histogram(now_e_r, bins=bins, weights=now_e_dE)[0]

        progress_bar.update()

    return bin_centrers, hist_dE / n_files / n_primaries


# %% r
# bin_centers, hist = get_E_dep_r('data/4CASINO/500', 100, 100, max_r=20, n_bins=20)
# bin_centers, hist = get_E_dep_r('data/4CASINO/1000', 100, 100, max_r=20, n_bins=20)
bin_centers, hist = get_E_dep_r('data/4CASINO/10000', 100, 100, max_r=20, n_bins=20)

hist /= np.sum(hist)

# %% r
# casino = np.loadtxt('notebooks/PMMA_distr_check/distributions/CL_r_0p5keV.dat')
# casino = np.loadtxt('notebooks/PMMA_distr_check/distributions/CL_r_1keV.dat')
casino = np.loadtxt('notebooks/PMMA_distr_check/distributions/CL_r_10keV.dat')

plt.figure(dpi=300)

# plt.plot(casino[:, 0], casino[:, 1] / 2, label='CASINO')
# plt.plot(casino[:, 0], casino[:, 1] / 3, label='CASINO')
plt.plot(casino[:, 0], casino[:, 1] / 10, label='CASINO')

plt.plot(bin_centers, hist, 'o-')

plt.title('CL_r, 500 eV')
plt.xlabel('r, nm')
plt.ylabel('hits')

plt.xlim(0, 20)
plt.ylim(0, 1.5)

plt.grid()
plt.legend()

plt.show()
# plt.savefig('CL_r_500eV.jpg')
