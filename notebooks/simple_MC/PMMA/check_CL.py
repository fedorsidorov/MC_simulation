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
            now_e_z = now_e_data[1:, 5]
            now_e_dE = now_e_data[1:, 6] + now_e_data[1:, 6]
            hist_dE += np.histogram(now_e_z, bins=bins, weights=now_e_dE)[0]

        progress_bar.update()

    return bin_centrers, hist_dE / n_files / n_primaries


def get_E_dep_r(folder, n_files, n_primaries, max_z, n_bins):

    bins = np.linspace(0, max_z, n_bins + 1)

    hist_dE = np.zeros(n_bins)
    bin_centrers = (bins[:-1] + bins[1:])/2

    progress_bar = tqdm(total=n_files, position=0)

    for i in range(n_files):

        now_data = np.load(folder + '/e_DATA_' + str(i) + '.npy')

        for n_el in range(n_primaries):
            now_e_data = now_data[np.where(now_data[:, 0] == n_el)]
            now_e_r = np.sqrt(now_e_data[1:, 3]**2 + now_e_data[1:, 3]**2)
            now_e_dE = now_e_data[1:, 6] + now_e_data[1:, 6]
            hist_dE += np.histogram(now_e_r, bins=bins, weights=now_e_dE)[0]

        progress_bar.update()

    return bin_centrers, hist_dE / n_files / n_primaries


# %% Z
bin_centers, hist = get_E_dep_z('data/4CASINO/500', 100, 100, max_z=40, n_bins=20)
# bin_centers, hist = get_E_dep_z('data/4CASINO/1000', 100, 100, max_z=80, n_bins=20)
# bin_centers, hist = get_E_dep_z('data/4CASINO/10000', 100, 100, max_z=3000, n_bins=20)

hist /= np.sum(hist)

# %% Z
casino_casnati = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Casnati/0.5keV/cl_z.dat')
casino_pouchou = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Pouchou/0.5keV/cl_z.dat')
casino_powell = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Powell/0.5keV/cl_z.dat')

# casino_casnati = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Casnati/1keV/cl_z.dat')
# casino_pouchou = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Pouchou/1keV/cl_z.dat')
# casino_powell = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Powell/1keV/cl_z.dat')

# casino_casnati = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Casnati/10keV/cl_z.dat')
# casino_pouchou = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Pouchou/10keV/cl_z.dat')
# casino_powell = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Powell/10keV/cl_z.dat')

plt.figure(dpi=300)

plt.plot(casino_casnati[:, 0], casino_casnati[:, 1] / 50)
plt.plot(casino_pouchou[:, 0], casino_pouchou[:, 1] / 70)
plt.plot(casino_powell[:, 0], casino_powell[:, 1] / 50)

# plt.plot(casino_casnati[:, 0], casino_casnati[:, 1] / 30)
# plt.plot(casino_pouchou[:, 0], casino_pouchou[:, 1] / 90)
# plt.plot(casino_powell[:, 0], casino_powell[:, 1] / 80)

# plt.plot(casino_casnati[:, 0], casino_casnati[:, 1] / 40)
# plt.plot(casino_pouchou[:, 0], casino_pouchou[:, 1] / 110)
# plt.plot(casino_powell[:, 0], casino_powell[:, 1] / 110)

plt.plot(bin_centers, hist, 'o-')

plt.show()

# %% r
# bin_centers, hist = get_E_dep_r('data/4CASINO/500', 100, 100, max_z=40, n_bins=20)
# bin_centers, hist = get_E_dep_r('data/4CASINO/1000', 100, 100, max_z=80, n_bins=20)
bin_centers, hist = get_E_dep_r('data/4CASINO/10000', 100, 100, max_z=3000, n_bins=20)

hist /= np.sum(hist)

# %% r
# casino_casnati = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Casnati/0.5keV/cl_r.dat')
# casino_pouchou = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Pouchou/0.5keV/cl_r.dat')
# casino_powell = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Powell/0.5keV/cl_r.dat')

# casino_casnati = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Casnati/1keV/cl_r.dat')
# casino_pouchou = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Pouchou/1keV/cl_r.dat')
# casino_powell = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Powell/1keV/cl_r.dat')

casino_casnati = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Casnati/10keV/cl_r.dat')
casino_pouchou = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Pouchou/10keV/cl_r.dat')
casino_powell = np.loadtxt('notebooks/simple_MC/PMMA/distributions/Powell/10keV/cl_r.dat')

plt.figure(dpi=300)

# plt.plot(casino_casnati[:, 0], casino_casnati[:, 1] / 50)
# plt.plot(casino_pouchou[:, 0], casino_pouchou[:, 1] / 70)
# plt.plot(casino_powell[:, 0], casino_powell[:, 1] / 50)

# plt.plot(casino_casnati[:, 0], casino_casnati[:, 1] / 30)
# plt.plot(casino_pouchou[:, 0], casino_pouchou[:, 1] / 90)
# plt.plot(casino_powell[:, 0], casino_powell[:, 1] / 80)

plt.plot(casino_casnati[:, 0], casino_casnati[:, 1] / 40)
plt.plot(casino_pouchou[:, 0], casino_pouchou[:, 1] / 110)
plt.plot(casino_powell[:, 0], casino_powell[:, 1] / 110)

plt.plot(bin_centers, hist, 'o-')

plt.show()

