import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
paper_plot = np.loadtxt('notebooks/MC_Si_check/2ndary_spectra.txt')
paper_x = paper_plot[:, 0]
paper_y = paper_plot[:, 1]

# %%
n_files = 330
n_bins = 50
hist = np.zeros(n_bins - 1)
bins = np.linspace(0, 4, n_bins)

# %%
progress_bar = tqdm(total=n_files, position=0)

for i in range(n_files):
    now_data = np.load('data/Si_10keV/e_DATA_' + str(i) + '.npy')
    e_2nd_arr = now_data[np.where(now_data[:, 8] != 0)[0], 8]
    hist += np.histogram(e_2nd_arr / 1000, bins=bins, normed=True)[0]
    progress_bar.update()

bin_centrers = (bins[:-1] + bins[1:])/2

# %%
plt.figure(dpi=300)
plt.semilogy(bin_centrers, hist, 'ro')
plt.plot(paper_x, paper_y * 150)

plt.xlim(0, 4)
plt.show()

# %% check range
n_bins = 10

# bins = np.linspace(10, 10000, n_bins + 1)
bins = np.logspace(2, 4, n_bins + 1)

hist = np.zeros(n_bins)
hist_n = np.zeros(n_bins)

progress_bar = tqdm(total=1000, position=0)

total_range = 0
n_electrons = 0

for i in range(330):
    now_data = np.load('data/Si_10keV_old/e_DATA_' + str(i) + '.npy')

    n_electrons = int(np.max(now_data[:, 0]))
    # n_electrons = 1

    for n_el in range(n_electrons):
        now_e_data = now_data[np.where(now_data[:, 0] == n_el)]

        now_e_coords = now_e_data[:, 4:7]
        # now_e_coords = now_e_data[:, 6]
        now_e_lens = np.linalg.norm(now_e_coords[1:, :] - now_e_coords[:-1, :], axis=1)
        # now_e_lens = np.abs(now_e_coords[1:] - now_e_coords[:-1])

        total_range += np.sum(now_e_lens)
        n_electrons += 1

        now_e_ranges = np.zeros(len(now_e_lens))

        for k in range(len(now_e_ranges)):
            now_e_ranges[k] = np.sum(now_e_lens[k:])

        now_e_Es = now_e_data[:-1, -1]

        hist += np.histogram(now_e_Es, bins=bins, weights=now_e_ranges)[0]
        hist_n += np.histogram(now_e_Es, bins=bins)[0]

    progress_bar.update()

# %%
paper_range = np.loadtxt('notebooks/MC_Si_check/Si_true_range.txt')
# paper_range = np.loadtxt('notebooks/MC_Si_check/Si_projected_range.txt')

bin_centrers = (bins[:-1] + bins[1:])/2

plt.figure(dpi=300)
plt.loglog(bin_centrers, hist / hist_n)
plt.loglog(paper_range[:, 0], paper_range[:, 1], 'ro')
plt.grid()
plt.show()

# %%
progress_bar = tqdm(total=n_files*2, position=0)

for i in range(n_files * 2):
    now_data = np.load('/Volumes/Transcend/Si_10keV/e_DATA_' + str(i) + '.npy')
    e_2nd_arr = now_data[np.where(now_data[:, 8] != 0)[0], 8]
    hist += np.histogram(e_2nd_arr / 1000, bins=bins, normed=True)[0]
    progress_bar.update()

bin_centrers = (bins[:-1] + bins[1:])/2

# %% check dose deposition
n_bins = 40

hist_dE = np.zeros(n_bins)
hist_ds = np.zeros(n_bins)
hist_n = np.zeros(n_bins)
# bins = np.logspace(0, 3, 21)
bins = np.linspace(1, 1500, n_bins + 1)
bin_centrers = (bins[:-1] + bins[1:])/2

progress_bar = tqdm(total=1000, position=0)

for i in range(1000):
    now_data = np.load('data/e_DATA_Si_10keV/e_DATA_' + str(i) + '.npy')

    n_electrons = int(np.max(now_data[:, 0]))
    # n_electrons = 1

    for n_el in range(n_electrons):
        now_e_data = now_data[np.where(now_data[:, 0] == n_el)]

        now_e_coords = now_e_data[:, 4:7]
        now_e_z = now_e_data[1:, 6]

        # now_e_lens = np.zeros(len(now_e_coords))
        now_e_lens = np.linalg.norm(now_e_coords[1:, :] - now_e_coords[:-1, :], axis=1)
        # now_e_lens = np.abs(now_e_coords[1:] - now_e_coords[:-1])

        now_e_dE = now_e_data[1:, 7]
        now_e_ds = now_e_lens

        hist_dE += np.histogram(now_e_z, bins=bins, weights=now_e_dE)[0]
        hist_ds += np.histogram(now_e_z, bins=bins, weights=now_e_ds)[0]

    progress_bar.update()

# %%
plt.figure(dpi=300)
# plt.loglog(bin_centrers, hist_dE)
# plt.loglog(bin_centrers, hist_ds)
# plt.loglog(bin_centrers, hist_dE / hist_n)
plt.plot(bin_centrers, hist_dE / hist_ds, 'ro')
# plt.loglog(bin_centrers, hist_ds / hist_dE)
# plt.loglog(paper_range[:, 0], paper_range[:, 1], 'ro')
plt.grid()
plt.show()

# %% new check dose deposition
bins = np.arange(1, 2000, 1)
n_bins = len(bins) - 1

hist_dE = np.zeros(n_bins)
hist_n = np.zeros(n_bins)

n_e_total = 0

bin_centrers = (bins[:-1] + bins[1:])/2

progress_bar = tqdm(total=1000, position=0)

for i in range(1000):
    now_data = np.load('data/Si_10keV_old/e_DATA_' + str(i) + '.npy')

    n_electrons = int(np.max(now_data[:, 0]))
    # n_electrons = 1

    for n_el in range(n_electrons):
        now_e_data = now_data[np.where(now_data[:, 0] == n_el)]

        # now_e_coords = now_e_data[:, 4:7]
        now_e_z = now_e_data[1:, 6]

        # now_e_lens = np.zeros(len(now_e_coords))
        # now_e_lens = np.linalg.norm(now_e_coords[1:, :] - now_e_coords[:-1, :], axis=1)
        # now_e_lens = np.abs(now_e_coords[1:] - now_e_coords[:-1])

        now_e_dE = now_e_data[1:, 7]
        # now_e_ds = now_e_lens

        hist_dE += np.histogram(now_e_z, bins=bins, weights=now_e_dE)[0]
        hist_n += np.histogram(now_e_z, bins=bins)[0]

        n_e_total += 1

    progress_bar.update()

# %%
plt.figure(dpi=300)
# plt.loglog(bin_centrers, hist_dE)
# plt.loglog(bin_centrers, hist_n)
# plt.loglog(bin_centrers, hist_dE / n_e_total)
plt.loglog(bin_centrers, hist_dE / 103 / 100)
# plt.loglog(bin_centrers, hist_ds / hist_dE)
# plt.loglog(paper_range[:, 0], paper_range[:, 1], 'ro')
plt.grid()
plt.show()



