import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% check range
n_bins = 10

# bins = np.linspace(10, 10000, n_bins + 1)
bins = np.logspace(2, 4, n_bins + 1)

hist = np.zeros(n_bins)
hist_n = np.zeros(n_bins)

hist_z = np.zeros(n_bins)
hist_n_z = np.zeros(n_bins)

n_files = 50
progress_bar = tqdm(total=n_files, position=0)

for i in range(n_files):
    now_data = np.load('data/MC_Si/10keV_15eV_Epl_20eV_LSP/e_DATA_' + str(i) + '.npy')
    # now_data = np.load('data/MC_Si/10keV_15eV_Epl_20eV_GNT/e_DATA_' + str(i) + '.npy')

    n_electrons = int(np.max(now_data[:, 0]))

    for n_el in range(n_electrons):
        now_e_data = now_data[np.where(now_data[:, 0] == n_el)]

        now_E = now_e_data[0, 9]

        now_e_coords = now_e_data[:, 4:7]
        now_e_z = now_e_data[:, 6]

        now_e_lens = np.linalg.norm(now_e_coords[1:, :] - now_e_coords[:-1, :], axis=1)
        now_e_lens_z = np.abs(now_e_coords[1:, -1] - now_e_coords[:-1, -1])

        now_range = np.sum(now_e_lens)
        now_range_z = np.sum(now_e_lens_z)

        hist += np.histogram(now_E, bins=bins, weights=now_range)[0]
        hist_n += np.histogram(now_E, bins=bins)[0]

        hist_z += np.histogram(now_E, bins=bins, weights=now_range_z)[0]
        hist_n_z += np.histogram(now_E, bins=bins)[0]

    progress_bar.update()

# %%
paper_range = np.loadtxt('notebooks/MC_Si_check/Si_true_range.txt')
paper_range_z = np.loadtxt('notebooks/MC_Si_check/Si_projected_range.txt')

bin_centrers = (bins[:-1] + bins[1:])/2

plt.figure(dpi=300)

plt.loglog(bin_centrers, hist / hist_n, label='my')
plt.loglog(bin_centrers, hist_z / hist_n_z, label='my z')

plt.loglog(paper_range[:, 0], paper_range[:, 1], 'ro', label='paper')
plt.loglog(paper_range_z[:, 0], paper_range_z[:, 1], 'ro', label='paper z')

plt.xlabel('electron energy, eV')
plt.ylabel('electron range, nm')

plt.legend()
plt.grid()
plt.show()
# plt.savefig('projected_range.jpg')
