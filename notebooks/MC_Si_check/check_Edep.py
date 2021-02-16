import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% new check dose deposition
bins = np.arange(1, 2000, 1)
n_bins = len(bins) - 1

# hist_dE_100 = np.zeros(n_bins)
# hist_dE_1 = np.zeros(n_bins)
hist_dE_10 = np.zeros(n_bins)

n_e_total = 0

bin_centrers = (bins[:-1] + bins[1:])/2

n_files = 50
progress_bar = tqdm(total=n_files, position=0)

for i in range(n_files):

    # now_data = np.load('data/Si_100eV/e_DATA_' + str(i) + '.npy')
    # now_data = np.load('data/Si_1keV/e_DATA_' + str(i) + '.npy')
    now_data = np.load('data/MC_Si/10keV_15eV_Epl_20eV_LSP/e_DATA_' + str(i) + '.npy')
    # now_data = np.load('data/MC_Si/10keV_15eV_Epl_20eV_GNT/e_DATA_' + str(i) + '.npy')

    n_electrons = int(np.max(now_data[:, 0]))

    for n_el in range(n_electrons):
        now_e_data = now_data[np.where(now_data[:, 0] == n_el)]
        now_e_z = now_e_data[1:, 6]
        now_e_dE = now_e_data[1:, 7]

        # hist_dE_100 += np.histogram(now_e_z, bins=bins, weights=now_e_dE)[0]
        # hist_dE_1 += np.histogram(now_e_z, bins=bins, weights=now_e_dE)[0]
        hist_dE_10 += np.histogram(now_e_z, bins=bins, weights=now_e_dE)[0]

        n_e_total += 1

    progress_bar.update()

# %%
paper_100eV = np.loadtxt('notebooks/MC_Si_check/Si_Edep_100eV.txt')
paper_1keV = np.loadtxt('notebooks/MC_Si_check/Si_Edep_1keV.txt')
paper_10keV = np.loadtxt('notebooks/MC_Si_check/Si_Edep_10keV.txt')

plt.figure(dpi=300)

# plt.loglog(bin_centrers, hist_dE_100 / 100 / 100, label='my 100 eV')
# plt.loglog(bin_centrers, hist_dE_1 / 100 / 100, label='my 1 keV')
plt.loglog(bin_centrers, hist_dE_10 / 31 / 100, label='my 10 keV')

# plt.loglog(paper_100eV[:, 0], paper_100eV[:, 1], 'o', label='paper 100 eV')
# plt.loglog(paper_1keV[:, 0], paper_1keV[:, 1], 'o', label='paper 1 keV')
plt.loglog(paper_10keV[:, 0], paper_10keV[:, 1], 'o', label='paper 10 keV')

plt.xlim(1e+0, 5e+3)
plt.ylim(1e-2, 1e+3)

plt.xlabel('depth, nm')
plt.ylabel('dose, eV/nm')

plt.legend()
plt.grid()
plt.show()
# plt.savefig('E_dep.jpg')
